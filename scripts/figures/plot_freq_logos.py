#!/usr/bin/env python3
"""
plot_freq_logos.py
------------------
Nucleotide frequency log-odds identity logos for tRNA isotypes.

For each (isotype × domain):
  log2_odds[pos][nuc] = log2(p_iso / p_bg)

where p = (count + pseudocount) / (total + 4*pseudocount).

Positions with max |log2-odds| < min-logodds are omitted.
Anticodon (34-36) is shown as a shaded gap with no letters.
X-axis shows only Sprinzl multiples of 5 (straight, black).

Usage
-----
    python dsthesis/scripts/figures/plot_freq_logos.py
    python dsthesis/scripts/figures/plot_freq_logos.py --isotypes Ala Gly
    python dsthesis/scripts/figures/plot_freq_logos.py --domains Bacteria
    python dsthesis/scripts/figures/plot_freq_logos.py --min-logodds 0.5
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logomaker

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")
import sys
sys.path.insert(0, os.path.join(DATA_ROOT, "scripts"))
from utils.iso_classes import normalize_isotypes

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",          default=os.path.join(DATA_ROOT, "data/trnascan_GTDB_r226_dedup_taxonomy.csv"))
parser.add_argument("--outdir",       default=os.path.join(DATA_ROOT, "results/figures/freq_logos"))
parser.add_argument("--isotypes",     nargs="+", default=None)
parser.add_argument("--domains",      nargs="+", default=None)
parser.add_argument("--top-n",        type=int,   default=None,
                    help="Keep only the top-N positions by IC (default: no cap)")
parser.add_argument("--pseudocount",  type=float, default=0.5)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

NUCS        = ["A", "C", "G", "U"]
NUC_IDX     = {"A": 0, "C": 1, "G": 2, "U": 3, "T": 3}
AC_EXCL     = {"34", "35", "36", "74", "75", "76"}
AC_LOOP     = {"34", "35", "36"}


def _sp_is_mult5(label: str) -> bool:
    try:
        return int(label) % 5 == 0
    except (ValueError, TypeError):
        return False


def _is_integer_sp(label: str) -> bool:
    """True for plain-integer Sprinzl labels (1–76); False for 17a, 20a, e1, etc."""
    try:
        int(label)
        return True
    except (ValueError, TypeError):
        return False


def _fill_canonical_5x(labels: list[str]) -> list[str]:
    """Insert any missing mult-of-5 Sprinzl labels within the existing integer range."""
    sp_ints = [int(lbl) for lbl in labels if _is_integer_sp(lbl) and int(lbl) > 0]
    if not sp_ints:
        return labels
    lo, hi = min(sp_ints), max(sp_ints)
    missing = {str(n) for n in range(5, hi + 1, 5) if lo <= n <= hi and str(n) not in set(labels)}
    if not missing:
        return labels
    return sorted(set(labels) | missing, key=_sp_sort_key)


def _sp_sort_key(label: str) -> tuple:
    if label.startswith("-"):
        try: return (-1, -int(label[1:]))
        except ValueError: return (-1, 0)
    if label.startswith("e"):
        try: return (46, int(label[1:]))
        except ValueError: return (46, 0)
    base = label.rstrip("ab")
    try: return (int(base), 0 if not label[len(base):] else ord(label[len(base)]))
    except ValueError: return (50, 0)


# ── Sprinzl map ───────────────────────────────────────────────────────────────
def parse_pairs(ss: str) -> dict[int, int]:
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == ">": stack.append(i)
        elif c == "<":
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs


def build_sprinzl_map(ss: str) -> dict[int, str]:
    pairs = parse_pairs(ss)
    n     = len(ss)
    five_prime = sorted(i for i in pairs if i < pairs[i])
    stems, cur = [], []
    for p in five_prime:
        if not cur: cur = [p]
        elif p == cur[-1]+1 and pairs[p] == pairs[cur[-1]]-1: cur.append(p)
        else: stems.append(cur); cur = [p]
    if cur: stems.append(cur)
    if len(stems) < 4: return {}
    sp = {}
    acc5 = stems[0]; acc3 = sorted(pairs[p] for p in acc5)
    overhang = [p for p in range(acc5[0]) if p not in pairs]
    for i, p in enumerate(reversed(overhang)):
        sp[p] = str(-(i + 1))
    for i, p in enumerate(acc5): sp[p] = str(i+1)
    for i, p in enumerate(acc3): sp[p] = str(66+i)
    disc = sorted(p for p in range(acc3[-1]+1, n) if p not in pairs)
    for i, p in enumerate(disc): sp[p] = str(73+i)
    d5 = stems[1]
    for i, p in enumerate([p for p in range(acc5[-1]+1, d5[0]) if p not in pairs]):
        sp[p] = str(8+i)
    d3 = sorted(pairs[p] for p in d5); nd = len(d5)
    for i, p in enumerate(d5): sp[p] = str(10+i)
    for i, p in enumerate(d3): sp[p] = str(25-nd+1+i)
    d_loop = [p for p in range(d5[-1]+1, d3[0]) if p not in pairs]
    for i, p in enumerate(d_loop):
        labs = ["14","15","16","17","17a","18","19","20","20a","20b","21"]
        if i < len(labs): sp[p] = labs[i]
    ac5 = stems[2]
    lk2 = [p for p in range(d3[-1]+1, ac5[0]) if p not in pairs]
    if lk2: sp[lk2[-1]] = "26"
    ac3 = sorted(pairs[p] for p in ac5)
    for i, p in enumerate(ac5): sp[p] = str(27+i)
    for i, p in enumerate(ac3): sp[p] = str(39+i)
    for i, p in enumerate([p for p in range(ac5[-1]+1, ac3[0]) if p not in pairs]):
        sp[p] = str(32+i)
    t5 = stems[-1]; t3 = sorted(pairs[p] for p in t5)
    for i, p in enumerate(t5): sp[p] = str(49+i)
    for i, p in enumerate(t3): sp[p] = str(61+i)
    for i, p in enumerate([p for p in range(t5[-1]+1, t3[0]) if p not in pairs]):
        sp[p] = str(54+i)
    var = [p for p in range(ac3[-1]+1, t5[0])]
    if len(stems) == 5:
        for i, p in enumerate(var): sp[p] = f"e{i+1}"
    else:
        for i, p in enumerate([p for p in var if p not in pairs]): sp[p] = str(44+i)
    return sp


# ── Load CSV ──────────────────────────────────────────────────────────────────
print("Reading CSV …", flush=True)
df_all = pd.read_csv(
    args.csv,
    usecols=["primary_sequence", "secondary_structure",
             "Anticodon_predicted_isotype", "domain"],
    low_memory=False,
)
df_all = df_all.dropna(subset=["primary_sequence", "secondary_structure",
                                "Anticodon_predicted_isotype", "domain"])
df_all["primary_sequence"] = df_all["primary_sequence"].str.upper().str.replace("T", "U", regex=False)

df_all = normalize_isotypes(df_all)

# Apply domain filter to everything; isotype filter only to isotype-specific counts
if args.domains:
    df_all = df_all[df_all["domain"].isin(args.domains)]

df = df_all.copy()
if args.isotypes:
    df = df_all[df_all["Anticodon_predicted_isotype"].isin(args.isotypes)]

print(f"  {len(df_all):,} total sequences  |  {len(df):,} for selected isotypes", flush=True)

# ── Precompute Sprinzl maps for all unique secondary structures ───────────────
print("Building Sprinzl maps …", flush=True)
unique_ss = df_all["secondary_structure"].unique()
ss_to_spmap: dict[str, dict[int, str]] = {}
for ss in unique_ss:
    m = build_sprinzl_map(ss)
    if m:
        ss_to_spmap[ss] = m
print(f"  {len(ss_to_spmap):,} / {len(unique_ss):,} SS parsed successfully", flush=True)

# ── Modal secondary structure per (iso, domain) → canonical display positions ──
_modal_disp: dict[tuple, list[str]] = {}
CCA_TAIL = {"74", "75", "76"}

for (iso, dom), grp in df.groupby(["Anticodon_predicted_isotype", "domain"], sort=False):
    sub = grp.dropna(subset=["primary_sequence", "secondary_structure"])
    if sub.empty:
        continue
    modal_len = sub["primary_sequence"].str.len().mode()[0]
    modal_ss  = (sub[sub["primary_sequence"].str.len() == modal_len]
                 ["secondary_structure"].mode()[0])
    sp_map = ss_to_spmap.get(modal_ss)
    if sp_map:
        labels = sorted(
            ({lbl for lbl in sp_map.values() if lbl not in CCA_TAIL} | AC_LOOP),
            key=_sp_sort_key,
        )
        _modal_disp[(iso, dom)] = labels

# ── Count nucleotides per (iso, domain, sprinzl_label) ───────────────────────
# iso_cnts[(iso, dom)][sp_lbl] = np.array([A, C, G, U])

print("Counting nucleotides per Sprinzl position …", flush=True)

iso_cnts: dict[tuple, dict[str, np.ndarray]] = {}

for (ss, iso, dom), grp in df_all.groupby(
    ["secondary_structure", "Anticodon_predicted_isotype", "domain"], sort=False
):
    sp_map = ss_to_spmap.get(ss)
    if not sp_map:
        continue

    key = (iso, dom)
    if key not in iso_cnts:
        iso_cnts[key] = {}

    seqs = grp["primary_sequence"].tolist()

    for seq_idx, sp_lbl in sp_map.items():
        col = np.array(
            [NUC_IDX.get(s[seq_idx], -1) if seq_idx < len(s) else -1 for s in seqs],
            dtype=np.int8,
        )
        counts = np.array([(col == i).sum() for i in range(4)], dtype=np.int64)
        if counts.sum() == 0:
            continue

        if sp_lbl not in iso_cnts[key]:
            iso_cnts[key][sp_lbl] = np.zeros(4, dtype=np.int64)
        iso_cnts[key][sp_lbl] += counts

print("  Done counting.", flush=True)

# ── Frequency helper ──────────────────────────────────────────────────────────
def compute_freq(
    iso_cnt: dict[str, np.ndarray],
    pc: float,
) -> dict[str, np.ndarray]:
    """Return {sp_lbl: frequency_array (shape 4, sums to 1)} for all positions."""
    freq = {}
    for sp_lbl, iso_c in iso_cnt.items():
        total = iso_c.sum() + 4 * pc
        freq[sp_lbl] = (iso_c + pc) / total
    return freq


# ── Logo-building helper (information content) ────────────────────────────────
def _build_ic_df(
    freq: dict[str, np.ndarray],
    disp_labels: list[str],
) -> pd.DataFrame:
    rows = []
    for sp_lbl in disp_labels:
        if sp_lbl not in freq:
            rows.append([0.0, 0.0, 0.0, 0.0])
        else:
            p = freq[sp_lbl]
            with np.errstate(divide="ignore", invalid="ignore"):
                ic = 2.0 + float(np.sum(np.where(p > 0, p * np.log2(p), 0.0)))
            ic = max(0.0, ic)
            rows.append(list((p * ic).astype(float)))
    return pd.DataFrame(rows, columns=NUCS)


def plot_logo(ax, iso: str, dom: str) -> int:
    key = (iso, dom)
    if key not in iso_cnts:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return 0

    freq = compute_freq(iso_cnts[key], args.pseudocount)
    if not freq:
        ax.text(0.5, 0.5, "no signal", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return 0

    disp_labels = _modal_disp.get(key)
    if not disp_labels:
        disp_labels = sorted(
            ({sp for sp in freq.keys() if sp not in CCA_TAIL} | AC_LOOP),
            key=_sp_sort_key,
        )
    disp_labels = _fill_canonical_5x(disp_labels)
    # Blank out anticodon and CCA tail columns
    freq_display = {sp: f for sp, f in freq.items() if sp not in AC_EXCL}
    ac_cols = [i for i, lbl in enumerate(disp_labels) if lbl in AC_LOOP]

    logo_df = _build_ic_df(freq_display, disp_labels)
    logomaker.Logo(logo_df, ax=ax, color_scheme="classic",
                   font_name="DejaVu Sans", stack_order="big_on_top",
                   baseline_width=0.5, flip_below=False)

    major_idx   = [i for i, lbl in enumerate(disp_labels) if _sp_is_mult5(lbl)]
    named_idx   = [i for i, lbl in enumerate(disp_labels) if not _is_integer_sp(lbl)]
    minor_idx   = [i for i, lbl in enumerate(disp_labels)
                   if _is_integer_sp(lbl) and not _sp_is_mult5(lbl)]
    named_set   = set(named_idx)
    labeled_idx = sorted(major_idx + named_idx)
    ax.set_xticks(labeled_idx)
    ax.set_xticklabels([disp_labels[i] for i in labeled_idx], fontsize=6, rotation=45, ha="right")
    for txt, idx in zip(ax.get_xticklabels(), labeled_idx):
        if idx in named_set:
            txt.set_fontsize(4.5)
    ax.set_xticks(minor_idx, minor=True)
    ax.tick_params(axis='x', which='major', length=4)
    ax.tick_params(axis='x', which='minor', length=2, labelbottom=False)
    ax.axhline(0, color="black", linewidth=0.4, zorder=0)
    ax.set_xlim(-0.5, len(disp_labels) - 0.5)
    ax.set_ylim(0, 2.0)

    if ac_cols:
        ax.axvspan(ac_cols[0] - 0.5, ac_cols[-1] + 0.5,
                   color="lightgrey", alpha=0.35, zorder=0)

    return len(freq_display)


# ── Determine which (iso, dom) pairs to plot ──────────────────────────────────
pairs = sorted(iso_cnts.keys())
pairs = [(iso, dom) for iso, dom in pairs if iso not in {"Undet", "Sup", "Ile2", "SeC"}]
if args.isotypes:
    pairs = [(iso, dom) for iso, dom in pairs if iso in args.isotypes]
if args.domains:
    pairs = [(iso, dom) for iso, dom in pairs if dom in args.domains]

isos_present = sorted(set(iso for iso, _ in pairs))
doms_present = sorted(set(dom for _, dom in pairs))

print(f"Plotting {len(pairs)} (isotype × domain) combinations …", flush=True)

# ── Individual logos ──────────────────────────────────────────────────────────
for iso, dom in pairs:
    n_disp = len(_modal_disp.get((iso, dom)) or iso_cnts[(iso, dom)].keys())
    fig_w  = max(8, 2 + n_disp * 0.12)
    fig, ax = plt.subplots(figsize=(fig_w, 2.2))
    plot_logo(ax, iso, dom)
    ax.set_ylabel("bits", fontsize=9)
    fig.tight_layout()
    stem = f"freq_logo_{iso}_{dom}"
    fig.savefig(os.path.join(args.outdir, f"{stem}.png"), bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(args.outdir, f"{stem}.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [{iso}/{dom}] → {stem}.png/.svg", flush=True)

# ── Per-isotype grid: domains as columns ─────────────────────────────────────
for iso in isos_present:
    dom_cols = [dom for dom in doms_present if (iso, dom) in iso_cnts]
    if not dom_cols:
        continue

    fig, axes = plt.subplots(1, len(dom_cols),
                             figsize=(max(10, len(dom_cols) * 7), 2.5),
                             squeeze=False)
    for col, dom in enumerate(dom_cols):
        ax = axes[0][col]
        plot_logo(ax, iso, dom)
        ax.set_ylabel("bits", fontsize=8)
        ax.set_title(dom, fontsize=9, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    stem = f"freq_logo_{iso}_all_domains"
    fig.savefig(os.path.join(args.outdir, f"{stem}.png"), bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(args.outdir, f"{stem}.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [{iso}] domain grid → {stem}.png/.svg", flush=True)

# ── All-isotypes grid ─────────────────────────────────────────────────────────
if len(isos_present) > 1 and len(doms_present) >= 1:
    N_ROWS = len(isos_present)
    N_COLS = len(doms_present)
    fig, axes = plt.subplots(N_ROWS, N_COLS,
                             figsize=(N_COLS * 7, N_ROWS * 2.0),
                             squeeze=False)
    for row, iso in enumerate(isos_present):
        for col, dom in enumerate(doms_present):
            ax = axes[row][col]
            if (iso, dom) in iso_cnts:
                plot_logo(ax, iso, dom)
            else:
                ax.set_visible(False)
                continue
            ax.tick_params(axis="y", labelsize=6)
            if row == 0:
                ax.set_title(dom, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{iso}\nbits", fontsize=8)

    fig.tight_layout()
    stem = "freq_logo_all_isotypes_all_domains"
    fig.savefig(os.path.join(args.outdir, f"{stem}.png"), bbox_inches="tight", dpi=150)
    fig.savefig(os.path.join(args.outdir, f"{stem}.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"\nFull grid → {stem}.png/.svg", flush=True)

print(f"\nDone. Results in: {args.outdir}/")
