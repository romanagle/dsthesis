#!/usr/bin/env python3
"""
plot_ism_logos.py
-----------------
Plot signed identity element logos from ISM attribution maps saved by
train_gnn.py.  One logo per (isotype × domain) combination.

Logo style
----------
  - One positive letter (mutation increases isotype logit) above baseline
  - One negative letter (mutation decreases isotype logit) below baseline
  - Y-axis: mean ISM Δlogit (row-centred)
  - X-axis: Sprinzl position labels, anticodon (34–36) excluded; loop shown

Inputs
------
  {emb-dir}/gnn_ism_saliency_{iso}_{domain}.npy   written by train_gnn.py
  {emb-dir}/perpos_embeddings.npz                  classes, max_len
  {csv}                                            modal SS → Sprinzl map

Usage
-----
    python scripts/plot_ism_logos.py
    python scripts/plot_ism_logos.py --emb-dir embeddings/gnn
    python scripts/plot_ism_logos.py --isotypes Ala Gly
    python scripts/plot_ism_logos.py --domains Bacteria
    python scripts/plot_ism_logos.py --min-sal 0.003 --top-n 25
"""

from __future__ import annotations

import argparse
import math
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
from utils.iso_classes import MERGE_ISOS


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--emb-dir",  default="embeddings/gnn",
                    help="Directory with perpos_embeddings.npz (for max_len / classes)")
parser.add_argument("--ism-dir",  default="gnn_final",
                    help="Directory with gnn_ism_saliency_*.npy files "
                         "(default: gnn_final; use this for clade ISM outputs)")
parser.add_argument("--csv",      default=f"{DATA_ROOT}/data/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--outdir",   default=None,
                    help="Output directory (default: {ism-dir}/ism_logos)")
parser.add_argument("--isotypes", nargs="+", default=None,
                    help="Isotypes to plot (default: all found)")
parser.add_argument("--domains",  nargs="+", default=None,
                    help="Domains/clades to plot (default: all found)")
parser.add_argument("--min-sal",  type=float, default=0.002)
parser.add_argument("--top-n",    type=int,   default=None)
parser.add_argument("--ac-pos",   nargs="+",  default=["34", "35", "36", "74", "75", "76"])
parser.add_argument("--clade-col", default="domain",
                    help="CSV column used to match saliency file suffixes to sequences "
                         "(default: 'domain'; use 'order'/'family'/etc. for clade ISM)")
parser.add_argument("--no-cross-iso-norm", action="store_true",
                    help="Skip cross-isotype mean subtraction in plot_logo. "
                         "Use this when --ism-dir points to background-subtracted "
                         "saliency files (class_bg/) that are already normalized.")
args = parser.parse_args()

ism_dir = args.ism_dir or args.emb_dir
outdir  = args.outdir or os.path.join(ism_dir, "ism_logos")
os.makedirs(outdir, exist_ok=True)

NUCS         = ["A", "C", "G", "U"]
AC_EXCL      = set(args.ac_pos)
AC_LOOP_LBLS = {"34", "35", "36"}   # anticodon positions shown as empty gap
HIGHLIGHT_POS = {"3", "70", "73"}  # key identity positions: red + bold on x-axis

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


def _fill_canonical_5x(disp_pos: list, tick_labs: list) -> tuple[list, list]:
    """Insert placeholder float positions for missing mult-of-5 Sprinzl labels."""
    sp_int_map = {int(lbl): p for p, lbl in zip(disp_pos, tick_labs)
                  if _is_integer_sp(lbl) and int(lbl) > 0}
    if not sp_int_map:
        return disp_pos, tick_labs
    lo, hi = min(sp_int_map), max(sp_int_map)
    missing = [n for n in range(5, hi + 1, 5) if lo <= n <= hi and n not in sp_int_map]
    if not missing:
        return disp_pos, tick_labs
    pairs = list(zip(disp_pos, tick_labs))
    for n in missing:
        prev_n = max((m for m in sp_int_map if m < n), default=None)
        next_n = min((m for m in sp_int_map if m > n), default=None)
        if prev_n is not None and next_n is not None:
            fake = (sp_int_map[prev_n] + sp_int_map[next_n]) / 2.0
        elif prev_n is not None:
            fake = float(sp_int_map[prev_n]) + 0.5
        else:
            fake = float(sp_int_map[next_n]) - 0.5
        pairs.append((fake, str(n)))
    pairs.sort(key=lambda x: x[0])
    return [p for p, _ in pairs], [lbl for _, lbl in pairs]

# ── Sprinzl map ───────────────────────────────────────────────────────────────
def parse_pairs(ss):
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == '>': stack.append(i)
        elif c == '<':
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs


def build_sprinzl_map(ss):
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
    # 5′ overhang: unpaired nucleotides before the acceptor stem
    # (e.g. G-1 in tRNA-His, which is a key HisRS identity element)
    overhang = [p for p in range(acc5[0]) if p not in pairs]
    for i, p in enumerate(reversed(overhang)):
        sp[p] = str(-(i + 1))   # G-1, G-2, … labelled as "-1", "-2", …
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


_sprinzl_cache = {}

def get_sprinzl_map(iso, domain):
    key = (iso, domain)
    if key not in _sprinzl_cache:
        usecols = ["primary_sequence", "secondary_structure",
                   "Anticodon_predicted_isotype", args.clade_col]
        if args.clade_col != "domain":
            usecols.append("domain")
        df = pd.read_csv(args.csv, usecols=list(set(usecols)), low_memory=False)
        sub = df[(df["Anticodon_predicted_isotype"] == iso) &
                 (df[args.clade_col] == domain)].dropna(
            subset=["primary_sequence", "secondary_structure"])
        if sub.empty:
            _sprinzl_cache[key] = {}
        else:
            modal_len = sub["primary_sequence"].str.len().mode()[0]
            modal_ss  = (sub[sub["primary_sequence"].str.len() == modal_len]
                         ["secondary_structure"].mode()[0])
            _sprinzl_cache[key] = build_sprinzl_map(modal_ss)
    return _sprinzl_cache[key]

# ── Sequence counts per (iso, clade) ─────────────────────────────────────────
print("Loading sequence counts …", flush=True)
_count_df = pd.read_csv(
    args.csv,
    usecols=["Anticodon_predicted_isotype", args.clade_col],
    low_memory=False,
).dropna(subset=["Anticodon_predicted_isotype", args.clade_col])
_seq_counts: dict[tuple, int] = (
    _count_df.groupby(["Anticodon_predicted_isotype", args.clade_col])
    .size()
    .to_dict()
)

# ── Load NPZ metadata ─────────────────────────────────────────────────────────
npz_path = os.path.join(args.emb_dir, "perpos_embeddings.npz")
print(f"Loading metadata from {npz_path} …", flush=True)
data    = np.load(npz_path, allow_pickle=True)
max_len = int(data["max_len"])
classes = list(data["classes"].astype(str)) if "classes" in data else []
print(f"  max_len={max_len}  classes={classes}", flush=True)

# ── Discover saliency files ───────────────────────────────────────────────────
# Files named: gnn_ism_saliency_{iso}_{domain}.npy
# Parse by splitting on first underscore after known class names, or just
# collect all and parse the (iso, domain) pair.
found = {}   # (iso, domain) → path
for fname in os.listdir(ism_dir):
    if not (fname.startswith("gnn_ism_saliency_") and fname.endswith(".npy")):
        continue
    stem = fname[len("gnn_ism_saliency_"):-len(".npy")]
    # Try each known class as a prefix match
    matched = False
    for iso in (classes or [stem.split("_")[0]]):
        if stem.startswith(iso + "_"):
            dom = stem[len(iso) + 1:]
            found[(iso, dom)] = os.path.join(ism_dir, fname)
            matched = True
            break
    if not matched:
        # Fallback: split on last underscore
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            found[(parts[0], parts[1])] = os.path.join(ism_dir, fname)

if not found:
    raise SystemExit(
        f"No gnn_ism_saliency_{{iso}}_{{domain}}.npy files found in {ism_dir}.\n"
        "Run train_gnn.py / compute_ism_clade.py first."
    )

# Isotypes merged into a parent — exclude their saliency files from plots
_MERGED_ISOS = set(MERGE_ISOS.keys())

# Apply filters
pairs = sorted(found.keys())
pairs = [(iso, dom) for iso, dom in pairs if iso not in _MERGED_ISOS]
if args.isotypes:
    pairs = [(iso, dom) for iso, dom in pairs if iso in args.isotypes]
if args.domains:
    pairs = [(iso, dom) for iso, dom in pairs if dom in args.domains]

print(f"Plotting {len(pairs)} (isotype × domain) combinations …", flush=True)

# ── Cross-isotype mean (per domain) ──────────────────────────────────────────
_domain_mean_cache = {}

def get_domain_mean(domain, max_len):
    """Mean ISM array across all isotypes for a given domain."""
    if domain not in _domain_mean_cache:
        arrays = []
        for (iso, dom), path in found.items():
            if dom == domain:
                arrays.append(np.load(path)[:max_len])
        if arrays:
            _domain_mean_cache[domain] = np.mean(arrays, axis=0)
        else:
            _domain_mean_cache[domain] = np.zeros((max_len, 4))
    return _domain_mean_cache[domain]


# ── Logo helpers ──────────────────────────────────────────────────────────────
def select_positions(sal, raw_to_sp):
    norms = np.linalg.norm(sal, axis=1)
    keep  = [p for p in range(len(norms))
             if norms[p] >= args.min_sal
             and p in raw_to_sp
             and raw_to_sp[p] not in AC_EXCL]
    if args.top_n and len(keep) > args.top_n:
        keep = sorted(keep, key=lambda p: norms[p], reverse=True)[:args.top_n]
        keep = sorted(keep)
    return keep


def build_logo_df(sal, display_positions, active_set):
    """Build logo DataFrame; positions in display_positions but not active_set get zero rows."""
    rows = []
    for p in display_positions:
        if not isinstance(p, (int, np.integer)) or p not in active_set:
            rows.append([0.0, 0.0, 0.0, 0.0])
        else:
            row = sal[p]
            best, worst = int(np.argmax(row)), int(np.argmin(row))
            logo_row = [0.0, 0.0, 0.0, 0.0]
            if row[best]  > 0: logo_row[best]  = float(row[best])
            if row[worst] < 0: logo_row[worst] = float(row[worst])
            rows.append(logo_row)
    return pd.DataFrame(rows, columns=NUCS)


def plot_logo(ax, sal, iso, domain):
    raw_to_sp = get_sprinzl_map(iso, domain)
    if not args.no_cross_iso_norm:
        domain_mean = get_domain_mean(domain, len(sal))
        sal = sal - domain_mean
    active_positions = select_positions(sal, raw_to_sp)
    if not active_positions:
        ax.text(0.5, 0.5, "no signal", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        ax.set_title(f"{iso} / {domain}", fontsize=8)
        return 0

    # Insert anticodon gap (Sprinzl 34–36) as empty columns in the display
    sp_to_raw   = {v: k for k, v in raw_to_sp.items()}
    ac_raw      = sorted(sp_to_raw[lbl] for lbl in AC_LOOP_LBLS if lbl in sp_to_raw)
    disp_pos    = sorted(set(active_positions) | set(ac_raw))
    active_set  = set(active_positions)
    ac_set      = set(ac_raw)

    tick_labs = [raw_to_sp.get(p, str(p)) for p in disp_pos]
    disp_pos, tick_labs = _fill_canonical_5x(disp_pos, tick_labs)

    logo_df = build_logo_df(sal, disp_pos, active_set)
    logomaker.Logo(logo_df, ax=ax, color_scheme="classic",
                   font_name="DejaVu Sans", stack_order="big_on_top",
                   baseline_width=0.5, flip_below=True)
    ax.tick_params(axis='y', labelsize=7)

    major_idx   = [i for i, lbl in enumerate(tick_labs) if _sp_is_mult5(lbl)]
    named_idx   = [i for i, lbl in enumerate(tick_labs) if not _is_integer_sp(lbl)]
    minor_idx   = [i for i, lbl in enumerate(tick_labs)
                   if _is_integer_sp(lbl) and not _sp_is_mult5(lbl)]
    named_set   = set(named_idx)
    labeled_idx = sorted(major_idx + named_idx)
    ax.set_xticks(labeled_idx)
    ax.set_xticklabels([tick_labs[i] for i in labeled_idx], fontsize=6, rotation=0, ha="center")
    for txt, idx in zip(ax.get_xticklabels(), labeled_idx):
        if idx in named_set:
            txt.set_fontsize(4.5)
            txt.set_rotation(45)
            txt.set_ha("right")
    ax.set_xticks(minor_idx, minor=True)
    ax.tick_params(axis='x', which='major', length=4)
    ax.tick_params(axis='x', which='minor', length=2, labelbottom=False)
    ax.axhline(0, color="black", linewidth=0.4, zorder=0)
    ax.set_xlim(-0.5, len(tick_labs) - 0.5)

    ac_cols = [i for i, p in enumerate(disp_pos) if p in ac_set]
    if ac_cols:
        ax.axvspan(ac_cols[0] - 0.5, ac_cols[-1] + 0.5,
                   color="lightgrey", alpha=0.35, zorder=0)

    n_seq = _seq_counts.get((iso, domain), 0)
    ax.text(0.99, 0.97, f"n = {n_seq:,}", transform=ax.transAxes,
            fontsize=7, ha="right", va="top", color="#444444")

    return len(active_positions)

# ── Individual logos ──────────────────────────────────────────────────────────
signal_pairs: set[tuple] = set()   # (iso, domain) pairs that produced signal

for iso, domain in pairs:
    sal = np.load(found[(iso, domain)])[:max_len]

    raw_to_sp = get_sprinzl_map(iso, domain)
    positions = select_positions(sal, raw_to_sp)
    if not positions:
        print(f"  [{iso}/{domain}] no positions above min-sal — skipping", flush=True)
        continue

    signal_pairs.add((iso, domain))
    fig_w = max(8, 2 + len(positions) * 0.25)
    fig, ax = plt.subplots(figsize=(fig_w, 3.5))
    n_pos = plot_logo(ax, sal, iso, domain)
    ax.set_ylabel("mean ISM Δlogit", fontsize=9)
    fig.tight_layout()
    stem = f"ism_logo_{iso}_{domain}"
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [{iso}/{domain}] {n_pos} positions → {stem}.{{svg,png}}", flush=True)

# ── Per-isotype grid: domains as columns ─────────────────────────────────────
isos_present = sorted(set(iso for iso, _ in pairs))
doms_present = sorted(set(dom for _, dom in pairs))

for iso in isos_present:
    dom_cols = [dom for dom in doms_present if (iso, dom) in signal_pairs]
    if not dom_cols:
        continue

    fig, axes = plt.subplots(1, len(dom_cols),
                             figsize=(max(10, len(dom_cols) * 7), 4),
                             squeeze=False)
    for col, dom in enumerate(dom_cols):
        ax  = axes[0][col]
        sal = np.load(found[(iso, dom)])[:max_len]
        plot_logo(ax, sal, iso, dom)
        ax.set_ylabel("Δlogit", fontsize=8)
        ax.set_title(f"{dom}", fontsize=9, fontweight="bold")
        ax.tick_params(axis="y", labelsize=7)

    fig.tight_layout()
    stem = f"ism_logo_{iso}_all_domains"
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [{iso}] domain grid → {stem}.{{svg,png}}", flush=True)

# ── All-isotypes grid: isotypes as rows, domains as columns ──────────────────
signal_doms = sorted(set(dom for _, dom in signal_pairs))

if len(isos_present) > 1 and len(signal_doms) >= 1:
    N_ROWS = len(isos_present)
    N_COLS = len(signal_doms)
    fig, axes = plt.subplots(N_ROWS, N_COLS,
                             figsize=(N_COLS * 7, N_ROWS * 3.2),
                             squeeze=False)

    for row, iso in enumerate(isos_present):
        for col, dom in enumerate(signal_doms):
            ax = axes[row][col]
            if (iso, dom) in signal_pairs:
                sal = np.load(found[(iso, dom)])[:max_len]
                plot_logo(ax, sal, iso, dom)
            else:
                ax.set_visible(False)
                continue
            ax.tick_params(axis="y", labelsize=6)
            if row == 0:
                ax.set_title(dom, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{iso}\nΔlogit", fontsize=8)

    fig.tight_layout()
    stem = "ism_logo_all_isotypes_all_domains"
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(outdir, f"{stem}.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nFull grid → {stem}.{{svg,png}}", flush=True)

    # ── Shared-scale version ──────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(N_ROWS, N_COLS,
                               figsize=(N_COLS * 7, N_ROWS * 3.2),
                               squeeze=False)
    for row, iso in enumerate(isos_present):
        for col, dom in enumerate(signal_doms):
            ax = axes2[row][col]
            if (iso, dom) in signal_pairs:
                sal = np.load(found[(iso, dom)])[:max_len]
                plot_logo(ax, sal, iso, dom)
            else:
                ax.set_visible(False)
                continue
            ax.tick_params(axis="y", labelsize=6)
            if row == 0:
                ax.set_title(dom, fontsize=9, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{iso}\nΔlogit", fontsize=8)

    # Collect global y-limits from all visible axes, then re-apply uniformly
    all_ymins, all_ymaxs = [], []
    for ax in axes2.flat:
        if ax.get_visible() and ax.lines or ax.patches:
            ylo, yhi = ax.get_ylim()
            all_ymins.append(ylo)
            all_ymaxs.append(yhi)
    if all_ymins:
        g_ymin = min(all_ymins)
        g_ymax = max(all_ymaxs)
        for ax in axes2.flat:
            if ax.get_visible():
                ax.set_ylim(g_ymin, g_ymax)

    fig2.tight_layout()
    stem2 = "ism_logo_all_isotypes_all_domains_shared_scale"
    for ext in ("svg", "png"):
        fig2.savefig(os.path.join(outdir, f"{stem2}.{ext}"),
                     bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print(f"Shared-scale grid → {stem2}.{{svg,png}}", flush=True)

print(f"\nDone. Results in: {outdir}/")
