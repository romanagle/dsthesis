#!/usr/bin/env python3
"""
plot_ism_logos_known_IEs.py
---------------------------
Same as plot_ism_logos.py but highlights known identity element positions
from known_IEs.csv on the x-axis in red/bold, per isotype.

Other positions are shown only at Sprinzl multiples of 5 (small, black).
Everything else is hidden to reduce crowding.

Usage
-----
    python dsthesis/scripts/figures/plot_ism_logos_known_IEs.py
    python dsthesis/scripts/figures/plot_ism_logos_known_IEs.py --isotypes Ala Gly
    python dsthesis/scripts/figures/plot_ism_logos_known_IEs.py --emb-dir embeddings/gnn
"""

from __future__ import annotations

import argparse
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logomaker

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--emb-dir",    default="embeddings/gnn")
parser.add_argument("--ism-dir",    default=None)
parser.add_argument("--csv",        default="data/dedup/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--known-IEs",  default="data/known_IEs.csv",
                    help="CSV of known identity elements per isotype/domain")
parser.add_argument("--outdir",     default=None,
                    help="Output directory (default: {ism-dir}/ism_logos_known_IEs)")
parser.add_argument("--isotypes",   nargs="+", default=None)
parser.add_argument("--domains",    nargs="+", default=None)
parser.add_argument("--min-sal",    type=float, default=0.002)
parser.add_argument("--top-n",      type=int,   default=None)
parser.add_argument("--ac-pos",     nargs="+",  default=["34", "35", "36", "74", "75", "76"])
parser.add_argument("--clade-col",  default="domain")
parser.add_argument("--no-cross-iso-norm", action="store_true")
args = parser.parse_args()

ism_dir = args.ism_dir or args.emb_dir
outdir  = args.outdir or os.path.join(ism_dir, "ism_logos_known_IEs")
os.makedirs(outdir, exist_ok=True)

NUCS         = ["A", "C", "G", "U"]
AC_EXCL      = set(args.ac_pos)
AC_LOOP_LBLS = {"34", "35", "36"}

# CSV isotype name → saliency file isotype name
_ISO_NAME_MAP = {"Met f": "Met"}


def _sp_is_mult5(label: str) -> bool:
    try:
        return int(label) % 5 == 0
    except (ValueError, TypeError):
        return False


def _is_integer_sp(label: str) -> bool:
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


# ── Parse known_IEs.csv ───────────────────────────────────────────────────────
def _extract_positions(text: str) -> set[str]:
    """Extract Sprinzl position labels from a free-text field."""
    positions: set[str] = set()
    if not text or text.strip() in ("…", "...", "—", "–", ""):
        return positions
    # Nucleotide letter followed by position number + optional a/b Sprinzl suffix.
    # "c" suffix (e.g. G34c = conditional IE) is NOT a Sprinzl label, so [ab]? stops there.
    for m in re.finditer(r"[ACGU](-?\d+)([ab])?", text):
        positions.add(m.group(1) + (m.group(2) or ""))
    # Handle "–73b" / "—73" style (em/en-dash before a bare number)
    for m in re.finditer(r"[–—](\d+)", text):
        positions.add(m.group(1))
    # Variable-loop labels e1, e2, …
    for m in re.finditer(r"\be(\d+)\b", text):
        positions.add(f"e{m.group(1)}")
    return positions


def load_known_IEs(csv_path: str) -> dict[str, dict[str, set[str]]]:
    """
    Returns {iso: {"Bacteria": set_of_positions, "Archaea": set_of_positions}}.
    Archaea "…" rows fall back to Bacteria positions.
    """
    known: dict[str, dict[str, set[str]]] = {}
    current_iso: str | None = None

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    for row in rows[2:]:   # skip 2 header rows
        if not any(cell.strip() for cell in row):
            continue
        while len(row) < 5:
            row.append("")

        iso_cell    = row[0].strip()
        domain_cell = row[1].strip()

        if iso_cell:
            current_iso = _ISO_NAME_MAP.get(iso_cell, iso_cell)

        if current_iso is None:
            continue

        if "(Bac)" in domain_cell:
            domain = "Bacteria"
        elif "(Arc)" in domain_cell:
            domain = "Archaea"
        else:
            continue

        all_text = " ".join(row[2:])
        pos = _extract_positions(all_text)

        if current_iso not in known:
            known[current_iso] = {}
        known[current_iso][domain] = pos

    # Archaea rows that were only "…" → fall back to Bacteria positions
    for iso, dom_dict in known.items():
        if not dom_dict.get("Archaea"):
            dom_dict["Archaea"] = dom_dict.get("Bacteria", set())

    return known


known_IEs = load_known_IEs(args.known_IEs)

# Quick sanity print
print("Known IE positions loaded:", flush=True)
for iso in sorted(known_IEs):
    for dom, pos in known_IEs[iso].items():
        print(f"  {iso}/{dom}: {sorted(pos)}", flush=True)

# ── Sprinzl map ───────────────────────────────────────────────────────────────
def parse_pairs(ss):
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == ">": stack.append(i)
        elif c == "<":
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


_sprinzl_cache: dict = {}

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


# ── Load NPZ metadata ─────────────────────────────────────────────────────────
npz_path = os.path.join(args.emb_dir, "perpos_embeddings.npz")
print(f"Loading metadata from {npz_path} …", flush=True)
data    = np.load(npz_path, allow_pickle=True)
max_len = int(data["max_len"])
classes = list(data["classes"].astype(str)) if "classes" in data else []
print(f"  max_len={max_len}  classes={classes}", flush=True)

# ── Discover saliency files ───────────────────────────────────────────────────
found: dict = {}
for fname in os.listdir(ism_dir):
    if not (fname.startswith("gnn_ism_saliency_") and fname.endswith(".npy")):
        continue
    stem = fname[len("gnn_ism_saliency_"):-len(".npy")]
    matched = False
    for iso in (classes or [stem.split("_")[0]]):
        if stem.startswith(iso + "_"):
            dom = stem[len(iso) + 1:]
            found[(iso, dom)] = os.path.join(ism_dir, fname)
            matched = True
            break
    if not matched:
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            found[(parts[0], parts[1])] = os.path.join(ism_dir, fname)

if not found:
    raise SystemExit(f"No gnn_ism_saliency files found in {ism_dir}.")

pairs = sorted(found.keys())
if args.isotypes:
    pairs = [(iso, dom) for iso, dom in pairs if iso in args.isotypes]
if args.domains:
    pairs = [(iso, dom) for iso, dom in pairs if dom in args.domains]

print(f"Plotting {len(pairs)} (isotype × domain) combinations …", flush=True)

# ── Cross-isotype mean ────────────────────────────────────────────────────────
_domain_mean_cache: dict = {}

def get_domain_mean(domain, length):
    if domain not in _domain_mean_cache:
        arrays = [np.load(path)[:length] for (_, dom), path in found.items() if dom == domain]
        _domain_mean_cache[domain] = np.mean(arrays, axis=0) if arrays else np.zeros((length, 4))
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
    rows = []
    for p in display_positions:
        if p not in active_set:
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
    highlight_pos = known_IEs.get(iso, {}).get(domain, set())

    raw_to_sp = get_sprinzl_map(iso, domain)
    if not args.no_cross_iso_norm:
        sal = sal - get_domain_mean(domain, len(sal))
    active_positions = select_positions(sal, raw_to_sp)
    if not active_positions:
        ax.text(0.5, 0.5, "no signal", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return 0

    sp_to_raw  = {v: k for k, v in raw_to_sp.items()}
    ac_raw     = sorted(sp_to_raw[lbl] for lbl in AC_LOOP_LBLS if lbl in sp_to_raw)
    disp_pos   = sorted(set(active_positions) | set(ac_raw))
    active_set = set(active_positions)
    ac_set     = set(ac_raw)

    tick_labs = [raw_to_sp.get(p, str(p)) for p in disp_pos]
    disp_pos, tick_labs = _fill_canonical_5x(disp_pos, tick_labs)

    logo_df = build_logo_df(sal, disp_pos, active_set)
    logomaker.Logo(logo_df, ax=ax, color_scheme="classic",
                   font_name="DejaVu Sans", stack_order="big_on_top",
                   baseline_width=0.5, flip_below=True)

    major_idx = [i for i, lbl in enumerate(tick_labs) if _sp_is_mult5(lbl)]
    minor_idx  = [i for i, lbl in enumerate(tick_labs)
                  if _is_integer_sp(lbl) and not _sp_is_mult5(lbl)]
    ax.set_xticks(major_idx)
    ax.set_xticklabels([tick_labs[i] for i in major_idx], fontsize=6, rotation=0, ha="center")
    ax.set_xticks(minor_idx, minor=True)
    ax.tick_params(axis='x', which='major', length=4)
    ax.tick_params(axis='x', which='minor', length=2, labelbottom=False)
    ax.axhline(0, color="black", linewidth=0.4, zorder=0)
    ax.set_xlim(-0.5, len(tick_labs) - 0.5)

    ac_cols = [i for i, p in enumerate(disp_pos) if p in ac_set]
    if ac_cols:
        ax.axvspan(ac_cols[0] - 0.5, ac_cols[-1] + 0.5,
                   color="lightgrey", alpha=0.35, zorder=0)

    return len(active_positions)


# ── Individual logos ──────────────────────────────────────────────────────────
for iso, domain in pairs:
    sal = np.load(found[(iso, domain)])[:max_len]

    raw_to_sp = get_sprinzl_map(iso, domain)
    positions = select_positions(sal, raw_to_sp)
    if not positions:
        print(f"  [{iso}/{domain}] no positions above min-sal — skipping", flush=True)
        continue

    fig_w = max(8, 2 + len(positions) * 0.25)
    fig, ax = plt.subplots(figsize=(fig_w, 3.5))
    n_pos = plot_logo(ax, sal, iso, domain)
    ax.set_ylabel("mean ISM Δlogit", fontsize=9)
    fig.tight_layout()
    stem = f"ism_logo_{iso}_{domain}"
    fig.savefig(os.path.join(outdir, f"{stem}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [{iso}/{domain}] {n_pos} positions → {stem}.png", flush=True)

# ── Per-isotype grid: domains as columns ─────────────────────────────────────
isos_present = sorted(set(iso for iso, _ in pairs))
doms_present = sorted(set(dom for _, dom in pairs))

for iso in isos_present:
    dom_cols = [dom for dom in doms_present if (iso, dom) in found]
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
    fig.savefig(os.path.join(outdir, f"{stem}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [{iso}] domain grid → {stem}.png", flush=True)

# ── All-isotypes grid ─────────────────────────────────────────────────────────
if len(isos_present) > 1 and len(doms_present) >= 1:
    N_ROWS = len(isos_present)
    N_COLS = len(doms_present)
    fig, axes = plt.subplots(N_ROWS, N_COLS,
                             figsize=(N_COLS * 7, N_ROWS * 3.2),
                             squeeze=False)
    for row, iso in enumerate(isos_present):
        for col, dom in enumerate(doms_present):
            ax = axes[row][col]
            if (iso, dom) in found:
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
    fig.savefig(os.path.join(outdir, f"{stem}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\nFull grid → {stem}.png", flush=True)

    # ── Shared-scale version ──────────────────────────────────────────────────
    fig2, axes2 = plt.subplots(N_ROWS, N_COLS,
                               figsize=(N_COLS * 7, N_ROWS * 3.2),
                               squeeze=False)
    for row, iso in enumerate(isos_present):
        for col, dom in enumerate(doms_present):
            ax = axes2[row][col]
            if (iso, dom) in found:
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

    all_ymins, all_ymaxs = [], []
    for ax in axes2.flat:
        if ax.get_visible() and (ax.lines or ax.patches):
            ylo, yhi = ax.get_ylim()
            all_ymins.append(ylo); all_ymaxs.append(yhi)
    if all_ymins:
        g_ymin, g_ymax = min(all_ymins), max(all_ymaxs)
        for ax in axes2.flat:
            if ax.get_visible():
                ax.set_ylim(g_ymin, g_ymax)

    fig2.tight_layout()
    stem2 = "ism_logo_all_isotypes_all_domains_shared_scale"
    fig2.savefig(os.path.join(outdir, f"{stem2}.png"), bbox_inches="tight", dpi=150)
    plt.close(fig2)
    print(f"Shared-scale grid → {stem2}.png", flush=True)

print(f"\nDone. Results in: {outdir}/")
