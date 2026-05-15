#!/usr/bin/env python3
"""
plot_pairwise_ism_comparison_logos.py
--------------------------------------
Pairwise ISM comparison logos showing which nucleotide mutations discriminate
one isotype over another at each Sprinzl position.

For each isotype pair (iso1, iso2) and each domain (Bacteria, Archaea):

  differential = saliency_iso1 - saliency_iso2   (shape: max_len × 4)

  - Letters above baseline = iso1-discriminative (mutation → iso1 classification)
  - Letters below baseline = iso2-discriminative (mutation → iso2 classification)

Inputs
------
  {emb-dir}/gnn_ism_saliency_{iso}_{domain}.npy   per-isotype ISM saliency maps
  {emb-dir}/perpos_embeddings.npz                  max_len, classes
  {csv}                                            modal SS → Sprinzl map

Usage
-----
    python dsthesis/scripts/figures/plot_pairwise_ism_comparison_logos.py
    python dsthesis/scripts/figures/plot_pairwise_ism_comparison_logos.py \\
        --pairs "Asp/Asn" "Glu/Gln" --domains Bacteria
    python dsthesis/scripts/figures/plot_pairwise_ism_comparison_logos.py \\
        --min-sal 0.003 --top-n 25
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

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--emb-dir", default="gnn_final",
                    help="Directory with gnn_ism_saliency_*.npy and perpos_embeddings.npz")
parser.add_argument("--csv", default="data/dedup/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--outdir", default=None,
                    help="Output directory (default: figures/ism_logos/pairwise_comparison)")
parser.add_argument("--pairs", nargs="+", default=None,
                    help="Isotype pairs as 'iso1/iso2' (default: all 5 standard pairs)")
parser.add_argument("--domains", nargs="+", default=None,
                    help="Domains to plot (default: Bacteria and Archaea)")
parser.add_argument("--min-sal", type=float, default=0.002,
                    help="Min |differential saliency| to include a position (default: 0.002)")
parser.add_argument("--top-n", type=int, default=None,
                    help="Keep only top-N most discriminative positions (default: all)")
args = parser.parse_args()

# Default 5 standard pairs
DEFAULT_PAIRS = [
    ("Asp", "Asn"),
    ("Glu", "Gln"),
    ("Tyr", "His"),
    ("Cys", "Trp"),
    ("Val", "Ile"),
]

if args.pairs:
    pairs_to_plot = []
    for p in args.pairs:
        parts = p.split("/")
        if len(parts) != 2:
            raise SystemExit(f"Invalid pair spec '{p}': use 'iso1/iso2'")
        pairs_to_plot.append((parts[0], parts[1]))
else:
    pairs_to_plot = DEFAULT_PAIRS

outdir = args.outdir or os.path.join("results", "figures", "ism_logos", "pairwise_comparison")
os.makedirs(outdir, exist_ok=True)

NUCS = ["A", "C", "G", "U"]
AC_EXCL    = {"34", "35", "36", "74", "75", "76"}
AC_LOOP    = {"34", "35", "36"}

# ── Load metadata ─────────────────────────────────────────────────────────────
npz_path = os.path.join(args.emb_dir, "perpos_embeddings.npz")
if not os.path.exists(npz_path):
    raise SystemExit(f"ERROR: {npz_path} not found")

data    = np.load(npz_path, allow_pickle=True)
max_len = int(data["max_len"])
classes = list(data["classes"].astype(str))
print(f"max_len={max_len}  classes={classes}", flush=True)

# ── Sprinzl map ───────────────────────────────────────────────────────────────
def parse_pairs_ss(ss: str) -> dict[int, int]:
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == '>': stack.append(i)
        elif c == '<':
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs


def build_sprinzl_map(ss: str) -> dict[int, str]:
    pairs = parse_pairs_ss(ss)
    n = len(ss)
    five_prime = sorted(i for i in pairs if i < pairs[i])
    stems, cur = [], []
    for p in five_prime:
        if not cur:
            cur = [p]
        elif p == cur[-1] + 1 and pairs[p] == pairs[cur[-1]] - 1:
            cur.append(p)
        else:
            stems.append(cur); cur = [p]
    if cur: stems.append(cur)
    if len(stems) < 4:
        return {}
    sp = {}
    acc5 = stems[0]
    acc3 = sorted(pairs[p] for p in acc5)
    overhang = [p for p in range(acc5[0]) if p not in pairs]
    for i, p in enumerate(reversed(overhang)):
        sp[p] = str(-(i + 1))
    for i, p in enumerate(acc5): sp[p] = str(i + 1)
    for i, p in enumerate(acc3): sp[p] = str(66 + i)
    disc = sorted(p for p in range(acc3[-1] + 1, n) if p not in pairs)
    for i, p in enumerate(disc): sp[p] = str(73 + i)
    d5 = stems[1]
    for i, p in enumerate([p for p in range(acc5[-1] + 1, d5[0]) if p not in pairs]):
        sp[p] = str(8 + i)
    d3 = sorted(pairs[p] for p in d5)
    nd = len(d5)
    for i, p in enumerate(d5): sp[p] = str(10 + i)
    for i, p in enumerate(d3): sp[p] = str(25 - nd + 1 + i)
    d_loop = [p for p in range(d5[-1] + 1, d3[0]) if p not in pairs]
    for i, p in enumerate(d_loop):
        labs = ["14", "15", "16", "17", "17a", "18", "19", "20", "20a", "20b", "21"]
        if i < len(labs): sp[p] = labs[i]
    ac5 = stems[2]
    lk2 = [p for p in range(d3[-1] + 1, ac5[0]) if p not in pairs]
    if lk2: sp[lk2[-1]] = "26"
    ac3 = sorted(pairs[p] for p in ac5)
    for i, p in enumerate(ac5): sp[p] = str(27 + i)
    for i, p in enumerate(ac3): sp[p] = str(39 + i)
    for i, p in enumerate([p for p in range(ac5[-1] + 1, ac3[0]) if p not in pairs]):
        sp[p] = str(32 + i)
    t5 = stems[-1]
    t3 = sorted(pairs[p] for p in t5)
    for i, p in enumerate(t5): sp[p] = str(49 + i)
    for i, p in enumerate(t3): sp[p] = str(61 + i)
    for i, p in enumerate([p for p in range(t5[-1] + 1, t3[0]) if p not in pairs]):
        sp[p] = str(54 + i)
    var = [p for p in range(ac3[-1] + 1, t5[0])]
    if len(stems) == 5:
        for i, p in enumerate(var): sp[p] = f"e{i + 1}"
    else:
        for i, p in enumerate([p for p in var if p not in pairs]): sp[p] = str(44 + i)
    return sp


_sp_cache: dict[tuple[str, str], dict[int, str]] = {}

def get_sprinzl_map(iso: str, domain: str) -> dict[int, str]:
    key = (iso, domain)
    if key not in _sp_cache:
        df = pd.read_csv(
            args.csv,
            usecols=["primary_sequence", "secondary_structure",
                     "Anticodon_predicted_isotype", "domain"],
            low_memory=False,
        )
        sub = df[
            (df["Anticodon_predicted_isotype"] == iso) &
            (df["domain"] == domain)
        ].dropna(subset=["primary_sequence", "secondary_structure"])
        if sub.empty:
            _sp_cache[key] = {}
        else:
            modal_len = sub["primary_sequence"].str.len().mode()[0]
            modal_ss  = (sub[sub["primary_sequence"].str.len() == modal_len]
                         ["secondary_structure"].mode()[0])
            _sp_cache[key] = build_sprinzl_map(modal_ss)
    return _sp_cache[key]


# ── Logo helpers ───────────────────────────────────────────────────────────────
def _sp_sort_key(label: str) -> tuple:
    if label.startswith("-"):
        try: return (-1, -int(label[1:]))
        except ValueError: return (-1, 0)
    if label.startswith("e"):
        try: return (100, int(label[1:]))
        except ValueError: return (100, 0)
    base = label.rstrip("ab")
    try: return (int(base), 0 if not label[len(base):] else ord(label[len(base)]))
    except ValueError: return (50, 0)


def select_positions(diff_sal: np.ndarray, raw_to_sp: dict[int, str]) -> list[int]:
    norms = np.abs(diff_sal).max(axis=1)
    keep  = [p for p in range(len(norms))
             if norms[p] >= args.min_sal
             and p in raw_to_sp
             and raw_to_sp[p] not in AC_EXCL]
    if args.top_n and len(keep) > args.top_n:
        keep = sorted(keep, key=lambda p: norms[p], reverse=True)[:args.top_n]
        keep = sorted(keep)
    return keep


def build_logo_df(diff_sal: np.ndarray,
                  display_positions: list[int],
                  active_set: set[int]) -> pd.DataFrame:
    rows = []
    for p in display_positions:
        if p not in active_set:
            rows.append([0.0, 0.0, 0.0, 0.0])
        else:
            row  = diff_sal[p]
            best  = int(np.argmax(row))
            worst = int(np.argmin(row))
            logo_row = [0.0, 0.0, 0.0, 0.0]
            if row[best]  > 0: logo_row[best]  = float(row[best])
            if row[worst] < 0: logo_row[worst] = float(row[worst])
            rows.append(logo_row)
    return pd.DataFrame(rows, columns=NUCS)


def plot_comparison_logo(ax, diff_sal: np.ndarray,
                         iso1: str, iso2: str, domain: str) -> int:
    raw_to_sp = get_sprinzl_map(iso1, domain)
    if not raw_to_sp:
        raw_to_sp = get_sprinzl_map(iso2, domain)
    if not raw_to_sp:
        ax.text(0.5, 0.5, "no Sprinzl map", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return 0

    active_positions = select_positions(diff_sal, raw_to_sp)
    if not active_positions:
        ax.text(0.5, 0.5, "no signal", ha="center", va="center",
                transform=ax.transAxes, fontsize=8)
        return 0

    sp_to_raw = {v: k for k, v in raw_to_sp.items()}
    ac_raw    = sorted(sp_to_raw[lbl] for lbl in AC_LOOP if lbl in sp_to_raw)
    disp_pos  = sorted(set(active_positions) | set(ac_raw))
    active_set = set(active_positions)

    logo_df   = build_logo_df(diff_sal, disp_pos, active_set)
    tick_labs = [raw_to_sp.get(p, str(p)) for p in disp_pos]

    logomaker.Logo(logo_df, ax=ax, color_scheme="classic",
                   font_name="DejaVu Sans", stack_order="big_on_top",
                   baseline_width=0.5, flip_below=True)

    ax.set_xticks(range(len(tick_labs)))
    ax.set_xticklabels(tick_labs, fontsize=6, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.4, zorder=0)
    ax.set_xlim(-0.5, len(tick_labs) - 0.5)

    ac_cols = [i for i, p in enumerate(disp_pos) if p in set(ac_raw)]
    if ac_cols:
        ax.axvspan(ac_cols[0] - 0.5, ac_cols[-1] + 0.5,
                   color="lightgrey", alpha=0.35, zorder=0)
        for lbl_obj, sp_lbl in zip(ax.get_xticklabels(), tick_labs):
            if sp_lbl in AC_LOOP:
                lbl_obj.set_color("grey")

    return len(active_positions)


# ── Discover domains ──────────────────────────────────────────────────────────
def find_domains(iso: str) -> list[str]:
    found = []
    for fname in os.listdir(args.emb_dir):
        if fname.startswith(f"gnn_ism_saliency_{iso}_") and fname.endswith(".npy"):
            dom = fname[len(f"gnn_ism_saliency_{iso}_"):-len(".npy")]
            found.append(dom)
    return sorted(found)


# ── Main loop ─────────────────────────────────────────────────────────────────
for iso1, iso2 in pairs_to_plot:
    doms1 = set(find_domains(iso1))
    doms2 = set(find_domains(iso2))
    shared_domains = sorted(doms1 & doms2)

    if args.domains:
        shared_domains = [d for d in shared_domains if d in args.domains]

    if not shared_domains:
        print(f"[{iso1}/{iso2}] no shared domains found — skipping", flush=True)
        continue

    print(f"\n[{iso1} vs {iso2}]  domains: {shared_domains}", flush=True)

    # Per-domain individual logos + collect for multi-domain grid
    domain_data: list[tuple[str, np.ndarray]] = []
    for domain in shared_domains:
        path1 = os.path.join(args.emb_dir, f"gnn_ism_saliency_{iso1}_{domain}.npy")
        path2 = os.path.join(args.emb_dir, f"gnn_ism_saliency_{iso2}_{domain}.npy")
        if not os.path.exists(path1) or not os.path.exists(path2):
            print(f"  [{domain}] missing saliency file — skipping", flush=True)
            continue

        sal1 = np.load(path1)[:max_len].astype(np.float32)
        sal2 = np.load(path2)[:max_len].astype(np.float32)
        diff = sal1 - sal2   # positive = iso1-discriminative, negative = iso2-discriminative

        raw_to_sp = get_sprinzl_map(iso1, domain) or get_sprinzl_map(iso2, domain)
        n_active  = len(select_positions(diff, raw_to_sp))
        if n_active == 0:
            print(f"  [{domain}] no positions above min-sal — skipping", flush=True)
            continue

        domain_data.append((domain, diff))

        # Individual logo
        fig_w = max(8, 2 + n_active * 0.25)
        fig, ax = plt.subplots(figsize=(fig_w, 3.5))
        n_pos = plot_comparison_logo(ax, diff, iso1, iso2, domain)
        ax.set_ylabel(f"ISM Δlogit ({iso1} − {iso2})", fontsize=9)
        ax.set_title(
            f"Pairwise ISM: tRNA-{iso1} vs tRNA-{iso2} ({domain})\n"
            f"+ = {iso1}-discriminative  |  − = {iso2}-discriminative  |  "
            f"Sprinzl positions  |  min|sal| = {args.min_sal}",
            fontsize=9,
        )
        fig.tight_layout()
        stem = f"pairwise_ism_logo_{iso1}vs{iso2}_{domain}"
        fig.savefig(os.path.join(outdir, f"{stem}.png"),
                    bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  [{domain}] {n_pos} positions → {stem}.png", flush=True)

    # Multi-domain comparison grid (one row per domain, stacked)
    if len(domain_data) > 1:
        n_rows = len(domain_data)
        # Estimate width from first domain's active positions
        raw_to_sp = get_sprinzl_map(iso1, domain_data[0][0]) or get_sprinzl_map(iso2, domain_data[0][0])
        first_active = select_positions(domain_data[0][1], raw_to_sp)
        ac_raw = [k for k, v in raw_to_sp.items() if v in AC_LOOP]
        n_cols_est = len(set(first_active) | set(ac_raw))
        fig_w = max(10, 2 + n_cols_est * 0.25)
        fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, 3.0 * n_rows), squeeze=False)

        for row_i, (domain, diff) in enumerate(domain_data):
            ax = axes[row_i][0]
            n_pos = plot_comparison_logo(ax, diff, iso1, iso2, domain)
            ax.set_ylabel(f"Δlogit\n({iso1}−{iso2})", fontsize=8)
            ax.set_title(domain, fontsize=9, fontweight="bold", loc="left", pad=2)
            ax.tick_params(axis="y", labelsize=7)
            if row_i < n_rows - 1:
                ax.set_xticklabels([])

        fig.suptitle(
            f"Pairwise ISM: tRNA-{iso1} vs tRNA-{iso2}  —  by domain\n"
            f"+ = {iso1}-discriminative  |  − = {iso2}-discriminative  |  "
            f"Sprinzl positions  |  min|sal| = {args.min_sal}",
            fontsize=10, y=1.01,
        )
        fig.tight_layout()
        stem = f"pairwise_ism_logo_{iso1}vs{iso2}_all_domains"
        fig.savefig(os.path.join(outdir, f"{stem}.png"),
                    bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Multi-domain → {stem}.png", flush=True)

print(f"\nDone. Results in: {outdir}/")
