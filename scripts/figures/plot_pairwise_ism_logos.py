#!/usr/bin/env python3
"""
plot_pairwise_ism_logos.py
--------------------------
Visualise pairwise ISM saliency from pairwise_ism_{iso}_{clade}.npz files
produced by compute_pairwise_ism.py.

For each (iso, clade), generates two figures:

1. Heatmap figure — one 4×4 panel per stem pair.
   Rows = 5′ nucleotide (A/C/G/U), cols = 3′ nucleotide.
   Colour = differential saliency (red = pro-isotype, blue = anti-isotype).
   Panels are ordered left-to-right by ascending Sprinzl position.
   Best (★) and worst (▼) base-pair combinations are annotated.

2. Summary bar chart — for each stem pair, the best and worst base-pair
   combination is shown as labelled bars, ordered by Sprinzl position.
   Useful for quickly scanning which stem pairs carry strong identity signals.

Usage:
    python dsthesis/scripts/figures/plot_pairwise_ism_logos.py \\
        --ism-dir gnn_final/pairwise_ism
    python dsthesis/scripts/figures/plot_pairwise_ism_logos.py \\
        --ism-dir gnn_final/pairwise_ism \\
        --isotypes Ala Gly --min-sal 0.003 --top-n 15
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

NUCS = ["A", "C", "G", "U"]

parser = argparse.ArgumentParser()
parser.add_argument("--ism-dir",  required=True,
                    help="Directory with pairwise_ism_*.npz files")
parser.add_argument("--outdir",   default=None,
                    help="Output directory (default: {ism-dir}/pairwise_logos)")
parser.add_argument("--isotypes", nargs="+", default=None)
parser.add_argument("--clades",   nargs="+", default=None)
parser.add_argument("--min-sal",  type=float, default=0.002,
                    help="Min |max(sal_disc)| over the 4×4 grid to include a pair (default: 0.002)")
parser.add_argument("--top-n",    type=int,   default=None,
                    help="Keep only the top-N most salient pairs (default: all)")
args = parser.parse_args()

outdir = args.outdir or os.path.join(args.ism_dir, "pairwise_logos")
os.makedirs(outdir, exist_ok=True)


# ── Discover pairwise_ism_*.npz files ────────────────────────────────────────
def find_npz_files(ism_dir: str) -> dict[tuple[str, str], str]:
    found = {}
    for fname in sorted(os.listdir(ism_dir)):
        if not (fname.startswith("pairwise_ism_") and fname.endswith(".npz")):
            continue
        stem = fname[len("pairwise_ism_"):-len(".npz")]
        # All standard tRNA isotype names are short and contain no underscores,
        # so the first "_" reliably separates iso from clade.
        sep = stem.find("_")
        if sep < 1:
            continue
        iso, clade = stem[:sep], stem[sep + 1:]
        found[(iso, clade)] = os.path.join(ism_dir, fname)
    return found


found = find_npz_files(args.ism_dir)
if not found:
    raise SystemExit(f"No pairwise_ism_*.npz files found in {args.ism_dir}")

pairs_to_plot = sorted(found.keys())
if args.isotypes:
    pairs_to_plot = [(iso, c) for iso, c in pairs_to_plot if iso in args.isotypes]
if args.clades:
    pairs_to_plot = [(iso, c) for iso, c in pairs_to_plot if c in args.clades]

print(f"Plotting {len(pairs_to_plot)} (iso, clade) combinations …", flush=True)


# ── Sprinzl sort key ──────────────────────────────────────────────────────────
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


# ── Load and filter a single npz ─────────────────────────────────────────────
def load_and_filter(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (sal_disc, sp5, sp3) after applying --min-sal and --top-n filters.
    sal_disc shape: (n_pairs_kept, 4, 4)
    """
    d        = np.load(path)
    sal_disc = d["sal_disc"].astype(np.float32)   # (n_pairs, 4, 4)
    sp5      = d["sprinzl_5"].astype(str)
    sp3      = d["sprinzl_3"].astype(str)

    # Keep pairs with at least one base-pair combination above the saliency threshold
    pair_mag = np.abs(sal_disc).max(axis=(1, 2))  # (n_pairs,)
    keep     = np.where(pair_mag >= args.min_sal)[0]

    if args.top_n and len(keep) > args.top_n:
        keep = keep[np.argsort(pair_mag[keep])[-args.top_n:]]

    # Sort kept indices by Sprinzl 5′ position
    # Use Python sorted (not np.argsort) — sort keys are tuples, which
    # np.argsort would convert to a 2-D array and sort incorrectly.
    sort_order = sorted(range(len(keep)), key=lambda i: _sp_sort_key(sp5[keep[i]]))
    keep = keep[sort_order]

    return sal_disc[keep], sp5[keep], sp3[keep]


# ── Heatmap figure ────────────────────────────────────────────────────────────
def plot_heatmaps(iso: str, clade: str,
                  sal_disc: np.ndarray, sp5: np.ndarray, sp3: np.ndarray) -> None:
    """Grid of 4×4 heatmaps, one per stem pair."""
    n_pairs = len(sp5)
    if n_pairs == 0:
        return

    n_cols   = min(n_pairs, 20)            # cap width; wrap into rows if needed
    n_rows   = (n_pairs + n_cols - 1) // n_cols
    fig_w    = max(6, n_cols * 1.4)
    fig_h    = max(3, n_rows * 2.2)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(fig_w, fig_h), squeeze=False)

    vmax = float(np.abs(sal_disc).max())
    if vmax == 0:
        vmax = 1e-6
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = "RdBu_r"

    for idx in range(n_pairs):
        row, col = divmod(idx, n_cols)
        ax  = axes[row][col]
        mat = sal_disc[idx]   # (4, 4): rows = ni (5′), cols = nj (3′)

        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(4)); ax.set_xticklabels(NUCS, fontsize=6)
        ax.set_yticks(range(4)); ax.set_yticklabels(NUCS, fontsize=6)
        ax.set_title(f"{sp5[idx]}:{sp3[idx]}", fontsize=7, fontweight="bold", pad=2)
        ax.tick_params(length=2)
        if col == 0:
            ax.set_ylabel("5′", fontsize=6)
        if row == n_rows - 1:
            ax.set_xlabel("3′", fontsize=6)

        # Annotate best and worst cells
        best_idx  = np.unravel_index(np.argmax(mat), mat.shape)
        worst_idx = np.unravel_index(np.argmin(mat), mat.shape)
        for (r, c_), sym in [(best_idx, "★"), (worst_idx, "▼")]:
            txt_col = "white" if abs(mat[r, c_]) > vmax * 0.6 else "black"
            ax.text(c_, r, sym, ha="center", va="center",
                    fontsize=8, color=txt_col, fontweight="bold")

    # Hide unused axes in the last row
    for idx in range(n_pairs, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    plt.colorbar(im, ax=axes.flat[min(n_pairs - 1, n_cols - 1)],
                 shrink=0.8, label="Δlogit (disc.)", pad=0.02)
    fig.suptitle(
        f"tRNA-{iso} ({clade})  —  pairwise ISM at stem positions\n"
        f"rows = 5′ nuc, cols = 3′ nuc  |  ★ best pair  ▼ worst pair"
        f"  |  min|sal| = {args.min_sal}",
        fontsize=9, y=1.01,
    )
    fig.tight_layout()

    stem = f"pairwise_ism_{iso}_{clade}_heatmaps"
    fig.savefig(os.path.join(outdir, f"{stem}.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [{iso}/{clade}] heatmaps ({n_pairs} pairs) → {stem}.png",
          flush=True)


# ── Summary bar chart ─────────────────────────────────────────────────────────
def plot_summary_bars(iso: str, clade: str,
                      sal_disc: np.ndarray, sp5: np.ndarray, sp3: np.ndarray) -> None:
    """Best and worst base-pair Δlogit per stem pair as labelled bars."""
    n_pairs = len(sp5)
    if n_pairs == 0:
        return

    best_vals  = np.array([sal_disc[i].max() for i in range(n_pairs)])
    worst_vals = np.array([sal_disc[i].min() for i in range(n_pairs)])

    def _pair_label(i: int, argfn) -> str:
        r, c = np.unravel_index(argfn(sal_disc[i]), (4, 4))
        return f"{NUCS[r]}{NUCS[c]}"

    best_labs  = [_pair_label(i, np.argmax) for i in range(n_pairs)]
    worst_labs = [_pair_label(i, np.argmin) for i in range(n_pairs)]

    x        = np.arange(n_pairs)
    fig_w    = max(6, n_pairs * 0.55)
    fig, ax  = plt.subplots(figsize=(fig_w, 3.5))

    ax.bar(x,  best_vals, color="#d62728", alpha=0.85, label="best pair",  zorder=2)
    ax.bar(x, worst_vals, color="#1f77b4", alpha=0.85, label="worst pair", zorder=2)

    pad = (best_vals.max() - worst_vals.min()) * 0.03
    for xi, (bv, wv, bl, wl) in enumerate(
            zip(best_vals, worst_vals, best_labs, worst_labs)):
        if bv > 0:
            ax.text(xi, bv + pad, bl, ha="center", va="bottom",
                    fontsize=6, rotation=60, color="#d62728")
        if wv < 0:
            ax.text(xi, wv - pad, wl, ha="center", va="top",
                    fontsize=6, rotation=60, color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s5}:{s3}" for s5, s3 in zip(sp5, sp3)],
                        rotation=45, ha="right", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5, zorder=1)
    ax.set_ylabel("Δlogit (differential, pair-centred)", fontsize=9)
    ax.set_title(
        f"tRNA-{iso} ({clade})  —  best / worst base pair per stem position\n"
        f"anticodon (34–36) excluded  |  min|sal| = {args.min_sal}",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    fig.tight_layout()

    stem = f"pairwise_ism_{iso}_{clade}_summary"
    fig.savefig(os.path.join(outdir, f"{stem}.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [{iso}/{clade}] summary ({n_pairs} pairs) → {stem}.png",
          flush=True)


# ── Per-isotype multi-clade heatmap grid ──────────────────────────────────────
def plot_multi_clade_summary(
    iso: str,
    clade_data: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
) -> None:
    """
    One row per clade, columns = union of all significant stem pairs.
    Cell = max(|sal_disc|) for that (clade, pair); absent pairs are white.
    """
    if not clade_data:
        return

    # Build union of pair labels present across all clades
    all_pair_labels = []
    seen = set()
    for _, sal_disc, sp5, sp3 in clade_data:
        for s5, s3 in zip(sp5, sp3):
            lbl = f"{s5}:{s3}"
            if lbl not in seen:
                all_pair_labels.append((lbl, s5, s3))
                seen.add(lbl)
    # Sort by Sprinzl 5′ then 3′
    all_pair_labels.sort(key=lambda t: (_sp_sort_key(t[1]), _sp_sort_key(t[2])))

    n_clades = len(clade_data)
    n_cols   = len(all_pair_labels)
    if n_cols == 0:
        return

    # Build magnitude matrix
    mag = np.zeros((n_clades, n_cols))
    for row_i, (clade, sal_disc, sp5, sp3) in enumerate(clade_data):
        pair_map = {f"{s5}:{s3}": sal_disc[k] for k, (s5, s3) in enumerate(zip(sp5, sp3))}
        for col_i, (lbl, _, _) in enumerate(all_pair_labels):
            if lbl in pair_map:
                mag[row_i, col_i] = float(np.abs(pair_map[lbl]).max())

    fig_w = max(8, n_cols * 0.5)
    fig_h = max(3, n_clades * 0.5 + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mag, cmap="Reds", aspect="auto",
                   vmin=0, vmax=float(mag.max()) or 1.0)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([t[0] for t in all_pair_labels],
                        rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(n_clades))
    ax.set_yticklabels([c for c, *_ in clade_data], fontsize=8)
    plt.colorbar(im, ax=ax, label="max|Δlogit| (disc.)", shrink=0.6)
    ax.set_title(
        f"tRNA-{iso}  —  pairwise ISM stem pair magnitude across clades\n"
        f"min|sal| = {args.min_sal}",
        fontsize=10,
    )
    fig.tight_layout()

    stem = f"pairwise_ism_{iso}_all_clades_overview"
    fig.savefig(os.path.join(outdir, f"{stem}.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [{iso}] multi-clade overview → {stem}.png", flush=True)


# ── Main plotting loop ────────────────────────────────────────────────────────
iso_clade_data: dict[str, list] = {}   # iso → [(clade, sal_disc, sp5, sp3)]

for iso, clade in pairs_to_plot:
    sal_disc, sp5, sp3 = load_and_filter(found[(iso, clade)])
    if len(sp5) == 0:
        print(f"  [{iso}/{clade}] no pairs above min-sal — skipping", flush=True)
        continue

    plot_heatmaps(iso, clade, sal_disc, sp5, sp3)
    plot_summary_bars(iso, clade, sal_disc, sp5, sp3)

    iso_clade_data.setdefault(iso, []).append((clade, sal_disc, sp5, sp3))

# Multi-clade overview per isotype (only when multiple clades present)
for iso, clade_list in iso_clade_data.items():
    if len(clade_list) > 1:
        plot_multi_clade_summary(iso, clade_list)

print(f"\nDone. Results in: {outdir}/")
