#!/usr/bin/env python3
"""
compare_film_vs_base.py
-----------------------
Compare baseline GNN vs FiLM-conditioned GNN on:
  1. Test classification accuracy
  2. Identity element (IE) scores for Ala and Gly
  3. ISM saliency heatmaps side-by-side for Ala and Gly (Bacteria + Archaea)

Usage
-----
    python scripts/figures/compare_film_vs_base.py
    python scripts/figures/compare_film_vs_base.py \
        --base-dir gnn_final --film-dir gnn_film \
        --outdir figures/film_comparison
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

parser = argparse.ArgumentParser()
parser.add_argument("--base-dir",     default="gnn_final")
parser.add_argument("--film-dir",     default="gnn_film")
parser.add_argument("--film-sal-dir", default=None,
                    help="Directory with Sprinzl-aligned FiLM saliency .npy files "
                         "(default: same as --film-dir)")
parser.add_argument("--outdir",   default="figures/film_comparison")
parser.add_argument("--isotypes", nargs="+", default=["Ala", "Gly", "His", "Asp", "Lys"])
parser.add_argument("--domains",  nargs="+", default=["Bacteria", "Archaea"])
args = parser.parse_args()
film_sal_dir = args.film_sal_dir or args.film_dir

os.makedirs(args.outdir, exist_ok=True)

# ── Load results JSON ─────────────────────────────────────────────────────────
def load_results(d):
    p = os.path.join(d, "gnn_results.json")
    with open(p) as f:
        return json.load(f)

base_res = load_results(args.base_dir)
film_res = load_results(args.film_dir)

# ── Load saliency arrays ──────────────────────────────────────────────────────
def load_sal(d, iso, domain):
    p = os.path.join(d, f"gnn_ism_saliency_{iso}_{domain}.npy")
    return np.load(p) if os.path.exists(p) else None

NUCS = ["A", "C", "G", "U"]

# ── Figure 1: Accuracy + IE scores ───────────────────────────────────────────
ie_keys = sorted(set(base_res.get("ie_scores", {}).keys()) |
                 set(film_res.get("ie_scores", {}).keys()))

fig1, axes = plt.subplots(1, 2, figsize=(10, 4))
fig1.subplots_adjust(wspace=0.35)

# accuracy
ax = axes[0]
vals  = [base_res["accuracy"] * 100, film_res["accuracy"] * 100]
bars  = ax.bar(["Baseline", "FiLM"], vals, color=["#4878CF", "#D65F5F"], width=0.5)
ax.set_ylim(min(vals) - 2, 100)
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Classification accuracy")
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.1, f"{v:.2f}%",
            ha="center", va="bottom", fontsize=9)

# IE scores
ax = axes[1]
if ie_keys:
    x      = np.arange(len(ie_keys))
    w      = 0.35
    b_vals = [base_res.get("ie_scores", {}).get(k, 0) for k in ie_keys]
    f_vals = [film_res.get("ie_scores", {}).get(k, 0) for k in ie_keys]
    ax.bar(x - w/2, b_vals, w, label="Baseline", color="#4878CF")
    ax.bar(x + w/2, f_vals, w, label="FiLM",     color="#D65F5F")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in ie_keys], fontsize=8)
    ax.set_ylabel("IE score (0–100)")
    ax.set_ylim(0, 100)
    ax.set_title("Identity element scores")
    ax.legend(fontsize=8)
else:
    ax.text(0.5, 0.5, "No IE scores in results JSON", ha="center", transform=ax.transAxes)

fig1.savefig(os.path.join(args.outdir, "accuracy_ie_comparison.png"),
             dpi=150, bbox_inches="tight")
fig1.savefig(os.path.join(args.outdir, "accuracy_ie_comparison.svg"),
             bbox_inches="tight")
plt.close(fig1)
print(f"Saved accuracy_ie_comparison.{{png,svg}}", flush=True)

# ── Figure 2: Saliency heatmaps side-by-side ─────────────────────────────────
pairs = [(iso, dom) for iso in args.isotypes for dom in args.domains
         if load_sal(args.base_dir, iso, dom) is not None or
            load_sal(film_sal_dir,   iso, dom) is not None]

if pairs:
    ncols  = 4   # base_B, film_B, base_A, film_A — or as available
    nrows  = len(pairs)
    # Layout: for each (iso, domain): [baseline | FiLM | diff]
    fig2, axes2 = plt.subplots(nrows, 3, figsize=(14, 2.2 * nrows + 0.5),
                                squeeze=False)
    fig2.subplots_adjust(hspace=0.5, wspace=0.3)

    col_titles = ["Baseline", "FiLM", "FiLM − Baseline"]
    for j, title in enumerate(col_titles):
        axes2[0, j].set_title(title, fontsize=10, fontweight="bold")

    vmax_global = 0.0
    sals = {}
    for iso, dom in pairs:
        b = load_sal(args.base_dir, iso, dom)
        f = load_sal(film_sal_dir,   iso, dom)
        sals[(iso, dom)] = (b, f)
        for arr in (b, f):
            if arr is not None:
                vmax_global = max(vmax_global, float(np.abs(arr).max()))

    for row_i, (iso, dom) in enumerate(pairs):
        b_sal, f_sal = sals[(iso, dom)]
        row_axes = axes2[row_i]
        row_axes[0].set_ylabel(f"{iso}\n{dom}", fontsize=8, rotation=0,
                               labelpad=50, va="center")

        for col_j, arr in enumerate([b_sal, f_sal]):
            ax = row_axes[col_j]
            if arr is not None:
                # Transpose to (4, L) for heatmap; keep only non-zero columns
                nonzero = np.any(arr != 0, axis=1)
                arr_t = arr[nonzero].T             # (4, L_nz)
                im = ax.imshow(arr_t, aspect="auto", cmap="RdBu_r",
                               vmin=-vmax_global, vmax=vmax_global,
                               interpolation="nearest")
                ax.set_yticks(range(4))
                ax.set_yticklabels(NUCS, fontsize=6)
                ax.set_xticks([])
            else:
                ax.text(0.5, 0.5, "missing", ha="center", transform=ax.transAxes,
                        fontsize=8, color="gray")
                ax.set_axis_off()

        ax_diff = row_axes[2]
        if b_sal is not None and f_sal is not None:
            diff = f_sal - b_sal
            nonzero = np.any(diff != 0, axis=1)
            diff_t  = diff[nonzero].T
            vd = float(np.abs(diff).max()) or 0.01
            ax_diff.imshow(diff_t, aspect="auto", cmap="RdBu_r",
                           vmin=-vd, vmax=vd, interpolation="nearest")
            ax_diff.set_yticks(range(4))
            ax_diff.set_yticklabels(NUCS, fontsize=6)
            ax_diff.set_xticks([])
        else:
            ax_diff.text(0.5, 0.5, "n/a", ha="center", transform=ax_diff.transAxes,
                         fontsize=8, color="gray")
            ax_diff.set_axis_off()

    fig2.savefig(os.path.join(args.outdir, "saliency_heatmap_comparison.png"),
                 dpi=150, bbox_inches="tight")
    fig2.savefig(os.path.join(args.outdir, "saliency_heatmap_comparison.svg"),
                 bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved saliency_heatmap_comparison.{{png,svg}}", flush=True)
else:
    print("No saliency files found for any requested (isotype, domain) pair — "
          "run training first.", flush=True)

# ── Figure 3: Per-position saliency difference (top changed positions) ────────
for iso in args.isotypes:
    for dom in args.domains:
        b_sal = load_sal(args.base_dir, iso, dom)
        f_sal = load_sal(film_sal_dir,   iso, dom)
        if b_sal is None or f_sal is None:
            continue
        diff = f_sal - b_sal                       # (L, 4)
        pos_change = np.abs(diff).max(axis=1)      # (L,) — max change at each position
        top_pos = np.argsort(pos_change)[::-1][:20]

        fig3, ax3 = plt.subplots(figsize=(10, 3))
        x = np.arange(len(top_pos))
        ax3.bar(x, pos_change[top_pos], color="#8B6A9E")
        ax3.set_xticks(x)
        ax3.set_xticklabels([str(p) for p in top_pos], fontsize=7, rotation=45)
        ax3.set_xlabel("Raw position index")
        ax3.set_ylabel("|FiLM − Baseline| max over nucleotides")
        ax3.set_title(f"{iso} {dom}: top 20 positions changed by FiLM")
        fig3.savefig(os.path.join(args.outdir, f"top_changes_{iso}_{dom}.png"),
                     dpi=150, bbox_inches="tight")
        plt.close(fig3)
print(f"Saved per-isotype top-change figures.", flush=True)
print("Done.")
