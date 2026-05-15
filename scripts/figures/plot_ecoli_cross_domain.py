#!/usr/bin/env python3
"""
plot_ecoli_cross_domain.py
--------------------------
Re-plot ecoli_vs_orders_cosine_ism.csv as a cross-domain comparison:
two side-by-side panels (Bacteria / Archaea) each showing the top-N
most distant orders, bars coloured by isotype.

Usage
-----
  python scripts/analysis/plot_ecoli_cross_domain.py
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

parser = argparse.ArgumentParser()
parser.add_argument("--csv",
    default="order_identity_embeddings_all_culturable_ecoli/ecoli_vs_orders_cosine_ism.csv")
parser.add_argument("--outdir",
    default="order_identity_embeddings_all_culturable_ecoli")
parser.add_argument("--top-n", type=int, default=15,
    help="Top N orders per domain to display")
args = parser.parse_args()

df = pd.read_csv(args.csv)
anchor_label = df["anchor_label"].iloc[0]
anchor_id    = df["anchor_genome_id"].iloc[0]
ism_domain   = "avg"  # inferred from filename context

# ── colour map shared across both panels ──────────────────────────────────
iso_unique = sorted(df["isotype"].unique())
cmap = plt.cm.tab20(np.linspace(0, 1, max(len(iso_unique), 1)))
iso_color = {iso: cmap[i] for i, iso in enumerate(iso_unique)}

domains = ["Bacteria", "Archaea"]
domain_subsets = {d: df[df["target_domain"] == d]
                      .sort_values("cosine_dist", ascending=False)
                      .head(args.top_n)
                  for d in domains}

fig = plt.figure(figsize=(16, 7))
fig.suptitle(
    f"Cross-domain tRNA identity distances from {anchor_label}\n"
    f"(anchor: {anchor_id}; ISM-weighted cosine distance)",
    fontsize=12, y=1.01,
)

gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.08)

y_max = df["cosine_dist"].max() * 1.08

for col_idx, domain in enumerate(domains):
    sub = domain_subsets[domain]
    ax = fig.add_subplot(gs[0, col_idx])

    bar_colors = [iso_color[iso] for iso in sub["isotype"]]
    ax.bar(range(len(sub)), sub["cosine_dist"],
           color=bar_colors, edgecolor="gray", linewidth=0.4)

    for i, (_, row) in enumerate(sub.iterrows()):
        label = f"{row['isotype']}  vs {row['target_order']}"
        ax.text(i, row["cosine_dist"] * 0.015, label,
                ha="center", va="bottom", fontsize=6.5,
                rotation=90, color="black", clip_on=True)

    ax.set_xticks([])
    ax.set_ylim(0, y_max)
    ax.set_title(f"{domain}  (top {args.top_n})", fontsize=11, pad=6)
    if col_idx == 0:
        ax.set_ylabel("Cosine distance (1 – similarity)", fontsize=10)
    else:
        ax.set_yticklabels([])

# ── shared legend ──────────────────────────────────────────────────────────
legend_handles = [Patch(facecolor=iso_color[iso], label=iso) for iso in iso_unique]
fig.legend(handles=legend_handles, title="Isotype", fontsize=7,
           ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.18),
           framealpha=0.9)

plt.tight_layout()
os.makedirs(args.outdir, exist_ok=True)
for ext in ("png", "pdf"):
    out = os.path.join(args.outdir, f"cross_domain_ecoli_distances.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved -> {out}")
plt.close(fig)
