#!/usr/bin/env python3
"""
plot_dedup_sprinzl_accuracy.py
-------------------------------
Reads per-level result JSONs from figures/dedup_sprinzl/results/ and plots
test accuracy vs dedup stringency, with dataset size as background bars.

Usage:
    python scripts/figures/plot_dedup_sprinzl_accuracy.py
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

INDIR  = f"{DATA_ROOT}/figures/dedup/results"
OUTDIR = f"{DATA_ROOT}/figures/dedup"

# Less stringent → more stringent (left to right)
ORDER = ["none", "species", "genus", "family", "order", "class", "phylum", "domain"]
XTICKLABELS = {
    "none":    "No dedup\n(all data)",
    "species": "Species",
    "genus":   "Genus",
    "family":  "Family",
    "order":   "Order",
    "class":   "Class",
    "phylum":  "Phylum",
    "domain":  "Domain",
}

results = {}
for name in ORDER:
    path = os.path.join(INDIR, f"result_{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            results[name] = json.load(f)
    else:
        print(f"  Missing: {path}")

present = [k for k in ORDER if k in results]
if not present:
    raise SystemExit("No result files found.")

accs   = [results[k]["test_acc"]     for k in present]
sizes  = [results[k]["n_rows_total"] for k in present]
used   = [results[k]["n_used"]       for k in present]
labels = [XTICKLABELS[k]             for k in present]
xs     = np.arange(len(present))

fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

bar_color = "#CBD5E1"
ax2.bar(xs, [s / 1e6 for s in sizes], color=bar_color, alpha=0.6,
        zorder=1, label="Dataset rows (total)")
ax2.bar(xs, [u / 1e6 for u in used],  color="#94A3B8", alpha=0.6,
        zorder=1, label="Rows used (cap=200,000)")
ax2.set_ylabel("Dataset size (M rows)", color="#64748B", fontsize=10)
ax2.tick_params(axis="y", colors="#64748B")
ax2.set_ylim(0, max(sizes) / 1e6 * 1.55)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f M"))

acc_color = "#1D4ED8"
ax1.plot(xs, accs, "o-", color=acc_color, linewidth=2.5, markersize=8, zorder=4)
for i, (a, k) in enumerate(zip(accs, present)):
    ax1.annotate(
        f"{a:.3f}", (i, a),
        textcoords="offset points", xytext=(0, 10),
        ha="center", fontsize=8.5, color=acc_color, fontweight="bold",
    )

ax1.set_xticks(xs)
ax1.set_xticklabels(labels, fontsize=9)
ax1.set_ylabel("Test accuracy", fontsize=11)
ax1.set_xlabel(
    "Deduplication level   [<-- less stringent          more stringent -->]",
    fontsize=10,
)
acc_min = min(accs)
ax1.set_ylim(max(0, acc_min - 0.08), 1.03)
ax1.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
ax1.set_axisbelow(True)

legend_elements = [
    Line2D([0], [0], color=acc_color, marker="o", linewidth=2, label="Test accuracy"),
    Patch(facecolor=bar_color, alpha=0.6, label="Dataset rows (total)"),
    Patch(facecolor="#94A3B8", alpha=0.6, label="Rows used (cap=200,000)"),
]
ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)

fig.tight_layout()

out_png = os.path.join(OUTDIR, "dedup_sprinzl_accuracy.png")
out_svg = os.path.join(OUTDIR, "dedup_sprinzl_accuracy.svg")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
print(f"Saved -> {out_png}, {out_svg}")
