#!/usr/bin/env python3
"""
plot_dedup_accuracy.py
----------------------
Reads per-level result JSONs from dedup_accuracy/ and plots
test accuracy vs dedup stringency, with dataset size as background bars.

Usage:
    python scripts/plot_dedup_accuracy.py
    python scripts/plot_dedup_accuracy.py --indir dedup_accuracy --outdir figures
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

parser = argparse.ArgumentParser()
parser.add_argument("--indir",  default=f"{DATA_ROOT}/dedup_accuracy")
parser.add_argument("--outdir", default=f"{DATA_ROOT}/figures")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ── Level order: most stringent -> least ──────────────────────────────────────
ORDER = ["100pct", "domain", "phylum", "class", "order", "family", "genus", "species"]
XTICKLABELS = {
    "100pct":  "100% id\n(mmseqs2)",
    "domain":  "Domain",
    "phylum":  "Phylum",
    "class":   "Class",
    "order":   "Order",
    "family":  "Family",
    "genus":   "Genus",
    "species": "Species",
}

# ── Load results ───────────────────────────────────────────────────────────────
results = {}
for name in ORDER:
    path = os.path.join(args.indir, f"result_{name}.json")
    if os.path.exists(path):
        with open(path) as f:
            results[name] = json.load(f)
    else:
        print(f"  Missing: {path}")

present = [k for k in ORDER if k in results]
if not present:
    raise SystemExit("No result files found — run classify_dedup.py for each level first.")

accs   = [results[k]["test_acc"]     for k in present]
sizes  = [results[k]["n_rows_total"] for k in present]
used   = [results[k]["n_used"]       for k in present]
labels = [XTICKLABELS[k]             for k in present]
xs     = np.arange(len(present))

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

# Background bars = full dataset size
bar_color = "#CBD5E1"
ax2.bar(xs, [s / 1e6 for s in sizes], color=bar_color, alpha=0.6,
        zorder=1, label="Dataset rows (total)")
ax2.bar(xs, [u / 1e6 for u in used],  color="#94A3B8", alpha=0.6,
        zorder=1, label="Rows used for classifier")
ax2.set_ylabel("Dataset size (M rows)", color="#64748B", fontsize=10)
ax2.tick_params(axis="y", colors="#64748B")
ax2.set_ylim(0, max(sizes) / 1e6 * 1.55)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f M"))

# Accuracy line
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
ax1.set_ylabel("Test accuracy (softmax, RNA-FM embeddings)", fontsize=11)
ax1.set_xlabel(
    "Deduplication level   [<-- more stringent          less stringent -->]",
    fontsize=10,
)
acc_min = min(accs)
ax1.set_ylim(max(0, acc_min - 0.08), 1.03)
ax1.yaxis.grid(True, linestyle="--", alpha=0.45, zorder=0)
ax1.set_axisbelow(True)
ax1.set_title(
    "RNA-FM + softmax: amino acid isotype accuracy\nby deduplication level (sequence -> amino acid)",
    fontsize=12,
)

# Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [
    Line2D([0], [0], color=acc_color, marker="o", linewidth=2, label="Test accuracy"),
    Patch(facecolor=bar_color, alpha=0.6, label="Dataset rows (total)"),
    Patch(facecolor="#94A3B8", alpha=0.6, label="Rows used (capped at 500K)"),
]
ax1.legend(handles=legend_elements, loc="lower left", fontsize=8)

fig.tight_layout()

out_path = os.path.join(args.outdir, "dedup_accuracy.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved -> {out_path}")

# ── Print summary table ────────────────────────────────────────────────────────
print(f"\n{'Level':<12}  {'Rows (total)':>14}  {'Rows used':>10}  {'Test acc':>9}")
print(f"{'─'*12}  {'─'*14}  {'─'*10}  {'─'*9}")
for k in present:
    r = results[k]
    print(f"{k:<12}  {r['n_rows_total']:>14,}  {r['n_used']:>10,}  {r['test_acc']:>9.4f}")
