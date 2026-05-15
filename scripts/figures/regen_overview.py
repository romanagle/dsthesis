"""
Regenerate 01_overview.png for figures/dedup_sprinzl/ using:
  - original stats hardcoded from figures/dedup_sprinzl/stats.txt
  - dedup stats computed live from the dedup CSV
"""
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import os

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

parser = argparse.ArgumentParser()
parser.add_argument("--csv",    default="/data/roma/data/dedup/trnascan_GTDB_r226_dedup.csv",
                    help="Path to the dedup CSV (default: /data/roma/data/dedup/trnascan_GTDB_r226_dedup.csv)")
parser.add_argument("--outdir", default=f"{DATA_ROOT}/figures/dedup_sprinzl",
                    help="Output directory for figures")
args = parser.parse_args()

OUT = Path(args.outdir)

COLORS = {"Original": "#4C72B0", "Deduplicated": "#DD8452"}
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Hardcoded from stats.txt (original CSV was deleted)
ORIG_TOTAL    = 6_068_145
ORIG_GENOMES  = 142_794
ORIG_CONTIGS  = 2_747_054
ORIG_SEQS     = 1_973_624  # unique sequences in raw taxonomy file

print("Loading dedup CSV …")
dedup = pd.read_csv(args.csv, low_memory=False, on_bad_lines="skip")
dedup = dedup[dedup["tRNAscanID"] != "tRNAscanID"].copy()

metrics = {
    "Total tRNAs":      (ORIG_TOTAL,   len(dedup)),
    "Unique genomes":   (ORIG_GENOMES, dedup["GenomeID"].nunique()),
    "Unique sequences": (ORIG_SEQS,    dedup["primary_sequence"].nunique()),
    "Unique contigs":   (ORIG_CONTIGS, dedup["ContigID"].nunique()),
}

labels  = list(metrics.keys())
orig_v  = [metrics[m][0] for m in labels]
dedup_v = [metrics[m][1] for m in labels]

x     = np.arange(len(labels))
width = 0.38

fig, ax = plt.subplots(figsize=(10, 5))
b1 = ax.bar(x - width/2, orig_v,  width, label="Original",    color=COLORS["Original"],     alpha=0.85)
b2 = ax.bar(x + width/2, dedup_v, width, label="Deduplicated", color=COLORS["Deduplicated"], alpha=0.85)

for bar in list(b1) + list(b2):
    v = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, v * 1.01,
            f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.1f}k",
            ha="center", va="bottom", fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6 else f"{v/1e3:.0f}k"))
ax.set_ylabel("Count")
ax.legend()
plt.tight_layout()
fig.savefig(OUT / "01_overview.png")
fig.savefig(OUT / "01_overview.svg")
plt.close()
print(f"Saved {OUT}/01_overview.png, {OUT}/01_overview.svg")
