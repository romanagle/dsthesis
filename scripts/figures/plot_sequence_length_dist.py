#!/usr/bin/env python3
"""
plot_sequence_length_dist.py
----------------------------
Plot the distribution of primary sequence lengths for each tRNA isotype
from the tRNAscan-SE output.

Usage
-----
    python scripts/plot_sequence_length_dist.py
    python scripts/plot_sequence_length_dist.py --isotypes Ala Gly His
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")
import sys
sys.path.insert(0, os.path.join(DATA_ROOT, "scripts"))
from utils.iso_classes import normalize_isotypes


parser = argparse.ArgumentParser()
parser.add_argument("--csv",     default="data/dedup/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--outdir",  default="figures")
parser.add_argument("--isotypes", nargs="*", default=None)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

print("Loading data …", flush=True)
df = pd.read_csv(args.csv, usecols=["primary_sequence", "Anticodon_predicted_isotype"],
                 low_memory=False)
df = df.dropna(subset=["primary_sequence", "Anticodon_predicted_isotype"])
df = normalize_isotypes(df)
df["length"] = df["primary_sequence"].str.len()

iso_list = sorted(df["Anticodon_predicted_isotype"].unique())
if args.isotypes:
    iso_list = [i for i in args.isotypes if i in iso_list]

print(f"  {len(df):,} sequences  |  {len(iso_list)} isotypes", flush=True)

# ── Plot ──────────────────────────────────────────────────────────────────────
N_COLS  = 5
n_rows  = (len(iso_list) + N_COLS - 1) // N_COLS

fig, axes = plt.subplots(n_rows, N_COLS,
                          figsize=(4 * N_COLS, 2.8 * n_rows),
                          squeeze=False)

all_lengths = df["length"]
global_min  = int(all_lengths.min())
global_max  = int(all_lengths.max())
bins        = np.arange(global_min, global_max + 2) - 0.5   # integer-centred bins

for ax_i, iso in enumerate(iso_list):
    ax   = axes[ax_i // N_COLS][ax_i % N_COLS]
    vals = df.loc[df["Anticodon_predicted_isotype"] == iso, "length"]

    ax.hist(vals, bins=bins, color="steelblue", edgecolor="none", alpha=0.85)

    med = vals.median()
    ax.axvline(med, color="firebrick", linewidth=1.2, linestyle="--")

    ax.set_title(iso, fontsize=11, fontweight="bold")
    ax.set_xlabel("length (nt)", fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.text(0.97, 0.95, f"n={len(vals):,}\nmed={int(med)}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color="gray")

for ax_i in range(len(iso_list), n_rows * N_COLS):
    axes[ax_i // N_COLS][ax_i % N_COLS].set_visible(False)

fig.suptitle("tRNA primary sequence length distributions by isotype\n"
             "(dashed line = median)", fontsize=13, y=1.01)
fig.tight_layout()

out_pdf = os.path.join(args.outdir, "sequence_length_dist.pdf")
out_png = out_pdf.replace(".pdf", ".png")
fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
fig.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
