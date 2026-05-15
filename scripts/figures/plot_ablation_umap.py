#!/usr/bin/env python3
"""
plot_umap.py
------------
UMAP visualization of RNA-FM and RiNALMo embeddings (masked and unmasked),
coloured by tRNA isotype.

Loads whichever embedding sets are present under embeddings/.
Subsamples sequences for speed (--n-samples, default 100k).

Usage
-----
    python scripts/plot_umap.py
    python scripts/plot_umap.py --n-samples 50000
    python scripts/plot_umap.py --emb-dir embeddings --outdir figures
"""

import argparse
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")
sys.path.insert(0, os.path.join(DATA_ROOT, "scripts"))
from utils.iso_classes import ISOTYPE_ORDER as ISOTYPES, MERGE_ISOS

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--input",     default=f"{DATA_ROOT}/trnascan_GTDB_r226_v2_final.csv")
parser.add_argument("--emb-dir",   default="embeddings")
parser.add_argument("--outdir",    default="figures")
parser.add_argument("--n-samples", type=int, default=100_000,
                    help="Sequences to subsample per panel for UMAP (default: 100k)")
parser.add_argument("--umap-neighbors", type=int, default=15)
parser.add_argument("--umap-min-dist",  type=float, default=0.1)
parser.add_argument("--seed",      type=int, default=42)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ── Isotype colour palette ────────────────────────────────────────────────────
CMAP = plt.get_cmap("tab20")
COLOURS = {iso: CMAP(i % 20) for i, iso in enumerate(ISOTYPES)}

# ── Build sequence → isotype lookup (stream CSV) ──────────────────────────────
print("Building sequence → isotype lookup …", flush=True)
seq_to_iso = {}

with open(args.input, newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    has_masked = "anticodon_masked_sequence" in reader.fieldnames
    for row in reader:
        iso = MERGE_ISOS.get(row["Anticodon_predicted_isotype"].strip(),
                             row["Anticodon_predicted_isotype"].strip())
        if iso not in ISOTYPES:
            continue
        seq_to_iso[row["primary_sequence"].strip()] = iso
        if has_masked:
            masked = row["anticodon_masked_sequence"].strip()
            if masked != "NA":
                seq_to_iso[masked] = iso

print(f"  {len(seq_to_iso):,} sequences indexed\n", flush=True)

# ── Discover available embedding sets ────────────────────────────────────────
PANEL_ORDER = ["rnafm_masked", "rinalmo_masked"]

def make_label(subdir):
    model  = "RNA-FM" if "rnafm" in subdir else "RiNALMo" if "rinalmo" in subdir else subdir
    suffix = "masked" if subdir.endswith("_masked") else "unmasked"
    return f"{model}\n{suffix}"

available = {}
for subdir in PANEL_ORDER:
    emb_path  = os.path.join(args.emb_dir, subdir, "embeddings.npy")
    seqs_path = os.path.join(args.emb_dir, subdir, "unique_sequences.npy")
    if os.path.exists(emb_path) and os.path.exists(seqs_path):
        seq_col = "anticodon_masked_sequence" if subdir.endswith("_masked") else "primary_sequence"
        available[make_label(subdir)] = (emb_path, seqs_path, seq_col)
    else:
        print(f"  SKIP {subdir!r}: embeddings not complete", flush=True)

if not available:
    print("No complete embedding sets found. Run the extract scripts first.",
          file=sys.stderr)
    sys.exit(1)

print(f"Found {len(available)} embedding set(s): {list(available)}\n", flush=True)

# ── UMAP ──────────────────────────────────────────────────────────────────────
import umap


rng = np.random.default_rng(args.seed)

n_cols  = min(2, len(available))
n_rows  = (len(available) + 1) // 2
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(8 * n_cols, 7 * n_rows),
                         squeeze=False)
axes_flat = axes.flatten()

for ax_idx, (label, (emb_path, seqs_path, seq_col)) in enumerate(available.items()):
    ax = axes_flat[ax_idx]
    print(f"[{ax_idx+1}/{len(available)}] {label!r}", flush=True)

    embs = np.load(emb_path, mmap_mode="r")  # (N, D)
    seqs = np.load(seqs_path)          # (N,)
    print(f"  Loaded {len(seqs):,} sequences  dim={embs.shape[1]}", flush=True)

    # assign isotype labels (empty string for sequences not in our class set)
    labels = np.array([seq_to_iso.get(s, "") for s in seqs])

    # subsample
    n = min(args.n_samples, len(seqs))
    idx = rng.choice(len(seqs), size=n, replace=False)
    embs_sub  = embs[idx]
    labels_sub = labels[idx]
    print(f"  Subsampled to {n:,}", flush=True)

    # fit UMAP
    print(f"  Fitting UMAP …", flush=True)
    reducer = umap.UMAP(
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        n_components=2,
        random_state=args.seed,
        verbose=False,
    )
    coords = reducer.fit_transform(embs_sub)   # (n, 2)
    print(f"  UMAP done", flush=True)

    # plot — draw each isotype separately so legend is clean
    present_isos = [iso for iso in ISOTYPES if iso in labels_sub]
    for iso in present_isos:
        mask = labels_sub == iso
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[COLOURS[iso]], s=1, alpha=0.4, linewidths=0, rasterized=True)

    ax.set_aspect('equal', adjustable='datalim')
    ax.set_xlabel("UMAP 1", fontsize=10)
    ax.set_ylabel("UMAP 2", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

# hide unused axes
for ax in axes_flat[len(available):]:
    ax.set_visible(False)

fig.tight_layout(rect=[0, 0.18, 1, 1.0])

# shared legend — placed below subplots in the reserved bottom space
patches = [mpatches.Patch(color=COLOURS[iso], label=iso)
           for iso in ISOTYPES if iso in seq_to_iso.values()]
fig.legend(handles=patches, title="Isotype",
           loc="lower center", ncol=8,
           bbox_to_anchor=(0.5, 0.02), fontsize=9)

out_png = os.path.join(args.outdir, "umap_embeddings.png")
out_svg = os.path.join(args.outdir, "umap_embeddings.svg")
fig.savefig(out_png, bbox_inches="tight", dpi=150)
fig.savefig(out_svg, bbox_inches="tight")
print(f"\nSaved:\n  {out_png}\n  {out_svg}")
