#!/usr/bin/env python3
"""
plot_centroid_heatmap.py
------------------------
For each isotype, computes the mean embedding (centroid) across all sequences
belonging to that isotype, then plots a pairwise cosine-distance heatmap
between centroids.

Shows RNA-FM unmasked vs masked side by side so you can see how much the
anticodon contributes to inter-isotype separation.

Usage
-----
    python scripts/plot_centroid_heatmap.py
    python scripts/plot_centroid_heatmap.py --models rnafm rinalmo
    python scripts/plot_centroid_heatmap.py --distance euclidean
"""

import argparse
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")
sys.path.insert(0, os.path.join(DATA_ROOT, "scripts"))
from utils.iso_classes import ISOTYPE_ORDER as ISOTYPES, MERGE_ISOS


parser = argparse.ArgumentParser()
parser.add_argument("--input",    default=f"{DATA_ROOT}/trnascan_GTDB_r226_v2_final.csv")
parser.add_argument("--emb-dir",  default="embeddings")
parser.add_argument("--outdir",   default="figures")
parser.add_argument("--distance", default="cosine", choices=["cosine", "euclidean"])
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)


# ── Step 1: build sequence → isotype lookup ───────────────────────────────────
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
            m = row["anticodon_masked_sequence"].strip()
            if m != "NA":
                seq_to_iso[m] = iso

print(f"  {len(seq_to_iso):,} sequences indexed\n", flush=True)


def cosine_distance_matrix(centroids: np.ndarray) -> np.ndarray:
    """Pairwise cosine distances between rows of centroids (N, D)."""
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normed = centroids / norms
    sim = normed @ normed.T
    return 1.0 - sim


def euclidean_distance_matrix(centroids: np.ndarray) -> np.ndarray:
    diff = centroids[:, None, :] - centroids[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def compute_centroids(emb_path, seqs_path):
    print(f"  Loading {emb_path} …", flush=True)
    embs = np.load(emb_path)          # (N, D)
    seqs = np.load(seqs_path)         # (N,)

    D = embs.shape[1]
    sums   = {iso: np.zeros(D, dtype=np.float64) for iso in ISOTYPES}
    counts = {iso: 0 for iso in ISOTYPES}

    for i, seq in enumerate(seqs):
        iso = seq_to_iso.get(seq)
        if iso is None:
            continue
        sums[iso]   += embs[i].astype(np.float64)
        counts[iso] += 1

    centroids = []
    present   = []
    for iso in ISOTYPES:
        if counts[iso] > 0:
            centroids.append(sums[iso] / counts[iso])
            present.append(iso)
            print(f"    {iso:6s}  n={counts[iso]:>10,}", flush=True)
        else:
            print(f"    {iso:6s}  MISSING", flush=True)

    return np.array(centroids, dtype=np.float32), present


# ── Step 2: discover and compute centroids for each available set ─────────────
PANEL_ORDER = ["rnafm_unmasked", "rnafm_masked", "rinalmo_unmasked", "rinalmo_masked"]
subdirs = [
    d for d in PANEL_ORDER
    if os.path.exists(os.path.join(args.emb_dir, d, "embeddings.npy"))
    and os.path.exists(os.path.join(args.emb_dir, d, "unique_sequences.npy"))
]

if not subdirs:
    print("No complete embedding sets found.", file=sys.stderr)
    sys.exit(1)

print(f"Found: {subdirs}\n", flush=True)

dist_fn = cosine_distance_matrix if args.distance == "cosine" else euclidean_distance_matrix

def make_title(subdir):
    model  = "RNA-FM" if "rnafm" in subdir else "RiNALMo" if "rinalmo" in subdir else subdir
    suffix = "masked" if subdir.endswith("_masked") else "unmasked"
    return f"{model} — {suffix}"

SETS = {}   # subdir -> (dist_matrix, isotypes)
for sd in subdirs:
    emb_path  = os.path.join(args.emb_dir, sd, "embeddings.npy")
    seqs_path = os.path.join(args.emb_dir, sd, "unique_sequences.npy")
    print(f"\n{make_title(sd)} centroids", flush=True)
    c, isos = compute_centroids(emb_path, seqs_path)
    SETS[sd] = (dist_fn(c), isos)

# ── Step 3: plot — 2-column grid, one panel per embedding set ─────────────────
n        = len(SETS)
n_cols   = 2
n_rows   = (n + 1) // 2
dist_label = "Cosine distance" if args.distance == "cosine" else "Euclidean distance"

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(9 * n_cols, 8 * n_rows),
                         squeeze=False)
axes_flat = axes.flatten()

for i, (sd, (D, isos)) in enumerate(SETS.items()):
    ax = axes_flat[i]
    im = ax.imshow(D, cmap="viridis", vmin=0, vmax=D.max())
    ax.set_xticks(range(len(isos))); ax.set_xticklabels(isos, rotation=90, fontsize=8)
    ax.set_yticks(range(len(isos))); ax.set_yticklabels(isos, fontsize=8)
    ax.set_title(make_title(sd), fontsize=13)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=dist_label)

for ax in axes_flat[n:]:
    ax.set_visible(False)

fig.suptitle(
    f"Isotype centroid pairwise {dist_label.lower()}",
    fontsize=15, y=1.01,
)
fig.tight_layout()

out_pdf = os.path.join(args.outdir, f"centroid_heatmap_{args.distance}.pdf")
out_png = out_pdf.replace(".pdf", ".png")
fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
fig.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
