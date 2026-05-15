#!/usr/bin/env python3
"""
plot_centroid_distances.py
--------------------------
For each sequence, computes:
  - distance to its correct isotype centroid
  - mean distance to all incorrect isotype centroids

Plots overlapping distributions (violin + KDE) for RNA-FM unmasked vs masked,
side by side.

Usage
-----
    python scripts/plot_centroid_distances.py
"""

from __future__ import annotations

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")
import sys
sys.path.insert(0, os.path.join(DATA_ROOT, "scripts"))
from utils.iso_classes import ISOTYPE_ORDER as ISOTYPES, MERGE_ISOS


parser = argparse.ArgumentParser()
parser.add_argument("--input",   default=f"{DATA_ROOT}/trnascan_GTDB_r226_v2_final.csv")
parser.add_argument("--emb-dir", default="embeddings")
parser.add_argument("--outdir",  default="figures")
parser.add_argument("--n-samples", type=int, default=100_000,
                    help="Max sequences to use per model (default: 100k)")
parser.add_argument("--seed",    type=int, default=42)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)


# ── Build sequence → isotype lookup ──────────────────────────────────────────
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


def cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distance between rows of a and b (both normalised row-wise)."""
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return 1.0 - (a_n * b_n).sum(axis=1)


def compute_distances(emb_path, seqs_path, rng, n_max):
    print(f"  Loading {emb_path} …", flush=True)
    embs = np.load(emb_path).astype(np.float32)
    seqs = np.load(seqs_path)

    # assign isotype labels
    iso_labels = np.array([seq_to_iso.get(s, None) for s in seqs])
    valid = np.array([l is not None for l in iso_labels])
    embs = embs[valid]
    iso_labels = iso_labels[valid]
    print(f"  {valid.sum():,} sequences with known isotype  dim={embs.shape[1]}", flush=True)

    # subsample
    n = min(n_max, len(embs))
    idx = rng.choice(len(embs), size=n, replace=False)
    embs = embs[idx]
    iso_labels = iso_labels[idx]
    print(f"  Subsampled to {n:,}", flush=True)

    # compute centroids from this sample
    D = embs.shape[1]
    sums   = {iso: np.zeros(D, np.float64) for iso in ISOTYPES}
    counts = {iso: 0 for iso in ISOTYPES}
    for i, iso in enumerate(iso_labels):
        sums[iso]   += embs[i].astype(np.float64)
        counts[iso] += 1

    centroids = {}
    for iso in ISOTYPES:
        if counts[iso] > 0:
            centroids[iso] = (sums[iso] / counts[iso]).astype(np.float32)

    present_isos = list(centroids.keys())
    centroid_mat = np.stack([centroids[iso] for iso in present_isos])  # (K, D)

    # for each sequence compute dist-to-correct and mean-dist-to-incorrect
    emb_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
    cent_n = centroid_mat / (np.linalg.norm(centroid_mat, axis=1, keepdims=True) + 1e-10)
    sim_mat = emb_n @ cent_n.T          # (N, K)
    dist_mat = 1.0 - sim_mat            # (N, K)

    iso_to_idx = {iso: i for i, iso in enumerate(present_isos)}

    correct_dists   = []
    incorrect_dists = []

    for i, iso in enumerate(iso_labels):
        if iso not in iso_to_idx:
            continue
        ci = iso_to_idx[iso]
        correct_dists.append(dist_mat[i, ci])
        others = np.delete(dist_mat[i], ci)
        incorrect_dists.append(others.mean())

    return np.array(correct_dists), np.array(incorrect_dists)


def make_label(subdir):
    model  = "RNA-FM" if "rnafm" in subdir else "RiNALMo" if "rinalmo" in subdir else subdir
    suffix = "masked" if subdir.endswith("_masked") else "unmasked"
    return f"{model} {suffix}"

PANEL_ORDER = ["rnafm_unmasked", "rnafm_masked", "rinalmo_unmasked", "rinalmo_masked"]

rng = np.random.default_rng(args.seed)

results = {}
for subdir in PANEL_ORDER:
    emb_path  = os.path.join(args.emb_dir, subdir, "embeddings.npy")
    seqs_path = os.path.join(args.emb_dir, subdir, "unique_sequences.npy")
    if not (os.path.exists(emb_path) and os.path.exists(seqs_path)):
        print(f"SKIP {subdir!r}: embeddings not complete", flush=True)
        continue
    label = make_label(subdir)
    print(f"\n{label}", flush=True)
    cor, inc = compute_distances(emb_path, seqs_path, rng, args.n_samples)
    results[label] = (cor, inc)

if not results:
    raise SystemExit("No embedding sets found.")

# ── Plot ──────────────────────────────────────────────────────────────────────
n = len(results)
n_cols = min(2, n)
n_rows = (n + 1) // 2
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), sharey=False,
                         squeeze=False)
axes_flat = axes.flatten()
for ax in axes_flat[n:]:
    ax.set_visible(False)

COLOR_CORRECT   = "#2196F3"   # blue
COLOR_INCORRECT = "#F44336"   # red

for ax, (label, (cor, inc)) in zip(axes_flat, results.items()):
    x_min = min(cor.min(), inc.min()) - 0.01
    x_max = max(cor.max(), inc.max()) + 0.01
    xs = np.linspace(x_min, x_max, 500)

    for arr, color, name in [
        (cor, COLOR_CORRECT,   "Correct centroid"),
        (inc, COLOR_INCORRECT, "Incorrect centroids (mean)"),
    ]:
        kde = gaussian_kde(arr, bw_method=0.05)
        ys = kde(xs)
        ax.fill_between(xs, ys, alpha=0.35, color=color)
        ax.plot(xs, ys, color=color, lw=1.8, label=name)
        ax.axvline(np.median(arr), color=color, lw=1.2, ls="--", alpha=0.8)

    ax.set_title(label, fontsize=13)
    ax.set_xlabel("Cosine distance to centroid", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(bottom=0)

fig.suptitle(
    "Distance to correct vs. incorrect isotype centroid\n(dashed = median)",
    fontsize=14, y=1.02,
)
fig.tight_layout()

out_pdf = os.path.join(args.outdir, "centroid_distance_distributions.pdf")
out_png = out_pdf.replace(".pdf", ".png")
fig.savefig(out_pdf, bbox_inches="tight", dpi=150)
fig.savefig(out_png, bbox_inches="tight", dpi=150)
print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
