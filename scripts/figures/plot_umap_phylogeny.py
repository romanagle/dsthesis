#!/usr/bin/env python3
"""
plot_umap_phylogeny.py
----------------------
Load precomputed identity-logo embeddings and re-plot UMAP with points
colored by bacterial phylum, to ask: do orders with similar identity
elements belong to the same phylum?

Also computes a silhouette score per isotype (phylum as the label) to
quantify how well IE-based clustering recapitulates phylogeny, and plots
within-phylum vs between-phylum cosine distance distributions.

Usage
-----
    python scripts/plot_umap_phylogeny.py
    python scripts/plot_umap_phylogeny.py --emb order_identity_embeddings/embeddings.npz
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import umap
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")


parser = argparse.ArgumentParser()
parser.add_argument("--emb",      default="order_identity_embeddings/embeddings.npz")
parser.add_argument("--taxonomy", default=f"{DATA_ROOT}/trnascan_GTDB_r226_v2_final_matched_taxonomy.csv")
parser.add_argument("--outdir",   default="order_identity_embeddings")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ── 1. Load embeddings ────────────────────────────────────────────────────────
print("Loading embeddings …", flush=True)
data       = np.load(args.emb, allow_pickle=True)
emb_flat   = data["emb_flat"]                   # (n_orders, n_iso, n_pos*4)
order_list = list(data["order_list"])
iso_list   = list(data["iso_list"])
n_orders, n_iso, n_dims = emb_flat.shape
print(f"  {n_orders} orders × {n_iso} isotypes × {n_dims} dims", flush=True)

# ── 2. Map orders → phylum ────────────────────────────────────────────────────
print("Mapping orders to phyla …", flush=True)
tax = (pd.read_csv(args.taxonomy, usecols=["order", "phylum"], low_memory=False)
         .drop_duplicates("order"))
order_phylum = (pd.DataFrame({"order": order_list})
                  .merge(tax, on="order", how="left")["phylum"]
                  .fillna("Unknown").tolist())

phyla_unique = sorted(set(order_phylum))
n_phyla      = len(phyla_unique)
print(f"  {n_phyla} phyla: {phyla_unique}", flush=True)

# Assign colors — use tab20 + extras if needed
cmap_base  = plt.cm.get_cmap("tab20", max(n_phyla, 20))
phy_colors = {p: cmap_base(i / max(n_phyla - 1, 1)) for i, p in enumerate(phyla_unique)}

# ── 3. Global UMAP (all order × isotype points, colored by phylum) ────────────
print("\nBuilding global UMAP …", flush=True)

rows, row_order, row_iso, row_phylum = [], [], [], []
for ii, iso in enumerate(iso_list):
    for oi, o in enumerate(order_list):
        v = emb_flat[oi, ii, :]
        if np.any(v != 0):
            rows.append(v)
            row_order.append(o)
            row_iso.append(iso)
            row_phylum.append(order_phylum[oi])

X      = np.array(rows, dtype=np.float64)
norms  = np.linalg.norm(X, axis=1, keepdims=True)
X_norm = X / np.where(norms == 0, 1.0, norms)

print(f"  Running UMAP on {len(X_norm)} points …", flush=True)
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric="cosine", random_state=42)
X_umap  = reducer.fit_transform(X_norm)

fig, ax = plt.subplots(figsize=(13, 10))
for phy in phyla_unique:
    mask = np.array([p == phy for p in row_phylum])
    ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
               label=phy, color=phy_colors[phy],
               s=22, alpha=0.75, linewidths=0)

ax.set_xlabel("UMAP 1", fontsize=11)
ax.set_ylabel("UMAP 2", fontsize=11)
ax.set_title(
    "UMAP of identity-logo embeddings: bacterial orders × isotypes\n"
    "colored by phylum  (each point = one order × isotype pair)",
    fontsize=12,
)
ax.legend(fontsize=8, ncol=2, framealpha=0.9, loc="best", title="Phylum")
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(args.outdir, f"umap_phylum.{ext}"),
                bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: umap_phylum.{pdf,png}", flush=True)

# ── 4. Per-isotype UMAP colored by phylum ─────────────────────────────────────
print("\nPer-isotype UMAP colored by phylum …", flush=True)

N_COLS = 4
n_rows = (n_iso + N_COLS - 1) // N_COLS
fig, axes = plt.subplots(n_rows, N_COLS,
                          figsize=(6 * N_COLS, 5.5 * n_rows), squeeze=False)

iso_silhouettes = {}

for ax_i, iso in enumerate(iso_list):
    ax  = axes[ax_i // N_COLS][ax_i % N_COLS]
    ii  = iso_list.index(iso)

    vecs    = emb_flat[:, ii, :].astype(np.float64)
    nonzero = np.any(vecs != 0, axis=1)
    if nonzero.sum() < 4:
        ax.set_visible(False)
        continue

    vecs_nz  = vecs[nonzero]
    ords_nz  = [order_list[i] for i in np.where(nonzero)[0]]
    phyla_nz = [order_phylum[i] for i in np.where(nonzero)[0]]

    nrm     = np.linalg.norm(vecs_nz, axis=1, keepdims=True)
    v_norm  = vecs_nz / np.where(nrm == 0, 1.0, nrm)

    n_neighbors = min(10, len(ords_nz) - 1)
    try:
        red    = umap.UMAP(n_components=2, n_neighbors=n_neighbors,
                           min_dist=0.1, metric="cosine", random_state=42)
        coords = red.fit_transform(v_norm)
    except Exception:
        coords = np.random.randn(len(ords_nz), 2)

    # Silhouette score (phylum labels) — only if ≥ 2 phyla represented
    phyla_labels = np.array(phyla_nz)
    unique_phy   = np.unique(phyla_labels)
    if len(unique_phy) >= 2 and len(ords_nz) >= 4:
        try:
            sil = silhouette_score(v_norm, phyla_labels, metric="cosine")
            iso_silhouettes[iso] = sil
        except Exception:
            iso_silhouettes[iso] = np.nan
    else:
        iso_silhouettes[iso] = np.nan

    # Plot
    for phy in unique_phy:
        mask = phyla_labels == phy
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   color=phy_colors[phy], s=30, alpha=0.85,
                   linewidths=0, label=phy)

    # Label each point with order name
    for j, name in enumerate(ords_nz):
        ax.annotate(name[:13], (coords[j, 0], coords[j, 1]),
                    fontsize=3.5, ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points")

    sil_str = f"  sil={iso_silhouettes[iso]:.2f}" if not np.isnan(iso_silhouettes.get(iso, np.nan)) else ""
    ax.set_title(f"{iso}{sil_str}", fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=7)
    ax.set_ylabel("UMAP 2", fontsize=7)
    ax.tick_params(labelsize=6)

for ax_i in range(n_iso, n_rows * N_COLS):
    axes[ax_i // N_COLS][ax_i % N_COLS].set_visible(False)

# Shared phylum legend on the figure
legend_handles = [mpatches.Patch(color=phy_colors[p], label=p) for p in phyla_unique]
fig.legend(handles=legend_handles, title="Phylum", fontsize=7, ncol=3,
           loc="lower center", bbox_to_anchor=(0.5, -0.02), framealpha=0.9)

fig.suptitle(
    "Per-isotype UMAP: bacterial orders colored by phylum\n"
    "Silhouette score (sil) measures how well IE-based clustering recovers phylogeny",
    fontsize=13, y=1.01,
)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(args.outdir, f"umap_phylum_per_isotype.{ext}"),
                bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: umap_phylum_per_isotype.{pdf,png}", flush=True)

# ── 5. Silhouette scores bar chart ────────────────────────────────────────────
print("\nPlotting silhouette scores …", flush=True)

sil_df = (pd.DataFrame.from_dict(iso_silhouettes, orient="index", columns=["silhouette"])
            .dropna()
            .sort_values("silhouette", ascending=False))

fig, ax = plt.subplots(figsize=(10, 5))
colors_sil = ["#2563EB" if s > 0 else "#DC2626" for s in sil_df["silhouette"]]
ax.bar(range(len(sil_df)), sil_df["silhouette"], color=colors_sil,
       edgecolor="gray", linewidth=0.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(len(sil_df)))
ax.set_xticklabels(sil_df.index, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Silhouette score (phylum labels, cosine metric)", fontsize=11)
ax.set_title(
    "How well do identity-element embeddings recapitulate bacterial phylogeny?\n"
    "Silhouette score per isotype  (blue = phyla cluster by IE, red = scattered)",
    fontsize=12,
)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(args.outdir, f"silhouette_by_isotype.{ext}"),
                bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: silhouette_by_isotype.{pdf,png}", flush=True)

print("\nSilhouette scores (higher = IE clustering matches phylogeny better):")
print(sil_df.to_string())

# ── 6. Within vs between phylum cosine distance boxplot ───────────────────────
print("\nComputing within vs between phylum distances …", flush=True)

within, between, within_iso, between_iso = [], [], [], []

for ii, iso in enumerate(iso_list):
    vecs    = emb_flat[:, ii, :].astype(np.float64)
    nonzero = np.any(vecs != 0, axis=1)
    if nonzero.sum() < 4:
        continue
    vecs_nz  = vecs[nonzero]
    phyla_nz = np.array([order_phylum[i] for i in np.where(nonzero)[0]])
    nrm      = np.linalg.norm(vecs_nz, axis=1, keepdims=True)
    v_norm   = vecs_nz / np.where(nrm == 0, 1.0, nrm)

    dmat = cdist(v_norm, v_norm, metric="cosine")   # (k, k) cosine distances
    n    = len(phyla_nz)
    for a in range(n):
        for b in range(a + 1, n):
            d = float(dmat[a, b])
            if phyla_nz[a] == phyla_nz[b]:
                within.append(d);      within_iso.append(iso)
            else:
                between.append(d);     between_iso.append(iso)

within_df  = pd.DataFrame({"dist": within,  "iso": within_iso,  "type": "Within phylum"})
between_df = pd.DataFrame({"dist": between, "iso": between_iso, "type": "Between phylum"})
dist_df    = pd.concat([within_df, between_df], ignore_index=True)

# One boxplot per isotype, split within vs between
iso_order = sil_df.index.tolist()   # sorted by silhouette
fig, ax   = plt.subplots(figsize=(14, 5))

positions_within  = np.arange(len(iso_order)) * 2.5
positions_between = positions_within + 0.9

bp_w = ax.boxplot(
    [dist_df[(dist_df["iso"] == iso) & (dist_df["type"] == "Within phylum")]["dist"].values
     for iso in iso_order],
    positions=positions_within, widths=0.7, patch_artist=True,
    boxprops=dict(facecolor="#93C5FD", linewidth=0.8),
    medianprops=dict(color="#1D4ED8", linewidth=1.5),
    whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
    flierprops=dict(marker=".", markersize=2, alpha=0.4),
)
bp_b = ax.boxplot(
    [dist_df[(dist_df["iso"] == iso) & (dist_df["type"] == "Between phylum")]["dist"].values
     for iso in iso_order],
    positions=positions_between, widths=0.7, patch_artist=True,
    boxprops=dict(facecolor="#FCA5A5", linewidth=0.8),
    medianprops=dict(color="#DC2626", linewidth=1.5),
    whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
    flierprops=dict(marker=".", markersize=2, alpha=0.4),
)

ax.set_xticks(positions_within + 0.45)
ax.set_xticklabels(iso_order, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Cosine distance", fontsize=11)
ax.set_title(
    "Within-phylum vs between-phylum cosine distances per isotype\n"
    "(isotypes sorted by silhouette score — left = IEs most phylogenetically informative)",
    fontsize=12,
)
ax.legend([bp_w["boxes"][0], bp_b["boxes"][0]],
          ["Within phylum", "Between phylum"], fontsize=9, loc="upper right")
ax.yaxis.grid(True, linestyle="--", alpha=0.35)
ax.set_axisbelow(True)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(args.outdir, f"within_between_phylum_dist.{ext}"),
                bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: within_between_phylum_dist.{pdf,png}", flush=True)

print(f"\nDone. All results in: {args.outdir}/")
