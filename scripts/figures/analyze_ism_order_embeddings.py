#!/usr/bin/env python3
"""
analyze_ism_order_embeddings.py
--------------------------------
Identify which bacterial orders are most orthogonal using ISM-weighted
identity embeddings.

For each (order, isotype):
    embedding = PFM_log_ratio(order, isotype) * |ISM_saliency(isotype, Bacteria)|

The ISM magnitude acts as a per-position weight, suppressing non-identity
positions and amplifying positions the GNN identified as discriminative.

Then computes pairwise cosine distances between orders (same metric as
identity_order_embeddings.py) and produces:
  - all_pairs_cosine_ism.csv
  - most_distant_pairs_ism.{pdf,png}
  - umap_orders_ism.{pdf,png}
  - umap_per_isotype_ism.{pdf,png}
  - silhouette_by_isotype_ism.{pdf,png}

Usage
-----
    python scripts/analyze_ism_order_embeddings.py
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import umap

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--emb-npz",  default=None,
                    help="embeddings.npz (default: auto-detect from --outdir)")
parser.add_argument("--ism-npz",  default="order_identity_embeddings_bacteria/ism_embeddings.npz")
parser.add_argument("--outdir",   default=None,
                    help="Output directory (default: same dir as --emb-npz)")
parser.add_argument("--ism-domain", default=None,
                    help="Which domain ISM to use as weights. "
                         "'avg' averages Bacteria+Archaea. "
                         "Default: auto-detect from embeddings domain(s).")
parser.add_argument("--top-n-pairs", type=int, default=20)
args = parser.parse_args()

# Resolve emb-npz default
if args.emb_npz is None:
    args.emb_npz = "order_identity_embeddings_bacteria/embeddings.npz"

if args.outdir is None:
    args.outdir = os.path.dirname(os.path.abspath(args.emb_npz))
os.makedirs(args.outdir, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading embeddings …", flush=True)
emb_data     = np.load(args.emb_npz, allow_pickle=True)
pfm_logratio = emb_data["embeddings"]          # (n_orders, n_iso_pfm, n_pos, 4)
order_list   = list(emb_data["order_list"])
iso_pfm      = list(emb_data["iso_list"])
n_orders     = len(order_list)
n_pos        = pfm_logratio.shape[2]

# Per-order domain labels (present when built with --domain All or Archaea)
if "order_domains" in emb_data:
    order_domains = list(emb_data["order_domains"].astype(str))
else:
    # Legacy bacteria-only file — all orders are Bacteria
    order_domains = ["Bacteria"] * n_orders

unique_emb_domains = sorted(set(order_domains))
print(f"  {n_orders} orders  ×  {len(iso_pfm)} isotypes  ×  {n_pos} positions", flush=True)
print(f"  Order domains present: {unique_emb_domains}", flush=True)

print("Loading ISM embeddings …", flush=True)
ism_data    = np.load(args.ism_npz, allow_pickle=True)
ism_sal     = ism_data["saliency"]             # (n_iso_ism, n_domains, n_pos, 4)
iso_ism     = list(ism_data["iso_list"])
domain_list = list(ism_data["domain_list"])
analysis_pos= list(ism_data["analysis_pos"])
print(f"  {len(iso_ism)} isotypes  ×  {len(domain_list)} domains  ×  {n_pos} positions", flush=True)

# Auto-detect ISM domain strategy
if args.ism_domain is None:
    if len(unique_emb_domains) > 1:
        args.ism_domain = "avg"
    else:
        args.ism_domain = unique_emb_domains[0]

if args.ism_domain == "avg":
    # Average ISM saliency across all available domains
    ism_weight_sal = ism_sal.mean(axis=1)        # (n_iso_ism, n_pos, 4)
    print(f"  ISM weights: averaging all {len(domain_list)} domains", flush=True)
elif args.ism_domain in domain_list:
    dom_idx        = domain_list.index(args.ism_domain)
    ism_weight_sal = ism_sal[:, dom_idx, :, :]   # (n_iso_ism, n_pos, 4)
    print(f"  ISM weights: using domain '{args.ism_domain}'", flush=True)
else:
    raise SystemExit(f"--ism-domain '{args.ism_domain}' not in {domain_list} and not 'avg'")

# ── Build ISM-weighted embeddings ─────────────────────────────────────────────
# For each isotype, the ISM weight matrix is the absolute ISM saliency
# (n_pos × 4).  Positions with no ISM signal stay at 0 weight.
#
# Only isotypes present in BOTH datasets are used; Leu/Ser (not in GNN) are
# carried through at full weight (ISM weight = 1.0) so they aren't dropped.

print("\nBuilding ISM-weighted embeddings …", flush=True)

iso_to_ism_idx = {iso: i for i, iso in enumerate(iso_ism)}
weighted = np.zeros_like(pfm_logratio)   # (n_orders, n_iso_pfm, n_pos, 4)

for ii, iso in enumerate(iso_pfm):
    pfm_block = pfm_logratio[:, ii, :, :]  # (n_orders, n_pos, 4)

    if iso in iso_to_ism_idx:
        ism_block = ism_weight_sal[iso_to_ism_idx[iso], :, :]    # (n_pos, 4)
        weight    = np.abs(ism_block)       # (n_pos, 4)
    else:
        # Leu / Ser: no ISM — use PFM as-is (weight = 1)
        print(f"  [{iso}] no ISM data — using unweighted PFM", flush=True)
        weight = np.ones((n_pos, 4), dtype=np.float32)

    weighted[:, ii, :, :] = pfm_block * weight[np.newaxis, :, :]

emb_flat = weighted.reshape(n_orders, len(iso_pfm), -1)   # (n_orders, n_iso, n_pos*4)
print(f"  Weighted embedding shape: {emb_flat.shape}", flush=True)

# ── Cosine similarity helpers ─────────────────────────────────────────────────
def cosine_sim_matrix(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    v     = vecs / norms
    return (v @ v.T).astype(np.float64)

# ── Per-isotype cosine distances ──────────────────────────────────────────────
print("\nComputing pairwise cosine distances per isotype …", flush=True)
all_pairs = []

for ii, iso in enumerate(iso_pfm):
    vecs    = emb_flat[:, ii, :]                      # (n_orders, D)
    nonzero = np.any(vecs != 0, axis=1)
    valid   = np.where(nonzero)[0]
    if len(valid) < 2:
        continue
    vecs_nz    = vecs[valid]
    orders_nz  = [order_list[i] for i in valid]
    sim        = cosine_sim_matrix(vecs_nz.astype(np.float64))

    n_v = len(valid)
    for a in range(n_v):
        for b in range(a + 1, n_v):
            dom_a = order_domains[valid[a]]
            dom_b = order_domains[valid[b]]
            if dom_a == dom_b:
                pair_type = f"within-{dom_a}"
            else:
                pair_type = "cross-domain"
            all_pairs.append({
                "isotype":     iso,
                "order_a":     orders_nz[a],
                "order_b":     orders_nz[b],
                "domain_a":    dom_a,
                "domain_b":    dom_b,
                "pair_type":   pair_type,
                "cosine_sim":  float(sim[a, b]),
                "cosine_dist": float(1.0 - sim[a, b]),
            })

pairs_df = (pd.DataFrame(all_pairs)
            .sort_values("cosine_dist", ascending=False)
            if all_pairs else pd.DataFrame())
pairs_df.to_csv(os.path.join(args.outdir, "all_pairs_cosine_ism.csv"), index=False)
print(f"  Total pairs: {len(pairs_df):,}", flush=True)
print(f"\n  Top {args.top_n_pairs} most distant pairs:")
print(pairs_df.head(args.top_n_pairs).to_string(index=False))

# ── Plot: most-distant pairs bar chart ───────────────────────────────────────
print("\nPlotting most-distant pairs …", flush=True)
top_pairs  = pairs_df.head(args.top_n_pairs).copy()

# Colour by pair_type when cross-domain data present, else by isotype
has_cross = "pair_type" in top_pairs.columns and top_pairs["pair_type"].nunique() > 1
if has_cross:
    type_palette = {
        "within-Bacteria": "#4878CF",
        "within-Archaea":  "#D65F5F",
        "cross-domain":    "#6ACC65",
    }
    bar_colors = [type_palette.get(t, "#888888") for t in top_pairs["pair_type"]]
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=c, label=t)
                      for t, c in type_palette.items()
                      if t in top_pairs["pair_type"].values]
    legend_title = "Pair type"
else:
    iso_unique = sorted(top_pairs["isotype"].unique())
    colors     = plt.cm.tab20(np.linspace(0, 1, max(len(iso_unique), 1)))
    iso_color  = {iso: colors[i] for i, iso in enumerate(iso_unique)}
    bar_colors = [iso_color[iso] for iso in top_pairs["isotype"]]
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=iso_color[iso], label=iso)
                      for iso in sorted(iso_color)]
    legend_title = "Isotype"

fig, ax = plt.subplots(figsize=(max(10, args.top_n_pairs * 0.55), 7))
ax.bar(range(len(top_pairs)), top_pairs["cosine_dist"],
       color=bar_colors, edgecolor="gray", linewidth=0.5)

dom_abbrev = {"Bacteria": "Bac", "Archaea": "Arc", "Unknown": "?"}
for i, (_, row) in enumerate(top_pairs.iterrows()):
    if "domain_a" in row and row["domain_a"] != row.get("domain_b", row["domain_a"]):
        tag_a = dom_abbrev.get(row["domain_a"], row["domain_a"])
        tag_b = dom_abbrev.get(row["domain_b"], row["domain_b"])
        label = f"{row['isotype']}  {row['order_a']}[{tag_a}] vs {row['order_b']}[{tag_b}]"
    else:
        label = f"{row['isotype']}  {row['order_a']} vs {row['order_b']}"
    ax.text(i, row["cosine_dist"] * 0.02, label,
            ha="center", va="bottom", fontsize=6,
            rotation=90, color="black", clip_on=True)

ax.legend(handles=legend_handles, title=legend_title, fontsize=7,
          ncol=4, loc="upper right", framealpha=0.9)
ax.set_xticks([])
ax.set_ylabel("Cosine distance (1 − similarity)", fontsize=11)
ax.set_title(
    f"Top {args.top_n_pairs} most distant order pairs — ISM-weighted identity embeddings\n"
    f"(ISM weights: {args.ism_domain}, "
    + ("coloured by pair type)" if has_cross else "coloured by isotype)"),
    fontsize=12,
)
ax.set_ylim(0, top_pairs["cosine_dist"].max() * 1.05)
ax.yaxis.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(args.outdir, f"most_distant_pairs_ism.{ext}"),
                bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: most_distant_pairs_ism.{pdf,png}", flush=True)

# ── UMAP: all (order × isotype) points ───────────────────────────────────────
print("\nRunning UMAP (all order × isotype) …", flush=True)
rows, row_orders, row_isos = [], [], []
for ii, iso in enumerate(iso_pfm):
    for oi, o in enumerate(order_list):
        v = emb_flat[oi, ii, :]
        if np.any(v != 0):
            rows.append(v)
            row_orders.append(o)
            row_isos.append(iso)

X      = np.array(rows, dtype=np.float64)
norms  = np.linalg.norm(X, axis=1, keepdims=True)
X_norm = X / np.where(norms == 0, 1.0, norms)

reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric="cosine", random_state=42)
X_umap  = reducer.fit_transform(X_norm)

iso_unique_umap = sorted(set(row_isos))
colors_umap = plt.cm.tab20(np.linspace(0, 1, len(iso_unique_umap)))
iso_cmap    = {iso: colors_umap[i] for i, iso in enumerate(iso_unique_umap)}

# Build domain label for each UMAP point
row_domains_umap = [order_domains[order_list.index(o)] for o in row_orders]

fig, ax = plt.subplots(figsize=(12, 9))
dom_markers = {"Bacteria": "o", "Archaea": "^", "Unknown": "s"}
for iso in iso_unique_umap:
    for dom, marker in dom_markers.items():
        mask = np.array([l == iso and d == dom
                         for l, d in zip(row_isos, row_domains_umap)])
        if mask.any():
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                       label=f"{iso} ({dom})" if len(unique_emb_domains) > 1 else iso,
                       s=28, alpha=0.75, color=iso_cmap[iso],
                       marker=marker, linewidths=0)

ax.set_xlabel("UMAP 1", fontsize=11)
ax.set_ylabel("UMAP 2", fontsize=11)
title_dom = "Bacteria+Archaea" if len(unique_emb_domains) > 1 else unique_emb_domains[0]
ax.set_title(
    f"UMAP of ISM-weighted identity embeddings: {title_dom} orders × isotypes\n"
    f"(cosine metric, ISM weights: {args.ism_domain}, each point = one order × isotype)"
    + ("\n▲ = Archaea  ● = Bacteria" if len(unique_emb_domains) > 1 else ""),
    fontsize=12,
)
ncol_leg = 3 if len(unique_emb_domains) == 1 else 4
ax.legend(fontsize=7, ncol=ncol_leg, framealpha=0.9, loc="best",
          title="Isotype" + (" / domain" if len(unique_emb_domains) > 1 else ""))
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(args.outdir, f"umap_orders_ism.{ext}"),
                bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: umap_orders_ism.{pdf,png}", flush=True)

# ── UMAP: per-isotype with labeled order points ───────────────────────────────
print("\nRunning per-isotype UMAPs …", flush=True)
N_ISO_UMAP  = len(iso_pfm)
N_COLS_UMAP = 4
n_rows_umap = (N_ISO_UMAP + N_COLS_UMAP - 1) // N_COLS_UMAP

fig, axes = plt.subplots(n_rows_umap, N_COLS_UMAP,
                          figsize=(6 * N_COLS_UMAP, 5.5 * n_rows_umap),
                          squeeze=False)

for ax_i, iso in enumerate(iso_pfm):
    ax   = axes[ax_i // N_COLS_UMAP][ax_i % N_COLS_UMAP]
    ii   = iso_pfm.index(iso)
    vecs = emb_flat[:, ii, :].astype(np.float64)

    nonzero = np.any(vecs != 0, axis=1)
    n_valid = nonzero.sum()
    if n_valid < 4:
        ax.set_visible(False)
        continue

    vecs_nz  = vecs[nonzero]
    ords_nz  = [order_list[i] for i in np.where(nonzero)[0]]
    nrm      = np.linalg.norm(vecs_nz, axis=1, keepdims=True)
    v_norm   = vecs_nz / np.where(nrm == 0, 1.0, nrm)

    n_neighbors = min(10, n_valid - 1)
    try:
        red    = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
                           metric="cosine", random_state=42)
        coords = red.fit_transform(v_norm)
    except Exception:
        coords = np.random.randn(n_valid, 2)

    ax.scatter(coords[:, 0], coords[:, 1], s=28, alpha=0.85, linewidths=0)
    ax.set_title(iso, fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=7)
    ax.set_ylabel("UMAP 2", fontsize=7)
    ax.tick_params(labelsize=6)
    for j, name in enumerate(ords_nz):
        ax.annotate(name[:14], (coords[j, 0], coords[j, 1]),
                    fontsize=3.5, ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points")

for ax_i in range(N_ISO_UMAP, n_rows_umap * N_COLS_UMAP):
    axes[ax_i // N_COLS_UMAP][ax_i % N_COLS_UMAP].set_visible(False)

fig.suptitle(
    "Per-isotype UMAP — ISM-weighted identity embeddings across bacterial orders\n"
    f"(cosine metric, ISM domain: {args.ism_domain})",
    fontsize=13, y=1.01,
)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(args.outdir, f"umap_per_isotype_ism.{ext}"),
                bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: umap_per_isotype_ism.{pdf,png}", flush=True)

print(f"\nDone. All results in: {args.outdir}/")
