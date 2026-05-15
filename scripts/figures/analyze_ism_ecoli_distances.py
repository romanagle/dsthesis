#!/usr/bin/env python3
"""
analyze_ism_ecoli_distances.py
------------------------------
Compute ISM-weighted identity-embedding cosine distances from a single
reference genome (default: Escherichia coli GCF_003697165.2_genomic) to each
order in the existing order-level embedding space.

The order embeddings are the same signed log-ratio tensors produced by
identity_order_embeddings.py:
    log2(freq_order / freq_global_isotype)

This script builds an analogous per-genome embedding:
    log2(freq_genome / freq_global_isotype)

It then applies the same ISM weights used by analyze_ism_order_embeddings.py
and computes, for each isotype, cosine distance between the reference genome
and every order with nonzero signal.

Outputs
-------
  ecoli_vs_orders_cosine_ism.csv
  most_distant_to_ecoli_ism.{pdf,png}

Usage
-----
  python scripts/analyze_ism_ecoli_distances.py
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PSEUDOCOUNT = 0.5
NUC_MAP = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}


def cosine_sim_rows(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Cosine similarity between each row in matrix and vector."""
    matrix_norm = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    if vector_norm == 0:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    matrix_norm = np.where(matrix_norm == 0, 1.0, matrix_norm)
    return ((matrix / matrix_norm[:, None]) @ (vector / vector_norm)).astype(np.float64)


parser = argparse.ArgumentParser()
parser.add_argument("--emb-npz", default="order_identity_embeddings_bacteria/embeddings.npz",
                    help="Order-level embeddings.npz from identity_order_embeddings.py")
parser.add_argument("--ism-npz", default="order_identity_embeddings_bacteria/ism_embeddings.npz",
                    help="ISM embedding file from build_ism_embeddings.py")
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")
parser.add_argument("--sprinzl", default=f"{DATA_ROOT}/trnascan_GTDB_r226_v2_sprinzl.csv",
                    help="Sprinzl-aligned CSV with pos_* columns")
parser.add_argument("--taxonomy", default=f"{DATA_ROOT}/trnascan_GTDB_r226_v2_final_matched_taxonomy.csv",
                    help="Full taxonomy CSV with GenomeID and species")
parser.add_argument("--genome-id", default="GCF_003697165.2_genomic",
                    help="Reference genome to anchor distances against")
parser.add_argument("--genome-label", default="Escherichia coli",
                    help="Human-readable label for the anchor genome in plots")
parser.add_argument("--ism-domain", default=None,
                    help="ISM domain to use: Bacteria, Archaea, or avg. Default: Bacteria for bacteria-only embeddings, else auto")
parser.add_argument("--top-n-pairs", type=int, default=20)
parser.add_argument("--outdir", default=None,
                    help="Output directory (default: sibling dir next to --emb-npz)")
args = parser.parse_args()

if args.outdir is None:
    emb_dir = os.path.dirname(os.path.abspath(args.emb_npz))
    parent = os.path.dirname(emb_dir)
    base = os.path.basename(emb_dir)
    args.outdir = os.path.join(parent, f"{base}_ecoli")
os.makedirs(args.outdir, exist_ok=True)

print("Loading order embeddings ...", flush=True)
emb_data = np.load(args.emb_npz, allow_pickle=True)
order_embeddings = emb_data["embeddings"]      # (n_orders, n_iso, n_pos, 4)
pfm_global_freq = emb_data["pfm_global_freq"]  # (n_iso, n_pos, 4)
order_list = list(emb_data["order_list"])
iso_list = list(emb_data["iso_list"])
analysis_pos = list(emb_data["analysis_pos"])
n_orders, n_iso, n_pos, _ = order_embeddings.shape
if "order_domains" in emb_data:
    order_domains = list(emb_data["order_domains"].astype(str))
else:
    order_domains = ["Bacteria"] * n_orders
print(f"  {n_orders} orders x {n_iso} isotypes x {n_pos} positions", flush=True)

print("Loading ISM embeddings ...", flush=True)
ism_data = np.load(args.ism_npz, allow_pickle=True)
ism_sal = ism_data["saliency"]                 # (n_iso_ism, n_domains, n_pos, 4)
iso_ism = list(ism_data["iso_list"])
domain_list = list(ism_data["domain_list"])
if args.ism_domain is None:
    unique_emb_domains = sorted(set(order_domains))
    args.ism_domain = "avg" if len(unique_emb_domains) > 1 else unique_emb_domains[0]
if args.ism_domain == "avg":
    ism_weight_sal = ism_sal.mean(axis=1)
    print(f"  ISM weights: averaging all {len(domain_list)} domains", flush=True)
elif args.ism_domain in domain_list:
    ism_weight_sal = ism_sal[:, domain_list.index(args.ism_domain), :, :]
    print(f"  ISM weights: using domain '{args.ism_domain}'", flush=True)
else:
    raise SystemExit(f"--ism-domain '{args.ism_domain}' not in {domain_list} and not 'avg'")
iso_to_ism_idx = {iso: i for i, iso in enumerate(iso_ism)}

print("Loading taxonomy and Sprinzl tables ...", flush=True)
pos_cols = [f"pos_{p}" for p in analysis_pos]
target_tax_rows = []
seen_ids = set()
with open(args.taxonomy, newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        if row["GenomeID"] != args.genome_id:
            continue
        trna_id = row["tRNAscanID"]
        if trna_id in seen_ids:
            continue
        seen_ids.add(trna_id)
        target_tax_rows.append({
            "tRNAscanID": trna_id,
            "GenomeID": row["GenomeID"],
            "domain": row["domain"],
            "order": row["order"],
            "species": row["species"],
        })

if not target_tax_rows:
    raise SystemExit(f"Genome '{args.genome_id}' not found in taxonomy table")
target_tax = pd.DataFrame(target_tax_rows)

target_ids = set(target_tax["tRNAscanID"])
sprinzl_rows = []
with open(args.sprinzl, newline="") as handle:
    reader = csv.DictReader(handle)
    for row in reader:
        if row["tRNAscanID"] not in target_ids:
            continue
        sprinzl_rows.append({
            "tRNAscanID": row["tRNAscanID"],
            "sprinzl_ok": row["sprinzl_ok"],
            "Anticodon_predicted_isotype": row["Anticodon_predicted_isotype"],
            **{col: row.get(col, "") for col in pos_cols},
        })

if not sprinzl_rows:
    raise SystemExit(f"No Sprinzl rows found for genome '{args.genome_id}'")

sprinzl_df = pd.DataFrame(sprinzl_rows)
sprinzl_df = sprinzl_df[sprinzl_df["sprinzl_ok"].astype(float) == 1].copy()
sprinzl_df = sprinzl_df.dropna(subset=["Anticodon_predicted_isotype"]).copy()
sprinzl_df = sprinzl_df[
    ~sprinzl_df["Anticodon_predicted_isotype"].isin(["Undet", "Sup", ""])
].copy()

genome_df = sprinzl_df.merge(target_tax, on="tRNAscanID", how="inner")
if genome_df.empty:
    raise SystemExit(f"No Sprinzl-valid rows found for genome '{args.genome_id}'")

species_values = sorted(set(genome_df["species"].dropna().astype(str)))
if species_values and args.genome_label == "Escherichia coli":
    args.genome_label = species_values[0]

print(f"  Anchor genome: {args.genome_id} ({args.genome_label})", flush=True)
print(f"  Rows retained: {len(genome_df)}", flush=True)

genome_pfm = np.zeros((n_iso, n_pos, 4), dtype=np.float32)
genome_counts = np.zeros((n_iso, n_pos), dtype=np.int32)
iso_idx = {iso: i for i, iso in enumerate(iso_list)}

genome_df = genome_df[genome_df["Anticodon_predicted_isotype"].isin(iso_idx)].copy()
genome_df["_ii"] = genome_df["Anticodon_predicted_isotype"].map(iso_idx).astype(int)

print("Building anchor genome embedding ...", flush=True)
for pi, sp in enumerate(analysis_pos):
    col = f"pos_{sp}"
    sub = genome_df[["_ii", col]].copy()
    vals = sub[col].astype(str).str.upper().map(NUC_MAP)
    valid = vals.notna()
    if not valid.any():
        continue
    sub = sub[valid].copy()
    vals = vals[valid].astype(int)
    ii_arr = sub["_ii"].values
    ni_arr = vals.values

    flat_idx = ii_arr * 4 + ni_arr
    counts = np.bincount(flat_idx, minlength=n_iso * 4)
    genome_pfm[:, pi, :] = counts.reshape(n_iso, 4).astype(np.float32)
    genome_counts[:, pi] = genome_pfm[:, pi, :].sum(axis=1).astype(np.int32)

genome_total = genome_pfm.sum(axis=2, keepdims=True) + PSEUDOCOUNT * 4
genome_freq = (genome_pfm + PSEUDOCOUNT) / genome_total
with np.errstate(divide="ignore", invalid="ignore"):
    genome_embedding = np.log2(genome_freq / pfm_global_freq).astype(np.float32)
genome_embedding[genome_counts == 0] = 0.0

print("Applying ISM weights ...", flush=True)
weighted_orders = np.zeros_like(order_embeddings)
weighted_genome = np.zeros_like(genome_embedding)
missing_ism = []
for ii, iso in enumerate(iso_list):
    if iso in iso_to_ism_idx:
        weight = np.abs(ism_weight_sal[iso_to_ism_idx[iso], :, :])
    else:
        weight = np.ones((n_pos, 4), dtype=np.float32)
        missing_ism.append(iso)
    weighted_orders[:, ii, :, :] = order_embeddings[:, ii, :, :] * weight[None, :, :]
    weighted_genome[ii, :, :] = genome_embedding[ii, :, :] * weight
if missing_ism:
    print(f"  No ISM data for {missing_ism}; using unit weights", flush=True)

print("Computing per-isotype distances to anchor genome ...", flush=True)
rows = []
for ii, iso in enumerate(iso_list):
    order_vecs = weighted_orders[:, ii, :, :].reshape(n_orders, -1)
    genome_vec = weighted_genome[ii, :, :].reshape(-1)
    genome_nonzero = np.any(genome_vec != 0)
    order_nonzero = np.any(order_vecs != 0, axis=1)
    valid = np.where(order_nonzero & genome_nonzero)[0]
    if len(valid) == 0:
        continue
    sims = cosine_sim_rows(order_vecs[valid].astype(np.float64), genome_vec.astype(np.float64))
    for local_idx, oi in enumerate(valid):
        rows.append({
            "isotype": iso,
            "anchor_label": args.genome_label,
            "anchor_genome_id": args.genome_id,
            "anchor_species": args.genome_label,
            "anchor_order": genome_df["order"].dropna().astype(str).mode().iat[0]
                            if genome_df["order"].notna().any() else "Unknown",
            "target_order": order_list[oi],
            "target_domain": order_domains[oi],
            "anchor_trna_count": int((genome_df["Anticodon_predicted_isotype"] == iso).sum()),
            "anchor_nonzero_positions": int((genome_counts[ii] > 0).sum()),
            "cosine_sim": float(sims[local_idx]),
            "cosine_dist": float(1.0 - sims[local_idx]),
        })

dist_df = pd.DataFrame(rows).sort_values("cosine_dist", ascending=False)
csv_path = os.path.join(args.outdir, "ecoli_vs_orders_cosine_ism.csv")
dist_df.to_csv(csv_path, index=False)
print(f"  Saved {len(dist_df):,} comparisons -> {csv_path}", flush=True)
print(dist_df.head(args.top_n_pairs).to_string(index=False))

print("Plotting top distant orders ...", flush=True)
top_df = dist_df.head(args.top_n_pairs).copy()
iso_unique = sorted(top_df["isotype"].unique())
colors = plt.cm.tab20(np.linspace(0, 1, max(len(iso_unique), 1)))
iso_color = {iso: colors[i] for i, iso in enumerate(iso_unique)}
bar_colors = [iso_color[iso] for iso in top_df["isotype"]]

from matplotlib.patches import Patch
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

legend_handles = [Patch(facecolor=iso_color[iso], label=iso) for iso in iso_unique]

fig, ax = plt.subplots(figsize=(max(10, args.top_n_pairs * 0.55), 7))
ax.bar(range(len(top_df)), top_df["cosine_dist"],
       color=bar_colors, edgecolor="gray", linewidth=0.5)

for i, (_, row) in enumerate(top_df.iterrows()):
    domain_tag = "Arc" if row["target_domain"] == "Archaea" else "Bac"
    label = f"{row['isotype']}  {args.genome_label} vs {row['target_order']}[{domain_tag}]"
    ax.text(i, row["cosine_dist"] * 0.02, label,
            ha="center", va="bottom", fontsize=6,
            rotation=90, color="black", clip_on=True)

ax.legend(handles=legend_handles, title="Isotype", fontsize=7,
          ncol=4, loc="upper right", framealpha=0.9)
ax.set_xticks([])
ax.set_ylabel("Cosine distance (1 - similarity)", fontsize=11)
ax.set_title(
    f"Top {args.top_n_pairs} orders most distant from {args.genome_label} - ISM-weighted identity embeddings\n"
    f"(anchor genome: {args.genome_id}; ISM weights: {args.ism_domain})",
    fontsize=12,
)
ax.set_ylim(0, top_df["cosine_dist"].max() * 1.05)
plt.tight_layout()

for ext in ("png", "pdf"):
    out = os.path.join(args.outdir, f"most_distant_to_ecoli_ism.{ext}")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved -> {out}", flush=True)
plt.close(fig)

print("Done.", flush=True)
