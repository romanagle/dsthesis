#!/usr/bin/env python3
"""
identity_order_embeddings.py
----------------------------
Build nucleotide-specific identity-logo embeddings per (order, isotype),
then compute pairwise cosine similarity to find the most orthogonal orders.

The embedding for (order O, isotype I) is a flat vector of length n_pos × 4:
    emb[O, I, pos, nuc] = log2( freq[O, I, pos, nuc] / freq_global[I, pos, nuc] )

where:
  - freq[O, I, pos, nuc]        = fraction of sequences in order O with isotype I
                                   that have nucleotide nuc at Sprinzl position pos
  - freq_global[I, pos, nuc]    = same quantity averaged over ALL bacteria for isotype I
                                   (the reference / "average isotype I tRNA")

Values are:
  > 0  → order O enriches nuc at pos relative to the global average for isotype I
  < 0  → order O depletes nuc at pos
  ≈ 0  → order O matches the global average

Cosine similarity now spans [-1, 1]: two orders with OPPOSITE enrichment patterns
(e.g. one prefers G, the other A at the same position) will score near -1.

Positions 34/35/36 (anticodon) are excluded.

Outputs
-------
  order_identity_embeddings/
    embeddings.npz          raw embeddings (n_orders × n_iso × n_pos × 4)
    cosine_sim/             per-isotype heatmaps of pairwise cosine similarity
    most_distant_pairs.pdf  bar chart: top most-distant (order, isotype) pairs
    pca_orders.pdf          PCA of all order × isotype embeddings
    embeddings_metadata.csv order/isotype index for the saved numpy arrays

Usage
-----
    python scripts/identity_order_embeddings.py
    python scripts/identity_order_embeddings.py --top-n-orders 30 --isotypes Ala Gly
    python scripts/identity_order_embeddings.py --min-count 200 --min-pos-count 30
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--sprinzl",
                    default=f"{DATA_ROOT}/trnascan_GTDB_r226_v2_sprinzl.csv",
                    help="Sprinzl-aligned CSV with pos_X columns")
parser.add_argument("--taxonomy",
                    default=f"{DATA_ROOT}/trnascan_GTDB_r226_v2_final_matched_taxonomy.csv",
                    help="Full taxonomy CSV with tRNAscanID + domain + order columns")
parser.add_argument("--domain",   default="Bacteria",
                    choices=["Bacteria", "Archaea", "All"],
                    help="Which phylogenetic domain(s) to include (default: Bacteria)")
parser.add_argument("--outdir",   default=None,
                    help="Output directory (default: order_identity_embeddings_<domain>)")
parser.add_argument("--top-n-orders", type=int, default=None,
                    help="Use the N largest orders per domain (default: 50 Bacteria, 30 Archaea)")
parser.add_argument("--min-count",    type=int, default=50,
                    help="Min sequences per (order, isotype) to include (default: 50)")
parser.add_argument("--min-pos-count", type=int, default=20,
                    help="Min sequences at a position for (order,iso) to contribute (default: 20)")
parser.add_argument("--isotypes",     nargs="*", default=None,
                    help="Restrict to these isotypes (default: all)")
parser.add_argument("--culturable-only", action="store_true",
                    help="Keep only genomes with ncbi_genome_category == 'none' (cultured isolates)")
parser.add_argument("--bac-metadata",
                    default="/data/kate/group_II/bac120_metadata_r226.tsv",
                    help="GTDB bac120 metadata TSV (for --culturable-only)")
parser.add_argument("--arc-metadata",
                    default="/data/conner/GII/seq_2_taxonomy/GTDB_metadata/ar53_metadata.tsv",
                    help="GTDB ar53 metadata TSV (for --culturable-only)")
args = parser.parse_args()

# Resolve defaults that depend on domain choice
if args.outdir is None:
    tag = args.domain.lower()
    suffix = "_culturable" if args.culturable_only else ""
    args.outdir = f"order_identity_embeddings_{tag}{suffix}"
if args.top_n_orders is None:
    args.top_n_orders = 50 if args.domain == "Bacteria" else 30

os.makedirs(args.outdir, exist_ok=True)
os.makedirs(os.path.join(args.outdir, "cosine_sim"), exist_ok=True)

PSEUDOCOUNT  = 0.5
AC_POSITIONS = {"34", "35", "36"}

# ── Sprinzl positions (same as plot_identity_logos_signed.py) ─────────────────
SPRINZL_ALL = (
    [str(i) for i in range(1, 44)] +
    ["44", "45", "46", "47",
     "47a","47b","47c","47d","47e","47f","47g","47h","47i","47j",
     "47k","47l","47m","47n","47o","47p", "48"] +
    [str(i) for i in range(49, 74)] +
    ["17a", "20a", "20b"]
)
ANALYSIS_POS = [p for p in SPRINZL_ALL if p not in AC_POSITIONS]
POS_COLS     = [f"pos_{p}" for p in ANALYSIS_POS]
NUC_MAP      = {"A": 0, "C": 1, "G": 2, "T": 3, "U": 3}
NUCS         = ["A", "C", "G", "U"]
n_pos        = len(ANALYSIS_POS)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 65)
print("Step 1: Loading data")
print("=" * 65)

print(f"  Loading taxonomy: {args.taxonomy}", flush=True)
tax_df = pd.read_csv(
    args.taxonomy,
    usecols=["tRNAscanID", "domain", "order"],
    low_memory=False,
).drop_duplicates("tRNAscanID")

if args.domain == "All":
    keep_domains = {"Bacteria", "Archaea"}
else:
    keep_domains = {args.domain}
keep_ids = set(tax_df.loc[tax_df["domain"].isin(keep_domains), "tRNAscanID"])
print(f"  Domain(s): {keep_domains}  →  {len(keep_ids):,} tRNAscanIDs", flush=True)

if args.culturable_only:
    print("  Building culturable-genome filter (ncbi_genome_category == 'none') …", flush=True)
    meta_frames = []
    for meta_path in [args.bac_metadata, args.arc_metadata]:
        if os.path.exists(meta_path):
            mf = pd.read_csv(meta_path, sep="\t",
                             usecols=["accession", "ncbi_genome_category"],
                             low_memory=False)
            meta_frames.append(mf)
    meta_df = pd.concat(meta_frames, ignore_index=True)
    meta_df["accession_norm"] = (meta_df["accession"]
                                 .str.replace(r"^RS_|^GB_", "", regex=True))
    culturable_accs = set(
        meta_df.loc[meta_df["ncbi_genome_category"] == "none", "accession_norm"]
    )
    # Join GenomeID from taxonomy CSV to match against metadata accessions
    tax_df2 = pd.read_csv(
        args.taxonomy,
        usecols=["tRNAscanID", "GenomeID"],
        low_memory=False,
    ).drop_duplicates("tRNAscanID")
    tax_df = tax_df.merge(tax_df2, on="tRNAscanID", how="left")
    tax_df["accession_norm"] = (tax_df["GenomeID"]
                                .str.replace(r"_genomic$", "", regex=True))
    culturable_ids = set(
        tax_df.loc[tax_df["accession_norm"].isin(culturable_accs), "tRNAscanID"]
    )
    before = len(keep_ids)
    keep_ids = keep_ids & culturable_ids
    print(f"  Culturable filter: {before:,} → {len(keep_ids):,} tRNAscanIDs", flush=True)

print(f"  Loading sprinzl CSV: {args.sprinzl}", flush=True)
load_cols  = ["tRNAscanID", "sprinzl_ok", "Anticodon_predicted_isotype"] + POS_COLS
pos_dtype  = {c: "category" for c in POS_COLS}
sprinzl_df = pd.read_csv(
    args.sprinzl,
    usecols=load_cols,
    dtype=pos_dtype,
    low_memory=False,
)
print(f"  Sprinzl rows loaded: {len(sprinzl_df):,}", flush=True)

# Filter: sprinzl_ok, selected domain(s), valid isotype
sprinzl_df = sprinzl_df[sprinzl_df["sprinzl_ok"].astype(float) == 1].copy()
sprinzl_df = sprinzl_df[sprinzl_df["tRNAscanID"].isin(keep_ids)].copy()
sprinzl_df = sprinzl_df[
    ~sprinzl_df["Anticodon_predicted_isotype"].isin(["Undet", "Sup", ""])
].copy()
sprinzl_df = sprinzl_df.dropna(subset=["Anticodon_predicted_isotype"]).copy()
print(f"  Rows after filtering: {len(sprinzl_df):,}", flush=True)

# Merge in order and domain
print("  Merging taxonomy ...", flush=True)
order_map  = tax_df.set_index("tRNAscanID")["order"].to_dict()
domain_map = tax_df.set_index("tRNAscanID")["domain"].to_dict()
sprinzl_df["order"]  = sprinzl_df["tRNAscanID"].map(order_map)
sprinzl_df["domain"] = sprinzl_df["tRNAscanID"].map(domain_map)
sprinzl_df = sprinzl_df.dropna(subset=["order"]).copy()
print(f"  Rows with order label: {len(sprinzl_df):,}", flush=True)

# ── 2. Select top N orders ────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 2: Selecting top orders")
print("=" * 65)

order_counts = sprinzl_df["order"].value_counts()   # used later for UMAP point sizing

if args.domain == "All":
    # Select top N per domain independently, then union
    top_orders = []
    for dom in sorted(keep_domains):
        dom_counts = (sprinzl_df[sprinzl_df["domain"] == dom]["order"]
                      .value_counts())
        dom_top = list(dom_counts.head(args.top_n_orders).index)
        print(f"  Top {args.top_n_orders} {dom} orders "
              f"(smallest: {dom_counts[dom_top[-1]]:,} seqs)", flush=True)
        top_orders.extend(dom_top)
    top_orders = list(dict.fromkeys(top_orders))   # dedup, preserve order
else:
    top_orders = list(order_counts.head(args.top_n_orders).index)
    print(f"  Top {args.top_n_orders} orders selected "
          f"(smallest: {order_counts[top_orders[-1]]:,} seqs)", flush=True)

print(f"  Total orders selected: {len(top_orders)}", flush=True)
sprinzl_df = sprinzl_df[sprinzl_df["order"].isin(top_orders)].copy()
print(f"  Rows in selected orders: {len(sprinzl_df):,}", flush=True)

# ── 3. Build frequency matrices ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 3: Building frequency matrices")
print("=" * 65)

iso_list = sorted(sprinzl_df["Anticodon_predicted_isotype"].unique())
if args.isotypes:
    iso_list = [i for i in args.isotypes if i in iso_list]
    print(f"  Restricting to isotypes: {iso_list}", flush=True)
n_iso = len(iso_list)

order_list = top_orders
n_orders   = len(order_list)
order_idx  = {o: i for i, o in enumerate(order_list)}

# Build order→domain mapping now (needed for per-domain global freq in Step 4)
order_domain_arr = np.array([
    sprinzl_df[sprinzl_df["order"] == o]["domain"].iloc[0]
    if o in sprinzl_df["order"].values else "Unknown"
    for o in order_list
], dtype=object)
iso_idx    = {iso: k for k, iso in enumerate(iso_list)}

# pfm[order, iso, pos, nuc] and global bg[pos, nuc]
pfm = np.zeros((n_orders, n_iso, n_pos, 4), dtype=np.float32)
bg  = np.zeros((n_iso,    n_pos, 4),         dtype=np.float64)

# Add order + iso integer indices
sprinzl_df["_oi"] = sprinzl_df["order"].map(order_idx)
sprinzl_df["_ii"] = sprinzl_df["Anticodon_predicted_isotype"].map(iso_idx)
# Drop rows not in selected isotypes
sprinzl_df = sprinzl_df.dropna(subset=["_oi", "_ii"]).copy()
sprinzl_df["_oi"] = sprinzl_df["_oi"].astype(int)
sprinzl_df["_ii"] = sprinzl_df["_ii"].astype(int)

print(f"  Accumulating over {n_pos} positions × {n_iso} isotypes × {n_orders} orders ...",
      flush=True)

# bg is the GLOBAL cross-isotype background (all bacteria, all orders, all isotypes).
# Computed separately so IC measures per-isotype deviation from the global background.
bg = np.zeros((n_pos, 4), dtype=np.float64)   # shape: (n_pos, 4)

for pi, sp in enumerate(ANALYSIS_POS):
    col  = f"pos_{sp}"
    sub  = sprinzl_df[["_oi", "_ii", col]].copy()
    vals = sub[col].astype(str).str.upper().map(NUC_MAP)
    valid = vals.notna()
    sub  = sub[valid].copy()
    vals = vals[valid].astype(int)

    oi_arr = sub["_oi"].values
    ii_arr = sub["_ii"].values
    ni_arr = vals.values

    # per (order, iso, nuc) count
    flat_idx = oi_arr * (n_iso * 4) + ii_arr * 4 + ni_arr
    counts   = np.bincount(flat_idx, minlength=n_orders * n_iso * 4)
    pfm[:, :, pi, :] += counts.reshape(n_orders, n_iso, 4).astype(np.float32)

    # global background: sum over ALL sequences at this position (all isotypes combined)
    bg_cnt = np.bincount(ni_arr, minlength=4)
    bg[pi, :] += bg_cnt

    if (pi + 1) % 20 == 0:
        print(f"  {pi+1}/{n_pos} positions done", flush=True)

print("  Done accumulating.", flush=True)

# ── 4. Compute per-isotype global reference frequency ────────────────────────
print("\n" + "=" * 65)
print("Step 4: Computing per-isotype global reference frequency")
print("=" * 65)

# pfm_global_freq[iso, pos, nuc] = global reference frequency for log-ratio embeddings.
# When --domain All, compute separately per domain so bacterial orders are normalised
# against the bacterial background and archaeal orders against the archaeal background.
if args.domain == "All":
    domain_global_freq = {}
    for dom in ["Bacteria", "Archaea"]:
        dom_idx = [oi for oi, d in enumerate(order_domain_arr) if d == dom]
        if dom_idx:
            pfm_dom = pfm[dom_idx].sum(axis=0)          # (n_iso, n_pos, 4)
            pfm_dom_tot = pfm_dom.sum(axis=2, keepdims=True) + PSEUDOCOUNT * 4
            domain_global_freq[dom] = (pfm_dom + PSEUDOCOUNT) / pfm_dom_tot
            print(f"  Domain '{dom}' global reference: {len(dom_idx)} orders", flush=True)
    # pfm_global_freq saved to npz uses bacteria (anchor genome for ecoli analysis is bacteria)
    pfm_global_freq = domain_global_freq.get("Bacteria",
                      domain_global_freq[next(iter(domain_global_freq))])
else:
    domain_global_freq = None
    pfm_global       = pfm.sum(axis=0)                        # (n_iso, n_pos, 4)
    pfm_global_total = pfm_global.sum(axis=2, keepdims=True) + PSEUDOCOUNT * 4
    pfm_global_freq  = (pfm_global + PSEUDOCOUNT) / pfm_global_total
print(f"  Global reference computed for {n_iso} isotypes × {n_pos} positions", flush=True)

# ── 5. Build order-level signed log-ratio embeddings ─────────────────────────
print("\n" + "=" * 65)
print("Step 5: Building signed log-ratio embeddings")
print("=" * 65)

# Per-(order, isotype) frequency
pfm_total = pfm.sum(axis=3, keepdims=True) + PSEUDOCOUNT * 4  # (n_orders, n_iso, n_pos, 1)
pfm_freq  = (pfm + PSEUDOCOUNT) / pfm_total                    # (n_orders, n_iso, n_pos, 4)

# Signed log-ratio: log2(order_freq / global_isotype_freq)
# When --domain All, each order is normalised against its own domain's global frequency
# so bacterial and archaeal identity elements are measured on comparable scales.
embeddings = np.zeros_like(pfm_freq)
with np.errstate(divide="ignore", invalid="ignore"):
    if domain_global_freq is not None:
        for oi in range(n_orders):
            dom = order_domain_arr[oi]
            gfreq = domain_global_freq.get(dom, pfm_global_freq)
            embeddings[oi] = np.log2(pfm_freq[oi] / gfreq).astype(np.float32)
    else:
        embeddings = np.log2(
            pfm_freq / pfm_global_freq[np.newaxis, :, :, :]
        ).astype(np.float32)                                    # (n_orders, n_iso, n_pos, 4)

# Zero out positions where the order has too few sequences for reliable estimates
pos_counts = pfm.sum(axis=3)                                   # (n_orders, n_iso, n_pos)
sparse_mask = pos_counts < args.min_pos_count                  # True where unreliable
embeddings[sparse_mask] = 0.0

n_zeroed = sparse_mask.sum()
print(f"  Zeroed {n_zeroed:,} sparse (order, iso, pos) entries "
      f"(< {args.min_pos_count} seqs)", flush=True)

print(f"  Embedding shape: {embeddings.shape}  "
      f"(n_orders={n_orders}, n_iso={n_iso}, n_pos={n_pos}, 4 nucs)", flush=True)

# Save raw embeddings
emb_flat = embeddings.reshape(n_orders, n_iso, -1)             # (n_orders, n_iso, n_pos*4)
save_kwargs = dict(
    embeddings=embeddings,
    emb_flat=emb_flat,
    pfm=pfm,
    pfm_global_freq=pfm_global_freq,            # bacteria (or single-domain) global freq
    order_list=np.array(order_list),
    order_domains=order_domain_arr,
    iso_list=np.array(iso_list),
    analysis_pos=np.array(ANALYSIS_POS),
)
if domain_global_freq is not None:
    for dom, gfreq in domain_global_freq.items():
        save_kwargs[f"pfm_global_freq_{dom.lower()}"] = gfreq
np.savez(os.path.join(args.outdir, "embeddings.npz"), **save_kwargs)

# Metadata CSV
meta_rows = []
for oi, o in enumerate(order_list):
    for ii, iso in enumerate(iso_list):
        cnt = int(pfm[oi, ii, :, :].sum())
        meta_rows.append({"order": o, "domain": order_domain_arr[oi],
                          "isotype": iso,
                          "order_idx": oi, "iso_idx": ii,
                          "total_nuc_count": cnt})
meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(os.path.join(args.outdir, "embeddings_metadata.csv"), index=False)
print(f"  Saved embeddings to {args.outdir}/embeddings.npz", flush=True)

print(f"\nRun analyze_ism_order_embeddings.py --emb-npz {args.outdir}/embeddings.npz to produce ISM-weighted plots.")
print("=" * 65)

# ── 6. Figures ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 6: Generating figures")
print("=" * 65)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap as umap_lib
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

TOP_N_PAIRS = 20

def cosine_sim_matrix(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    v = vecs / norms
    return (v @ v.T).astype(np.float64)

# ── Pairwise cosine distances per isotype ────────────────────────────────────
print("  Computing pairwise cosine distances …", flush=True)
all_pairs = []
for ii, iso in enumerate(iso_list):
    vecs = emb_flat[:, ii, :]
    nonzero = np.any(vecs != 0, axis=1)
    valid = np.where(nonzero)[0]
    if len(valid) < 2:
        continue
    vecs_nz = vecs[valid].astype(np.float64)
    orders_nz = [order_list[i] for i in valid]
    sim = cosine_sim_matrix(vecs_nz)
    n_v = len(valid)
    for a in range(n_v):
        for b in range(a + 1, n_v):
            dom_a = order_domain_arr[valid[a]]
            dom_b = order_domain_arr[valid[b]]
            pair_type = (f"within-{dom_a}" if dom_a == dom_b else "cross-domain")
            all_pairs.append({
                "isotype": iso, "order_a": orders_nz[a], "order_b": orders_nz[b],
                "domain_a": dom_a, "domain_b": dom_b, "pair_type": pair_type,
                "cosine_sim": float(sim[a, b]), "cosine_dist": float(1.0 - sim[a, b]),
            })

pairs_df = (pd.DataFrame(all_pairs).sort_values("cosine_dist", ascending=False)
            if all_pairs else pd.DataFrame())
pairs_df.to_csv(os.path.join(args.outdir, "all_pairs_cosine.csv"), index=False)
print(f"  Total pairs: {len(pairs_df):,}", flush=True)

# ── Per-isotype cosine-similarity heatmaps ───────────────────────────────────
print("  Plotting per-isotype cosine similarity heatmaps …", flush=True)
for ii, iso in enumerate(iso_list):
    vecs = emb_flat[:, ii, :].astype(np.float64)
    nonzero = np.any(vecs != 0, axis=1)
    valid = np.where(nonzero)[0]
    if len(valid) < 2:
        continue
    vecs_nz = vecs[valid]
    orders_nz = [order_list[i] for i in valid]
    sim = cosine_sim_matrix(vecs_nz)
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)
    try:
        Z = linkage(squareform(np.clip(dist, 0, None)), method="average")
        order_idx_cl = leaves_list(Z)
    except Exception:
        order_idx_cl = list(range(len(orders_nz)))
    sim_sorted = sim[np.ix_(order_idx_cl, order_idx_cl)]
    labels_sorted = [orders_nz[i] for i in order_idx_cl]
    n = len(labels_sorted)
    fig_size = max(5, n * 0.28)
    fig, ax = plt.subplots(figsize=(fig_size + 1, fig_size))
    im = ax.imshow(sim_sorted, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels_sorted, rotation=90, fontsize=max(3, 7 - n // 10))
    ax.set_yticks(range(n)); ax.set_yticklabels(labels_sorted, fontsize=max(3, 7 - n // 10))
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title(f"Identity-embedding cosine similarity — {iso}", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, "cosine_sim", f"cosine_sim_{iso}.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
print("  Saved cosine_sim/*.{pdf,png}", flush=True)

# ── Most-distant pairs bar chart ─────────────────────────────────────────────
if not pairs_df.empty:
    print("  Plotting most-distant pairs …", flush=True)
    top_pairs = pairs_df.head(TOP_N_PAIRS).copy()
    has_cross = "pair_type" in top_pairs.columns and top_pairs["pair_type"].nunique() > 1
    if has_cross:
        type_palette = {"within-Bacteria": "#4878CF", "within-Archaea": "#D65F5F", "cross-domain": "#6ACC65"}
        bar_colors = [type_palette.get(t, "#888888") for t in top_pairs["pair_type"]]
        legend_handles = [Patch(facecolor=c, label=t) for t, c in type_palette.items() if t in top_pairs["pair_type"].values]
        legend_title = "Pair type"
    else:
        iso_unique = sorted(top_pairs["isotype"].unique())
        colors = plt.cm.tab20(np.linspace(0, 1, max(len(iso_unique), 1)))
        iso_color = {iso: colors[i] for i, iso in enumerate(iso_unique)}
        bar_colors = [iso_color[iso] for iso in top_pairs["isotype"]]
        legend_handles = [Patch(facecolor=iso_color[iso], label=iso) for iso in sorted(iso_color)]
        legend_title = "Isotype"
    fig, ax = plt.subplots(figsize=(max(10, TOP_N_PAIRS * 0.55), 7))
    ax.bar(range(len(top_pairs)), top_pairs["cosine_dist"], color=bar_colors, edgecolor="gray", linewidth=0.5)
    dom_abbrev = {"Bacteria": "Bac", "Archaea": "Arc", "Unknown": "?"}
    for i, (_, row) in enumerate(top_pairs.iterrows()):
        if row.get("domain_a", "") != row.get("domain_b", ""):
            tag_a = dom_abbrev.get(row["domain_a"], row["domain_a"])
            tag_b = dom_abbrev.get(row["domain_b"], row["domain_b"])
            label = f"{row['isotype']}  {row['order_a']}[{tag_a}] vs {row['order_b']}[{tag_b}]"
        else:
            label = f"{row['isotype']}  {row['order_a']} vs {row['order_b']}"
        ax.text(i, row["cosine_dist"] * 0.02, label, ha="center", va="bottom",
                fontsize=6, rotation=90, color="black", clip_on=True)
    ax.legend(handles=legend_handles, title=legend_title, fontsize=7, ncol=4, loc="upper right", framealpha=0.9)
    ax.set_xticks([])
    ax.set_ylabel("Cosine distance (1 − similarity)", fontsize=11)
    title_dom = "+".join(sorted(set(order_domain_arr)))
    ax.set_title(f"Top {TOP_N_PAIRS} most distant order pairs — identity embeddings\n({title_dom}, coloured by {'pair type' if has_cross else 'isotype'})", fontsize=12)
    ax.set_ylim(0, top_pairs["cosine_dist"].max() * 1.05)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.outdir, f"most_distant_pairs.{ext}"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved: most_distant_pairs.{pdf,png}", flush=True)

# ── UMAP: all order × isotype ─────────────────────────────────────────────────
print("  Running UMAP (all order × isotype) …", flush=True)
rows_u, row_orders_u, row_isos_u = [], [], []
for ii, iso in enumerate(iso_list):
    for oi, o in enumerate(order_list):
        v = emb_flat[oi, ii, :]
        if np.any(v != 0):
            rows_u.append(v)
            row_orders_u.append(o)
            row_isos_u.append(iso)

X = np.array(rows_u, dtype=np.float64)
norms = np.linalg.norm(X, axis=1, keepdims=True)
X_norm = X / np.where(norms == 0, 1.0, norms)

reducer = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
X_umap = reducer.fit_transform(X_norm)

iso_unique_u = sorted(set(row_isos_u))
colors_u = plt.cm.tab20(np.linspace(0, 1, len(iso_unique_u)))
iso_cmap = {iso: colors_u[i] for i, iso in enumerate(iso_unique_u)}
row_doms_u = [order_domain_arr[order_list.index(o)] for o in row_orders_u]
unique_emb_domains = sorted(set(order_domain_arr))

fig, ax = plt.subplots(figsize=(12, 9))
dom_markers = {"Bacteria": "o", "Archaea": "^", "Unknown": "s"}
for iso in iso_unique_u:
    for dom, marker in dom_markers.items():
        mask = np.array([l == iso and d == dom for l, d in zip(row_isos_u, row_doms_u)])
        if mask.any():
            ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                       label=f"{iso} ({dom})" if len(unique_emb_domains) > 1 else iso,
                       s=28, alpha=0.75, color=iso_cmap[iso], marker=marker, linewidths=0)
ax.set_xlabel("UMAP 1", fontsize=11)
ax.set_ylabel("UMAP 2", fontsize=11)
title_dom = "Bacteria+Archaea" if len(unique_emb_domains) > 1 else unique_emb_domains[0]
ax.set_title(f"UMAP of identity embeddings: {title_dom} orders × isotypes\n(cosine metric, each point = one order × isotype)"
             + ("\n▲ = Archaea  ● = Bacteria" if len(unique_emb_domains) > 1 else ""), fontsize=12)
ncol_leg = 3 if len(unique_emb_domains) == 1 else 4
ax.legend(fontsize=7, ncol=ncol_leg, framealpha=0.9, loc="best",
          title="Isotype" + (" / domain" if len(unique_emb_domains) > 1 else ""))
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(args.outdir, f"umap_orders.{ext}"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: umap_orders.{pdf,png}", flush=True)

# ── UMAP: per-isotype with labeled order points ───────────────────────────────
print("  Running per-isotype UMAPs …", flush=True)
N_ISO_U = len(iso_list)
N_COLS_U = 4
n_rows_u = (N_ISO_U + N_COLS_U - 1) // N_COLS_U

fig, axes = plt.subplots(n_rows_u, N_COLS_U, figsize=(6 * N_COLS_U, 5.5 * n_rows_u), squeeze=False)
for ax_i, iso in enumerate(iso_list):
    ax = axes[ax_i // N_COLS_U][ax_i % N_COLS_U]
    ii = iso_list.index(iso)
    vecs = emb_flat[:, ii, :].astype(np.float64)
    nonzero = np.any(vecs != 0, axis=1)
    n_valid = nonzero.sum()
    if n_valid < 4:
        ax.set_visible(False)
        continue
    vecs_nz = vecs[nonzero]
    ords_nz = [order_list[i] for i in np.where(nonzero)[0]]
    nrm = np.linalg.norm(vecs_nz, axis=1, keepdims=True)
    v_norm = vecs_nz / np.where(nrm == 0, 1.0, nrm)
    n_neighbors = min(10, n_valid - 1)
    try:
        red = umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=0.1,
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
                    fontsize=3.5, ha="center", va="bottom", xytext=(0, 3), textcoords="offset points")
for ax_i in range(N_ISO_U, n_rows_u * N_COLS_U):
    axes[ax_i // N_COLS_U][ax_i % N_COLS_U].set_visible(False)
culturable_label = " (culturable only)" if args.culturable_only else ""
fig.suptitle(f"Per-isotype UMAP — identity embeddings across orders{culturable_label}\n(cosine metric)",
             fontsize=13, y=1.01)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(args.outdir, f"umap_per_isotype.{ext}"), bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: umap_per_isotype.{pdf,png}", flush=True)

print(f"\nDone. All results in: {args.outdir}/")
print("=" * 65)


