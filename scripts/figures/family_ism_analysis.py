#!/usr/bin/env python3
"""
family_ism_analysis.py
----------------------
Build clade-level ISM embeddings and identify clades whose identity elements
diverge from the global bacterial ISM logo.

Works at any taxonomic level (order, family, class, …) via --clade-col.

For each (clade, isotype) with a computed ISM saliency file:
  1. Map the raw-position ISM saliency to Sprinzl positions
  2. Compare to the global Bacteria ISM for that isotype (cosine distance)
  3. Build a (clade × isotype) embedding matrix

Outputs
-------
  {outdir}/
    embeddings.npz                   clade ISM at Sprinzl positions
    embeddings_metadata.csv          clade, isotype, n_seqs, cosine_dist_from_global
    all_pairs_cosine.csv             pairwise cosine distances per isotype
    most_divergent_from_global.png   top divergent (clade, isotype) bars
    umap_clades.png                  UMAP coloured by isotype + divergence
    cosine_sim/cosine_sim_{iso}.png  per-isotype clade similarity heatmaps
    divergent_examples/              logo comparisons: global vs clade ISM

Usage
-----
    python scripts/analysis/family_ism_analysis.py
    python scripts/analysis/family_ism_analysis.py --clade-col order
    python scripts/analysis/family_ism_analysis.py --clade-col family --top-n 30
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
import logomaker
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
import umap as umap_lib

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

parser = argparse.ArgumentParser()
parser.add_argument("--clade-col",      default="order",
                    help="Taxonomic column used when running compute_ism_clade.py "
                         "(order, family, class, …)")
parser.add_argument("--ism-dir",        default=None,
                    help="Directory with gnn_ism_saliency_{iso}_{clade}.npy files "
                         "(default: gnn_final/{clade_col}_ism)")
parser.add_argument("--global-ism-dir", default=f"{DATA_ROOT}/gnn_final",
                    help="Directory with gnn_ism_saliency_{iso}_Bacteria.npy files")
parser.add_argument("--csv",            default=f"{DATA_ROOT}/data/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--outdir",         default=None,
                    help="Output directory (default: {clade_col}_ism_embeddings/bacteria)")
parser.add_argument("--top-n",          type=int, default=20,
                    help="Top N divergent (clade, isotype) pairs for example logos")
parser.add_argument("--min-sal",        type=float, default=0.002,
                    help="Min |saliency| to show in logo plots")
args = parser.parse_args()

if args.ism_dir is None:
    args.ism_dir = f"{DATA_ROOT}/gnn_final/{args.clade_col}_ism"
if args.outdir is None:
    args.outdir = f"{DATA_ROOT}/{args.clade_col}_ism_embeddings/bacteria"

os.makedirs(args.outdir, exist_ok=True)
os.makedirs(os.path.join(args.outdir, "cosine_sim"), exist_ok=True)
os.makedirs(os.path.join(args.outdir, "divergent_examples"), exist_ok=True)

NUCS = ["A", "C", "G", "U"]

# ── Sprinzl position list ─────────────────────────────────────────────────────
SPRINZL_ALL = (
    [str(i) for i in range(1, 44)] +
    ["44", "45", "46", "47",
     "47a", "47b", "47c", "47d", "47e", "47f", "47g", "47h", "47i", "47j",
     "47k", "47l", "47m", "47n", "47o", "47p", "48"] +
    [str(i) for i in range(49, 74)] +
    ["17a", "20a", "20b"]
)
AC_POSITIONS = {"34", "35", "36"}
ANALYSIS_POS = [p for p in SPRINZL_ALL if p not in AC_POSITIONS]
n_pos        = len(ANALYSIS_POS)
sp_to_idx    = {sp: i for i, sp in enumerate(ANALYSIS_POS)}


# ── Sprinzl map builder (matches plot_ism_logos.py, includes G-1 overhang) ───
def parse_pairs(ss):
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == ">": stack.append(i)
        elif c == "<":
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs


def build_sprinzl_map(ss):
    pairs      = parse_pairs(ss)
    five_prime = sorted(i for i in pairs if i < pairs[i])
    stems, cur = [], []
    for p in five_prime:
        if not cur:
            cur = [p]
        elif p == cur[-1] + 1 and pairs[p] == pairs[cur[-1]] - 1:
            cur.append(p)
        else:
            stems.append(cur); cur = [p]
    if cur: stems.append(cur)
    if len(stems) < 4: return {}
    sp   = {}
    acc5 = stems[0]
    acc3 = sorted(pairs[p] for p in acc5)
    for i, p in enumerate(reversed([p for p in range(acc5[0]) if p not in pairs])):
        sp[p] = str(-(i + 1))
    for i, p in enumerate(acc5): sp[p] = str(i + 1)
    for i, p in enumerate(acc3): sp[p] = str(66 + i)
    disc = sorted(p for p in range(acc3[-1] + 1, len(ss)) if p not in pairs)
    for i, p in enumerate(disc): sp[p] = str(73 + i)
    d5 = stems[1]
    for i, p in enumerate([p for p in range(acc5[-1] + 1, d5[0]) if p not in pairs]):
        sp[p] = str(8 + i)
    d3 = sorted(pairs[p] for p in d5); nd = len(d5)
    for i, p in enumerate(d5): sp[p] = str(10 + i)
    for i, p in enumerate(d3): sp[p] = str(25 - nd + 1 + i)
    d_loop = [p for p in range(d5[-1] + 1, d3[0]) if p not in pairs]
    for i, p in enumerate(d_loop):
        labs = ["14", "15", "16", "17", "17a", "18", "19", "20", "20a", "20b", "21"]
        if i < len(labs): sp[p] = labs[i]
    ac5 = stems[2]
    lk2 = [p for p in range(d3[-1] + 1, ac5[0]) if p not in pairs]
    if lk2: sp[lk2[-1]] = "26"
    ac3 = sorted(pairs[p] for p in ac5)
    for i, p in enumerate(ac5): sp[p] = str(27 + i)
    for i, p in enumerate(ac3): sp[p] = str(39 + i)
    for i, p in enumerate([p for p in range(ac5[-1] + 1, ac3[0]) if p not in pairs]):
        sp[p] = str(32 + i)
    t5 = stems[-1]; t3 = sorted(pairs[p] for p in t5)
    for i, p in enumerate(t5): sp[p] = str(49 + i)
    for i, p in enumerate(t3): sp[p] = str(61 + i)
    for i, p in enumerate([p for p in range(t5[-1] + 1, t3[0]) if p not in pairs]):
        sp[p] = str(54 + i)
    var = [p for p in range(ac3[-1] + 1, t5[0])]
    if len(stems) == 5:
        for i, p in enumerate(var): sp[p] = f"e{i + 1}"
    else:
        for i, p in enumerate([p for p in var if p not in pairs]): sp[p] = str(44 + i)
    return sp


def ism_to_sprinzl(raw_npy: np.ndarray, sp_map: dict) -> np.ndarray:
    vec = np.zeros((n_pos, 4), dtype=np.float32)
    for raw_pos, sp_label in sp_map.items():
        if sp_label in sp_to_idx and raw_pos < raw_npy.shape[0]:
            vec[sp_to_idx[sp_label]] = raw_npy[raw_pos]
    return vec


def cosine_sim_1d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


def cosine_sim_matrix(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    v = vecs / norms
    return (v @ v.T).astype(np.float64)


# ── 1. Load global Bacteria ISM ───────────────────────────────────────────────
print("=" * 65)
print("Step 1: Loading global Bacteria ISM")
print("=" * 65)
global_iso_files: dict[str, str] = {}
for fname in os.listdir(args.global_ism_dir):
    if fname.startswith("gnn_ism_saliency_") and fname.endswith("_Bacteria.npy"):
        iso = fname[len("gnn_ism_saliency_"):-len("_Bacteria.npy")]
        global_iso_files[iso] = os.path.join(args.global_ism_dir, fname)
print(f"  {len(global_iso_files)} global isotypes: {sorted(global_iso_files)}", flush=True)

# ── 2. Load CSV ───────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 2: Loading CSV and building Sprinzl maps")
print("=" * 65)
df = pd.read_csv(
    args.csv,
    usecols=["primary_sequence", "secondary_structure",
             "Anticodon_predicted_isotype", "domain", args.clade_col],
    low_memory=False,
).dropna(subset=["primary_sequence", "secondary_structure",
                 "Anticodon_predicted_isotype", "domain", args.clade_col])
bac_df = df[df["domain"] == "Bacteria"].copy()
clade_iso_counts = bac_df.groupby([args.clade_col, "Anticodon_predicted_isotype"]).size()
print(f"  {len(bac_df):,} bacterial rows, "
      f"{bac_df[args.clade_col].nunique()} {args.clade_col}s", flush=True)

print(f"  Building per-({args.clade_col}, isotype) Sprinzl maps …", flush=True)
sprinzl_maps: dict[tuple[str, str], dict] = {}
for (clade, iso), sub in bac_df.groupby([args.clade_col, "Anticodon_predicted_isotype"]):
    if len(sub) < 3:
        sprinzl_maps[(clade, iso)] = {}
        continue
    modal_len = sub["primary_sequence"].str.len().mode()[0]
    modal_ss  = (sub[sub["primary_sequence"].str.len() == modal_len]
                 ["secondary_structure"].mode()[0])
    sprinzl_maps[(clade, iso)] = build_sprinzl_map(modal_ss)

global_sprinzl: dict[str, dict] = {}
for iso, sub in bac_df.groupby("Anticodon_predicted_isotype"):
    modal_len = sub["primary_sequence"].str.len().mode()[0]
    modal_ss  = (sub[sub["primary_sequence"].str.len() == modal_len]
                 ["secondary_structure"].mode()[0])
    global_sprinzl[iso] = build_sprinzl_map(modal_ss)
print(f"  Built Sprinzl maps for {len(sprinzl_maps)} ({args.clade_col}, isotype) pairs",
      flush=True)

# ── 3. Discover per-clade ISM files ──────────────────────────────────────────
print("\n" + "=" * 65)
print(f"Step 3: Discovering per-{args.clade_col} ISM files")
print("=" * 65)
found: dict[tuple[str, str], str] = {}  # (iso, clade) → path
known_isos = sorted(global_iso_files, key=len, reverse=True)

if not os.path.exists(args.ism_dir):
    raise SystemExit(
        f"ISM directory not found: {args.ism_dir}\n"
        f"Run: python scripts/run_family_ism.py --clade-col {args.clade_col} | bash"
    )

for fname in os.listdir(args.ism_dir):
    if not (fname.startswith("gnn_ism_saliency_") and fname.endswith(".npy")):
        continue
    stem = fname[len("gnn_ism_saliency_"):-len(".npy")]
    for iso in known_isos:
        if stem.startswith(iso + "_"):
            clade = stem[len(iso) + 1:]
            found[(iso, clade)] = os.path.join(args.ism_dir, fname)
            break

if not found:
    raise SystemExit(
        f"No ISM files found in {args.ism_dir}.\n"
        f"Run: python scripts/run_family_ism.py --clade-col {args.clade_col} | bash"
    )

iso_list   = sorted(set(iso   for iso, _     in found))
clade_list = sorted(set(clade for _,   clade in found))
n_iso      = len(iso_list)
n_clades   = len(clade_list)
iso_idx    = {iso:   i for i, iso   in enumerate(iso_list)}
clade_idx  = {clade: i for i, clade in enumerate(clade_list)}
print(f"  {len(found)} files → {n_clades} {args.clade_col}s × {n_iso} isotypes", flush=True)

# ── 4. Map global ISM to Sprinzl ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 4: Mapping global ISM to Sprinzl positions")
print("=" * 65)
global_emb: dict[str, np.ndarray] = {}
for iso in iso_list:
    if iso not in global_iso_files:
        global_emb[iso] = np.zeros((n_pos, 4), dtype=np.float32)
        print(f"  {iso}: no global file — zeroed", flush=True)
        continue
    raw = np.load(global_iso_files[iso]).astype(np.float32)
    global_emb[iso] = ism_to_sprinzl(raw, global_sprinzl.get(iso, {}))
    n_filled = int(np.any(global_emb[iso] != 0, axis=1).sum())
    print(f"  {iso}: {n_filled}/{n_pos} Sprinzl positions filled", flush=True)

# ── 5. Build clade ISM embedding matrix ──────────────────────────────────────
print("\n" + "=" * 65)
print("Step 5: Building clade ISM embedding matrix")
print("=" * 65)
saliency = np.zeros((n_clades, n_iso, n_pos, 4), dtype=np.float32)

for (iso, clade), path in sorted(found.items()):
    ci     = clade_idx[clade]
    ii     = iso_idx[iso]
    raw    = np.load(path).astype(np.float32)
    sp_map = sprinzl_maps.get((clade, iso), {})
    if sp_map:
        saliency[ci, ii] = ism_to_sprinzl(raw, sp_map)

n_nonzero = sum(1 for ci in range(n_clades) for ii in range(n_iso)
                if np.any(saliency[ci, ii] != 0))
print(f"  Embedding shape: {saliency.shape}", flush=True)
print(f"  Non-zero ({args.clade_col}, isotype) cells: {n_nonzero}", flush=True)

# ── 6. Divergence from global ISM ────────────────────────────────────────────
print("\n" + "=" * 65)
print("Step 6: Computing divergence from global ISM")
print("=" * 65)
emb_flat    = saliency.reshape(n_clades, n_iso, -1)
global_flat = {iso: global_emb[iso].ravel() for iso in iso_list}

meta_rows = []
for (iso, clade) in found:
    ci      = clade_idx[clade]
    ii      = iso_idx[iso]
    clade_vec = saliency[ci, ii].ravel()
    if np.all(clade_vec == 0):
        continue
    n_seqs = int(clade_iso_counts.get((clade, iso), 0))
    dist   = 1.0 - cosine_sim_1d(clade_vec, global_flat[iso])
    meta_rows.append({
        args.clade_col:            clade,
        "isotype":                 iso,
        "clade_idx":               ci,
        "iso_idx":                 ii,
        "n_seqs":                  n_seqs,
        "cosine_dist_from_global": dist,
    })

meta_df = (pd.DataFrame(meta_rows)
           .sort_values("cosine_dist_from_global", ascending=False)
           .reset_index(drop=True))
meta_df.to_csv(os.path.join(args.outdir, "embeddings_metadata.csv"), index=False)
print(f"  {len(meta_df)} cells with ISM data", flush=True)
top = meta_df.iloc[0]
print(f"  Max divergence: {top['cosine_dist_from_global']:.4f} "
      f"({top[args.clade_col]} — {top['isotype']})", flush=True)

# ── 7. Save embeddings ────────────────────────────────────────────────────────
np.savez_compressed(
    os.path.join(args.outdir, "embeddings.npz"),
    saliency    = saliency,
    emb_flat    = emb_flat,
    clade_list  = np.array(clade_list),
    iso_list    = np.array(iso_list),
    analysis_pos= np.array(ANALYSIS_POS),
    global_emb  = np.stack([global_emb[iso] for iso in iso_list]),
)
print(f"  Saved embeddings.npz  {saliency.shape}", flush=True)

# ── 8. Pairwise cosine distances per isotype ──────────────────────────────────
print("\n" + "=" * 65)
print("Step 7: Pairwise cosine distances")
print("=" * 65)
all_pairs = []
for ii, iso in enumerate(iso_list):
    vecs    = emb_flat[:, ii, :]
    nonzero = np.any(vecs != 0, axis=1)
    valid   = np.where(nonzero)[0]
    if len(valid) < 2:
        continue
    vecs_nz  = vecs[valid].astype(np.float64)
    clades_nz = [clade_list[i] for i in valid]
    sim      = cosine_sim_matrix(vecs_nz)
    n_v      = len(valid)
    for a in range(n_v):
        for b in range(a + 1, n_v):
            all_pairs.append({
                "isotype":    iso,
                "clade_a":    clades_nz[a],
                "clade_b":    clades_nz[b],
                "cosine_sim": float(sim[a, b]),
                "cosine_dist": float(1.0 - sim[a, b]),
            })

if all_pairs:
    pairs_df = pd.DataFrame(all_pairs).sort_values("cosine_dist", ascending=False)
    pairs_df.to_csv(os.path.join(args.outdir, "all_pairs_cosine.csv"), index=False)
    print(f"  {len(pairs_df):,} pairs saved", flush=True)

# ── 9. Figure: most divergent from global ─────────────────────────────────────
print("\n" + "=" * 65)
print("Step 8: Figures")
print("=" * 65)
TOP_N    = args.top_n
top_rows = meta_df.drop_duplicates(args.clade_col).head(TOP_N)

iso_unique = sorted(top_rows["isotype"].unique())
colors_tab = plt.cm.tab20(np.linspace(0, 1, max(len(iso_unique), 1)))
iso_color  = {iso: colors_tab[i] for i, iso in enumerate(iso_unique)}
bar_colors = [iso_color[row["isotype"]] for _, row in top_rows.iterrows()]

fig, ax = plt.subplots(figsize=(max(10, len(top_rows) * 0.65), 6))
ax.bar(range(len(top_rows)), top_rows["cosine_dist_from_global"],
       color=bar_colors, edgecolor="gray", linewidth=0.5)
for i, (_, row) in enumerate(top_rows.iterrows()):
    ax.text(i, row["cosine_dist_from_global"] * 0.02,
            f"{row[args.clade_col]}\n({row['isotype']}, n={row['n_seqs']:,})",
            ha="center", va="bottom", fontsize=5, rotation=90, clip_on=True)
ax.legend(
    handles=[mpatches.Patch(facecolor=iso_color[iso], label=iso) for iso in iso_unique],
    title="Isotype", fontsize=7, ncol=4, loc="upper right", framealpha=0.9,
)
ax.set_xticks([])
ax.set_ylabel("Cosine distance from global Bacteria ISM", fontsize=11)
ax.set_title(
    f"Top {len(top_rows)} {args.clade_col}s with most divergent identity elements\n"
    f"(one entry per {args.clade_col}; isotype shown is the most divergent)",
    fontsize=12,
)
ax.yaxis.grid(True, linestyle="--", alpha=0.4); ax.set_axisbelow(True)
ax.set_ylim(0, top_rows["cosine_dist_from_global"].max() * 1.08)
fig.tight_layout()
fig.savefig(os.path.join(args.outdir, "most_divergent_from_global.png"),
            bbox_inches="tight", dpi=150)
plt.close(fig)
print("  Saved: most_divergent_from_global.png", flush=True)

# ── 10. UMAP ─────────────────────────────────────────────────────────────────
print("  Running UMAP …", flush=True)
rows_u: list[np.ndarray] = []
labels_clade_u: list[str] = []
labels_iso_u:   list[str] = []
for ci, clade in enumerate(clade_list):
    for ii, iso in enumerate(iso_list):
        v = emb_flat[ci, ii]
        if np.any(v != 0):
            rows_u.append(v)
            labels_clade_u.append(clade)
            labels_iso_u.append(iso)

if len(rows_u) >= 4:
    X      = np.array(rows_u, dtype=np.float64)
    norms  = np.linalg.norm(X, axis=1, keepdims=True)
    X_norm = X / np.where(norms == 0, 1.0, norms)
    reducer = umap_lib.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                            metric="cosine", random_state=42)
    X_umap = reducer.fit_transform(X_norm)

    iso_unique_u = sorted(set(labels_iso_u))
    colors_u     = plt.cm.tab20(np.linspace(0, 1, len(iso_unique_u)))
    iso_cmap     = {iso: colors_u[i] for i, iso in enumerate(iso_unique_u)}
    div_lookup   = {(r[args.clade_col], r["isotype"]): r["cosine_dist_from_global"]
                    for _, r in meta_df.iterrows()}
    divs         = np.array([div_lookup.get((c, iso), 0.0)
                             for c, iso in zip(labels_clade_u, labels_iso_u)])

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    for iso in iso_unique_u:
        mask = np.array([l == iso for l in labels_iso_u])
        if mask.any():
            axes[0].scatter(X_umap[mask, 0], X_umap[mask, 1],
                            label=iso, s=18, alpha=0.7, color=iso_cmap[iso], linewidths=0)
    axes[0].set_title(f"UMAP of {args.clade_col} ISM embeddings\n(coloured by isotype)",
                      fontsize=12)
    axes[0].legend(fontsize=6, ncol=3, framealpha=0.9, title="Isotype")

    sc = axes[1].scatter(X_umap[:, 0], X_umap[:, 1],
                         c=divs, cmap="hot_r", s=18, alpha=0.85, linewidths=0)
    plt.colorbar(sc, ax=axes[1], label="Cosine dist from global ISM")
    axes[1].set_title(
        f"UMAP of {args.clade_col} ISM embeddings\n(coloured by divergence from global)",
        fontsize=12,
    )
    for ax in axes:
        ax.set_xlabel("UMAP 1", fontsize=10); ax.set_ylabel("UMAP 2", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "umap_clades.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  Saved: umap_clades.png", flush=True)

# ── 11. Per-isotype cosine similarity heatmaps ───────────────────────────────
print("  Plotting per-isotype heatmaps …", flush=True)
for ii, iso in enumerate(iso_list):
    vecs    = emb_flat[:, ii, :].astype(np.float64)
    nonzero = np.any(vecs != 0, axis=1)
    valid   = np.where(nonzero)[0]
    if len(valid) < 2:
        continue
    vecs_nz   = vecs[valid]
    clades_nz = [clade_list[i] for i in valid]
    sim       = cosine_sim_matrix(vecs_nz)
    dist      = 1.0 - sim; np.fill_diagonal(dist, 0.0)
    try:
        Z        = linkage(squareform(np.clip(dist, 0, None)), method="average")
        order_cl = leaves_list(Z)
    except Exception:
        order_cl = list(range(len(clades_nz)))
    sim_s    = sim[np.ix_(order_cl, order_cl)]
    labels_s = [clades_nz[i] for i in order_cl]
    n        = len(labels_s)
    fs       = max(5, n * 0.22)
    fig, ax  = plt.subplots(figsize=(fs + 1, fs))
    im       = ax.imshow(sim_s, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    tick_fs  = max(2, 6 - n // 15)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels_s, rotation=90, fontsize=tick_fs)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels_s, fontsize=tick_fs)
    plt.colorbar(im, ax=ax, label="Cosine similarity")
    ax.set_title(f"{args.clade_col.capitalize()} ISM cosine similarity — {iso}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "cosine_sim", f"cosine_sim_{iso}.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
print("  Saved: cosine_sim/*.png", flush=True)

# ── 12. Divergent example logos ───────────────────────────────────────────────
print(f"  Plotting top {args.top_n} divergent {args.clade_col} logos vs global …", flush=True)


def plot_logo(ax, sal_2d: np.ndarray, title: str, min_sal: float = 0.002) -> None:
    active     = np.any(np.abs(sal_2d) >= min_sal, axis=1)
    active_idx = np.where(active)[0]
    if not active_idx.size:
        ax.text(0.5, 0.5, "no signal above threshold",
                ha="center", va="center", transform=ax.transAxes, fontsize=8)
        ax.set_title(title, fontsize=9)
        return
    sal_sub   = sal_2d[active_idx]
    sp_labels = [ANALYSIS_POS[i] for i in active_idx]
    df_logo   = pd.DataFrame(sal_sub, columns=["A", "C", "G", "U"])
    try:
        logo = logomaker.Logo(df_logo, ax=ax, color_scheme="classic",
                              vpad=0.05, width=0.9, font_name="DejaVu Sans")
        logo.style_spines(visible=False)
        ax.set_xticks(range(len(sp_labels)))
        ax.set_xticklabels(sp_labels, rotation=90, fontsize=max(3, 7 - len(sp_labels) // 10))
    except Exception as e:
        ax.text(0.5, 0.5, str(e), ha="center", va="center",
                transform=ax.transAxes, fontsize=6)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Δlogit", fontsize=8)
    ax.set_title(title, fontsize=9)


for _, row in meta_df.head(args.top_n).iterrows():
    clade  = row[args.clade_col]
    iso    = row["isotype"]
    ci     = int(row["clade_idx"])
    ii     = int(row["iso_idx"])
    n_seqs = int(row["n_seqs"])
    dist   = float(row["cosine_dist_from_global"])

    fig, axes = plt.subplots(2, 1, figsize=(22, 8), constrained_layout=True)
    plot_logo(axes[0], global_emb[iso],
              f"Global Bacteria ISM — {iso}", args.min_sal)
    plot_logo(axes[1], saliency[ci, ii],
              f"{clade} — {iso}  (n={n_seqs:,} seqs,  cosine dist = {dist:.3f})",
              args.min_sal)
    fig.suptitle(
        f"Identity element divergence: {clade}  vs  global Bacteria\n"
        f"Isotype: {iso}  |  {args.clade_col}",
        fontsize=13,
    )
    safe     = clade.replace("/", "_").replace(" ", "_")
    out_path = os.path.join(args.outdir, "divergent_examples",
                            f"{iso}_{safe}_vs_global.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  {iso}_{safe}_vs_global.png", flush=True)

print(f"\nDone. All results in: {args.outdir}/")
print("=" * 65)
