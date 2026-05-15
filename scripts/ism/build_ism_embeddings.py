#!/usr/bin/env python3
"""
build_ism_embeddings.py
-----------------------
Convert per-(isotype, domain) GNN ISM saliency maps to Sprinzl-indexed
embeddings and save to order_identity_embeddings/ism_embeddings.npz.

For each (isotype, domain) pair:
  1. Load gnn_ism_saliency_{iso}_{domain}.npy  (shape: max_len × 4)
  2. Find the modal secondary structure in the CSV to build a raw→Sprinzl map
  3. Extract ISM values at the 89 Sprinzl analysis positions (anticodon excluded)

Output
------
  order_identity_embeddings/ism_embeddings.npz
    saliency     : (n_iso, n_domains, n_pos, 4)   ISM values at Sprinzl positions
    emb_flat     : (n_iso, n_domains, n_pos*4)     flattened
    iso_list     : (n_iso,)
    domain_list  : (n_domains,)
    analysis_pos : (n_pos,)                        Sprinzl position labels

Usage
-----
    python scripts/build_ism_embeddings.py
    python scripts/build_ism_embeddings.py --ism-dir gnn_final --outdir order_identity_embeddings
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ism-dir", default="gnn_final",
                    help="Directory containing gnn_ism_saliency_{iso}_{domain}.npy files")
parser.add_argument("--csv",     default="data/dedup/trnascan_GTDB_r226_dedup_taxonomy.csv",
                    help="CSV with primary_sequence, secondary_structure, "
                         "Anticodon_predicted_isotype, domain columns")
parser.add_argument("--outdir",  default="order_identity_embeddings")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ── Sprinzl position list (same as identity_order_embeddings.py) ─────────────
SPRINZL_ALL = (
    [str(i) for i in range(1, 44)] +
    ["44", "45", "46", "47",
     "47a","47b","47c","47d","47e","47f","47g","47h","47i","47j",
     "47k","47l","47m","47n","47o","47p", "48"] +
    [str(i) for i in range(49, 74)] +
    ["17a", "20a", "20b"]
)
AC_POSITIONS  = {"34", "35", "36"}
ANALYSIS_POS  = [p for p in SPRINZL_ALL if p not in AC_POSITIONS]
n_pos         = len(ANALYSIS_POS)
sp_to_idx     = {sp: i for i, sp in enumerate(ANALYSIS_POS)}
NUCS          = ["A", "C", "G", "U"]

# ── Sprinzl map builder (copied from plot_ism_logos.py) ──────────────────────
def parse_pairs(ss):
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == '>': stack.append(i)
        elif c == '<':
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs


def build_sprinzl_map(ss):
    pairs = parse_pairs(ss)
    n     = len(ss)
    five_prime = sorted(i for i in pairs if i < pairs[i])
    stems, cur = [], []
    for p in five_prime:
        if not cur: cur = [p]
        elif p == cur[-1]+1 and pairs[p] == pairs[cur[-1]]-1: cur.append(p)
        else: stems.append(cur); cur = [p]
    if cur: stems.append(cur)
    if len(stems) < 4: return {}
    sp = {}
    acc5 = stems[0]; acc3 = sorted(pairs[p] for p in acc5)
    for i, p in enumerate(acc5): sp[p] = str(i+1)
    for i, p in enumerate(acc3): sp[p] = str(66+i)
    disc = sorted(p for p in range(acc3[-1]+1, n) if p not in pairs)
    for i, p in enumerate(disc): sp[p] = str(73+i)
    d5 = stems[1]
    for i, p in enumerate([p for p in range(acc5[-1]+1, d5[0]) if p not in pairs]):
        sp[p] = str(8+i)
    d3 = sorted(pairs[p] for p in d5); nd = len(d5)
    for i, p in enumerate(d5): sp[p] = str(10+i)
    for i, p in enumerate(d3): sp[p] = str(25-nd+1+i)
    d_loop = [p for p in range(d5[-1]+1, d3[0]) if p not in pairs]
    for i, p in enumerate(d_loop):
        labs = ["14","15","16","17","17a","18","19","20","20a","20b","21"]
        if i < len(labs): sp[p] = labs[i]
    ac5 = stems[2]
    lk2 = [p for p in range(d3[-1]+1, ac5[0]) if p not in pairs]
    if lk2: sp[lk2[-1]] = "26"
    ac3 = sorted(pairs[p] for p in ac5)
    for i, p in enumerate(ac5): sp[p] = str(27+i)
    for i, p in enumerate(ac3): sp[p] = str(39+i)
    for i, p in enumerate([p for p in range(ac5[-1]+1, ac3[0]) if p not in pairs]):
        sp[p] = str(32+i)
    t5 = stems[-1]; t3 = sorted(pairs[p] for p in t5)
    for i, p in enumerate(t5): sp[p] = str(49+i)
    for i, p in enumerate(t3): sp[p] = str(61+i)
    for i, p in enumerate([p for p in range(t5[-1]+1, t3[0]) if p not in pairs]):
        sp[p] = str(54+i)
    var = [p for p in range(ac3[-1]+1, t5[0])]
    if len(stems) == 5:
        for i, p in enumerate(var): sp[p] = f"e{i+1}"
    else:
        for i, p in enumerate([p for p in var if p not in pairs]): sp[p] = str(44+i)
    return sp


# ── Discover ISM files ────────────────────────────────────────────────────────
found: dict[tuple[str, str], str] = {}  # (iso, domain) → path
for fname in os.listdir(args.ism_dir):
    if not (fname.startswith("gnn_ism_saliency_") and fname.endswith(".npy")):
        continue
    stem  = fname[len("gnn_ism_saliency_"):-len(".npy")]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        found[(parts[0], parts[1])] = os.path.join(args.ism_dir, fname)

if not found:
    raise SystemExit(f"No ISM files found in {args.ism_dir}")

iso_list    = sorted(set(iso for iso, _ in found))
domain_list = sorted(set(dom for _, dom in found))
n_iso       = len(iso_list)
n_domains   = len(domain_list)
iso_idx     = {iso: i for i, iso in enumerate(iso_list)}
dom_idx     = {dom: i for i, dom in enumerate(domain_list)}

print(f"Found {len(found)} ISM files")
print(f"  Isotypes ({n_iso}): {iso_list}")
print(f"  Domains  ({n_domains}): {domain_list}")
print(f"  Analysis positions: {n_pos}", flush=True)

# ── Load CSV for Sprinzl maps ─────────────────────────────────────────────────
print(f"\nLoading CSV: {args.csv} …", flush=True)
df = pd.read_csv(
    args.csv,
    usecols=["primary_sequence", "secondary_structure",
             "Anticodon_predicted_isotype", "domain"],
    low_memory=False,
).dropna(subset=["primary_sequence", "secondary_structure",
                 "Anticodon_predicted_isotype", "domain"])
print(f"  {len(df):,} rows loaded", flush=True)

# Pre-build Sprinzl maps for all (iso, domain) pairs
print("Building Sprinzl maps …", flush=True)
sprinzl_maps: dict[tuple[str, str], dict[int, str]] = {}
for iso, dom in found:
    sub = df[(df["Anticodon_predicted_isotype"] == iso) &
             (df["domain"] == dom)]
    if sub.empty:
        print(f"  [{iso}/{dom}] no rows in CSV — empty map", flush=True)
        sprinzl_maps[(iso, dom)] = {}
        continue
    modal_len = sub["primary_sequence"].str.len().mode()[0]
    modal_ss  = (sub[sub["primary_sequence"].str.len() == modal_len]
                 ["secondary_structure"].mode()[0])
    sprinzl_maps[(iso, dom)] = build_sprinzl_map(modal_ss)
    n_mapped = sum(1 for sp in sprinzl_maps[(iso, dom)].values()
                   if sp in sp_to_idx)
    print(f"  [{iso}/{dom}] modal_len={modal_len}  mapped_positions={n_mapped}", flush=True)

# ── Build embeddings ──────────────────────────────────────────────────────────
print("\nExtracting ISM values at Sprinzl positions …", flush=True)
saliency = np.zeros((n_iso, n_domains, n_pos, 4), dtype=np.float32)

for (iso, dom), path in sorted(found.items()):
    ii = iso_idx[iso]
    di = dom_idx[dom]

    ism = np.load(path).astype(np.float32)   # (max_len, 4)
    raw_to_sp = sprinzl_maps[(iso, dom)]

    n_filled = 0
    for raw_pos, sp_label in raw_to_sp.items():
        if sp_label not in sp_to_idx:
            continue
        if raw_pos >= ism.shape[0]:
            continue
        si = sp_to_idx[sp_label]
        saliency[ii, di, si, :] = ism[raw_pos, :]
        n_filled += 1

    print(f"  [{iso}/{dom}] filled {n_filled} positions", flush=True)

emb_flat = saliency.reshape(n_iso, n_domains, -1)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(args.outdir, "ism_embeddings.npz")
np.savez_compressed(
    out_path,
    saliency    = saliency,
    emb_flat    = emb_flat,
    iso_list    = np.array(iso_list),
    domain_list = np.array(domain_list),
    analysis_pos= np.array(ANALYSIS_POS),
)
print(f"\nSaved → {out_path}")
print(f"  saliency shape : {saliency.shape}")
print(f"  emb_flat shape : {emb_flat.shape}")
print(f"  iso_list       : {iso_list}")
print(f"  domain_list    : {domain_list}")
print(f"  analysis_pos   : {len(ANALYSIS_POS)} positions")
print("\nDone.")
