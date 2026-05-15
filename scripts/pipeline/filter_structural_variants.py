#!/usr/bin/env python3
"""
filter_structural_variants.py
------------------------------
For each (isotype × structural_variant), keep sequences only if ≥75% of
that group's sequences are concentrated in ≤3 bacterial orders.
Sequences whose structural variant is scattered across orders are dropped
as likely noise rather than biology.

Structural variant is defined as the combination of:
  - sprinzl_type     (type_I / type_II)
  - d_ins_count      (0-3: how many of 17a, 20a, 20b are occupied)
  - var_arm_length   (0-16: how many of 47a-47p are occupied)

Inputs
------
  --sprinzl     trnascan_GTDB_r226_v2_sprinzl.csv
  --taxonomy    /data/roma/trnascan_GTDB_r226_v2_final_matched_taxonomy.csv
  --output      data/sprinzl_filtered.csv

Parameters
----------
  --top-orders  3      max orders that must account for the threshold (default 3)
  --threshold   0.75   min fraction concentrated in top-N orders (default 0.75)
  --domain      Bacteria  restrict to this domain (default: Bacteria)

Usage
-----
    python scripts/filter_structural_variants.py
    python scripts/filter_structural_variants.py --top-orders 3 --threshold 0.75
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--sprinzl",    default="trnascan_GTDB_r226_v2_sprinzl.csv")
parser.add_argument("--taxonomy",   default=f"{DATA_ROOT}/trnascan_GTDB_r226_v2_final_matched_taxonomy.csv")
parser.add_argument("--output",     default="data/sprinzl_filtered.csv")
parser.add_argument("--top-orders", type=int,   default=3)
parser.add_argument("--threshold",  type=float, default=0.75)
parser.add_argument("--domain",     default=None,
                    help="Filter to a domain (e.g. Bacteria). Default: all domains.")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output), exist_ok=True)

D_INSERT   = ["17a", "20a", "20b"]
VAR_INSERT = ["47a","47b","47c","47d","47e","47f","47g",
              "47h","47i","47j","47k","47l","47m","47n","47o","47p"]

# ── 1. Load taxonomy ──────────────────────────────────────────────────────────
print("Loading taxonomy ...", flush=True)
tax = pd.read_csv(
    args.taxonomy,
    usecols=["tRNAscanID", "domain", "order"],
    low_memory=False,
).drop_duplicates("tRNAscanID")

if args.domain:
    tax = tax[tax["domain"] == args.domain]
    print(f"  {args.domain} genomes: {len(tax):,} tRNAscanIDs", flush=True)
else:
    print(f"  All domains: {len(tax):,} tRNAscanIDs", flush=True)

order_map = tax.set_index("tRNAscanID")["order"].to_dict()

# ── 2. Load sprinzl CSV ───────────────────────────────────────────────────────
print("Loading sprinzl CSV ...", flush=True)
load_cols = (
    ["tRNAscanID", "sprinzl_ok", "Anticodon_predicted_isotype", "sprinzl_type"]
    + [f"pos_{p}" for p in D_INSERT + VAR_INSERT]
)
df = pd.read_csv(args.sprinzl, usecols=load_cols, low_memory=False)
print(f"  Loaded: {len(df):,} rows", flush=True)

# ── 3. Filter: sprinzl_ok, domain, valid isotype ─────────────────────────────
df = df[df["sprinzl_ok"].astype(float) == 1].copy()
df = df[df["tRNAscanID"].isin(order_map)].copy()
df = df[~df["Anticodon_predicted_isotype"].isin(["Undet", "Sup", ""])].copy()
df = df.dropna(subset=["Anticodon_predicted_isotype"]).copy()
print(f"  After filters: {len(df):,} rows", flush=True)

# ── 4. Join order ─────────────────────────────────────────────────────────────
df["order"] = df["tRNAscanID"].map(order_map)
df = df.dropna(subset=["order"]).copy()
print(f"  With order label: {len(df):,} rows", flush=True)

# ── 5. Compute structural variant signature ───────────────────────────────────
print("Computing structural variant signatures ...", flush=True)

for p in D_INSERT + VAR_INSERT:
    col = f"pos_{p}"
    df[col] = df[col].astype(str).str.strip().replace({"": "__", "nan": "__"})

d_cols   = [f"pos_{p}" for p in D_INSERT]
var_cols = [f"pos_{p}" for p in VAR_INSERT]

df["d_ins_count"]   = (df[d_cols]   != "__").sum(axis=1).astype(np.int8)
df["var_arm_length"] = (df[var_cols] != "__").sum(axis=1).astype(np.int8)
df["struct_variant"] = (
    df["sprinzl_type"].astype(str) + "_d" +
    df["d_ins_count"].astype(str)  + "_v" +
    df["var_arm_length"].astype(str)
)

# ── 6. For each (isotype × struct_variant), test order concentration ──────────
# The dominant variant per isotype is always kept (it's the canonical structure,
# naturally spread across many orders). Only minority variants are tested.
print(
    f"Filtering minority variants: keep if top {args.top_orders} orders "
    f"account for ≥{args.threshold*100:.0f}% of sequences ...",
    flush=True,
)

# Find the dominant (most common) variant per isotype
dominant_variant = (
    df.groupby(["Anticodon_predicted_isotype", "struct_variant"])
    .size()
    .reset_index(name="n")
    .sort_values("n", ascending=False)
    .drop_duplicates("Anticodon_predicted_isotype")
    .set_index("Anticodon_predicted_isotype")["struct_variant"]
    .to_dict()
)

keep_mask = pd.Series(False, index=df.index)
summary_rows = []

for (iso, variant), grp in df.groupby(
    ["Anticodon_predicted_isotype", "struct_variant"], sort=False
):
    n = len(grp)
    order_counts = grp["order"].value_counts()
    top_n_frac   = order_counts.iloc[: args.top_orders].sum() / n
    is_dominant  = dominant_variant.get(iso) == variant

    # Always keep the dominant variant; apply concentration test to minorities
    keep = is_dominant or (top_n_frac >= args.threshold)
    if keep:
        keep_mask.loc[grp.index] = True

    summary_rows.append({
        "isotype":         iso,
        "struct_variant":  variant,
        "n_seqs":          n,
        "is_dominant":     is_dominant,
        "top_orders_frac": round(top_n_frac, 4),
        "top_orders":      ", ".join(order_counts.index[: args.top_orders].tolist()),
        "kept":            keep,
    })

summary_df = pd.DataFrame(summary_rows).sort_values(
    ["isotype", "n_seqs"], ascending=[True, False]
)

# ── 7. Report ─────────────────────────────────────────────────────────────────
n_kept    = keep_mask.sum()
n_dropped = (~keep_mask).sum()
print(f"\n  Kept:    {n_kept:>8,} sequences ({100*n_kept/len(df):.1f}%)")
print(f"  Dropped: {n_dropped:>8,} sequences ({100*n_dropped/len(df):.1f}%)")

print("\n  Per-isotype summary (dropped variants):")
dropped = summary_df[~summary_df["kept"]]
if len(dropped):
    print(dropped[["isotype","struct_variant","n_seqs","top_orders_frac","top_orders"]]
          .to_string(index=False))
else:
    print("  (none dropped)")

# ── 8. Save ───────────────────────────────────────────────────────────────────
out = df[keep_mask].drop(columns=["d_ins_count", "var_arm_length", "struct_variant"])
out.to_csv(args.output, index=False)
print(f"\nSaved filtered CSV → {args.output}  ({len(out):,} rows)", flush=True)

summary_path = args.output.replace(".csv", "_variant_summary.csv")
summary_df.to_csv(summary_path, index=False)
print(f"Saved variant summary → {summary_path}", flush=True)
