#!/usr/bin/env python3
"""
build_training_dataset.py
--------------------------
Build the cleaned training dataset in five steps:

  1. Start with all tRNAscan sequences with valid Sprinzl mappings (~4.7M)
  2. Filter by confidence score: drop sequences below mean − N SD,
     computed separately for Bacteria and Archaea
  3. Deduplicate at family level: keep the highest conf_score copy per
     (primary_sequence, family)
  4. For each isotype × domain, drop the N least frequent structural variants
     (sprinzl_type + D-loop insertions + variable arm length)

After each step a table of tRNA counts and unique genome counts is printed,
broken down by domain.

Outputs
-------
  --output   data/training_dataset.csv
  --summary  data/training_dataset_variant_summary.csv

Usage
-----
    python scripts/build_training_dataset.py
    python scripts/build_training_dataset.py --drop-n-variants 10 --conf-sd 2.0
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv",             default=f"{DATA_ROOT}/data/trnascan_GTDB_r226_dedup_taxonomy.csv",
                    help="Main CSV with sequences + taxonomy (trnascan_GTDB_r226_dedup_taxonomy.csv)")
parser.add_argument("--sprinzl",         default=f"{DATA_ROOT}/data/sprinzl_filtered.csv",
                    help="CSV with sprinzl_ok/sprinzl_type/pos_* columns (sprinzl_filtered.csv)")
parser.add_argument("--output",          default="data/training_dataset.csv")
parser.add_argument("--summary",         default="data/training_dataset_variant_summary.csv")
parser.add_argument("--drop-n-variants", type=int,   default=10,
                    help="Drop the N least frequent structural variants per isotype × domain (default: 10)")
parser.add_argument("--conf-sd",         type=float, default=2.0,
                    help="Drop sequences below mean − N SD of conf_score per domain (default: 2.0)")
args = parser.parse_args()

os.makedirs(os.path.dirname(args.output), exist_ok=True)

D_INSERT   = ["17a", "20a", "20b"]
VAR_INSERT = ["47a","47b","47c","47d","47e","47f","47g",
              "47h","47i","47j","47k","47l","47m","47n","47o","47p"]

DOMAIN_ORDER = ["Bacteria", "Archaea"]


# ── Reporting helper ──────────────────────────────────────────────────────────
def print_stats(df: pd.DataFrame, label: str) -> None:
    """Print tRNA count and unique genome count per domain, plus total."""
    W = 56
    print(f"\n  ── {label} {'─' * max(0, W - len(label))}")
    has_genome = "GenomeID" in df.columns
    has_domain = "domain"   in df.columns
    col_w = 12

    hdr = f"    {'Domain':<{col_w}}  {'tRNAs':>10}  {'Genomes':>9}"
    print(hdr)
    print("    " + "─" * (len(hdr) - 4))

    if has_domain:
        for dom in DOMAIN_ORDER:
            sub = df[df["domain"] == dom]
            if sub.empty:
                continue
            g = sub["GenomeID"].nunique() if has_genome else "—"
            g_str = f"{g:>9,}" if isinstance(g, int) else f"{'—':>9}"
            print(f"    {dom:<{col_w}}  {len(sub):>10,}  {g_str}")

    g_total = df["GenomeID"].nunique() if has_genome else "—"
    g_str = f"{g_total:>9,}" if isinstance(g_total, int) else f"{'—':>9}"
    print(f"    {'TOTAL':<{col_w}}  {len(df):>10,}  {g_str}", flush=True)


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("Step 1: Loading data")
print("=" * 60)

print(f"  Loading main CSV: {args.csv} ...", flush=True)
base_cols = (
    ["tRNAscanID", "GenomeID", "primary_sequence", "secondary_structure",
     "anticodon_start", "Anticodon_predicted_isotype",
     "domain", "phylum", "class", "order", "family", "conf_score"]
)
df = pd.read_csv(args.csv, usecols=base_cols, low_memory=False)
df["conf_score"] = pd.to_numeric(df["conf_score"], errors="coerce")
print(f"  Main CSV rows: {len(df):,}", flush=True)

print(f"  Loading sprinzl columns: {args.sprinzl} ...", flush=True)
sprinzl_join_cols = (
    ["tRNAscanID", "sprinzl_ok", "sprinzl_type"]
    + [f"pos_{p}" for p in D_INSERT + VAR_INSERT]
)
sprinzl_df = pd.read_csv(args.sprinzl,
                          usecols=[c for c in sprinzl_join_cols
                                   if c in pd.read_csv(args.sprinzl, nrows=0).columns],
                          low_memory=False)
df = df.merge(sprinzl_df, on="tRNAscanID", how="inner")
print(f"  Rows after sprinzl join: {len(df):,}", flush=True)

# Filter: valid Sprinzl mapping and valid isotype
df = df[df["sprinzl_ok"].astype(float) == 1].copy()
df = df[~df["Anticodon_predicted_isotype"].isin(["Undet", "Sup", "SeC", ""])].copy()
df = df.dropna(subset=["Anticodon_predicted_isotype", "family", "domain", "conf_score"]).copy()

# Merge Ile2 → Ile
df["Anticodon_predicted_isotype"] = df["Anticodon_predicted_isotype"].replace("Ile2", "Ile")
print(f"  Ile2 merged into Ile", flush=True)
print_stats(df, "After sprinzl_ok + isotype filter + taxonomy")

# ── 2. Confidence score filter (per domain) ───────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 2: Confidence score filter (−{args.conf_sd} SD floor per domain)")
print("=" * 60)

keep_conf = pd.Series(False, index=df.index)
for domain, grp in df.groupby("domain"):
    mu  = grp["conf_score"].mean()
    sd  = grp["conf_score"].std()
    lo  = mu - args.conf_sd * sd
    mask = grp["conf_score"] >= lo
    keep_conf.loc[grp.index[mask]] = True
    print(f"  {domain}: mean={mu:.1f}  sd={sd:.1f}  "
          f"floor={lo:.1f}  "
          f"kept={mask.sum():,}  dropped={(~mask).sum():,}", flush=True)

df = df[keep_conf].copy()
print_stats(df, "After confidence score filter")

# ── 3. Family-level deduplication (keep highest conf_score) ──────────────────
print("\n" + "=" * 60)
print("Step 3: Family-level deduplication (highest conf_score per duplicate)")
print("=" * 60)

df_dedup = (
    df.sort_values("conf_score", ascending=False)
      .drop_duplicates(subset=["primary_sequence", "family"])
      .copy()
)

# Restore one best-confidence tRNA per genome that lost all sequences
lost = df[~df["GenomeID"].isin(df_dedup["GenomeID"])]
if len(lost):
    rescue = (lost.sort_values("conf_score", ascending=False)
                  .drop_duplicates(subset=["GenomeID"]))
    print(f"  Restoring best tRNA for {rescue['GenomeID'].nunique():,} "
          f"fully-dropped genomes", flush=True)
    df_dedup = pd.concat([df_dedup, rescue], ignore_index=True)

df = df_dedup
print_stats(df, "After family-level deduplication")

# ── 4. Compute structural variant signature ───────────────────────────────────
print("\n" + "=" * 60)
print("Step 4: Computing structural variant signatures")
print("=" * 60)

for p in D_INSERT + VAR_INSERT:
    col = f"pos_{p}"
    df[col] = df[col].astype(str).str.strip().replace({"": "__", "nan": "__"})

d_cols   = [f"pos_{p}" for p in D_INSERT]
var_cols = [f"pos_{p}" for p in VAR_INSERT]

df["d_ins_count"]    = (df[d_cols]   != "__").sum(axis=1).astype(np.int8)
df["var_arm_length"] = (df[var_cols] != "__").sum(axis=1).astype(np.int8)
df["struct_variant"] = (
    df["sprinzl_type"].astype(str) + "_d" +
    df["d_ins_count"].astype(str)  + "_v" +
    df["var_arm_length"].astype(str)
)
print(f"  Unique structural variants: {df['struct_variant'].nunique()}", flush=True)

# ── 5. Drop N least frequent structural variants per isotype × domain ─────────
print("\n" + "=" * 60)
print(f"Step 5: Dropping {args.drop_n_variants} least frequent variants per isotype × domain")
print("=" * 60)

summary_rows = []
keep_mask = pd.Series(True, index=df.index)

for (iso, domain), grp in df.groupby(["Anticodon_predicted_isotype", "domain"]):
    variant_counts = grp["struct_variant"].value_counts()
    n_variants = len(variant_counts)

    n_to_drop     = min(args.drop_n_variants, n_variants - 1)  # always keep at least 1
    drop_variants = set(variant_counts.tail(n_to_drop).index) if n_to_drop > 0 else set()

    for variant, count in variant_counts.items():
        dropped = variant in drop_variants
        summary_rows.append({
            "isotype":         iso,
            "domain":          domain,
            "struct_variant":  variant,
            "n_seqs":          count,
            "frac_of_isotype": round(count / len(grp), 6),
            "dropped":         dropped,
        })
        if dropped:
            keep_mask.loc[grp[grp["struct_variant"] == variant].index] = False

df = df[keep_mask].copy()
print_stats(df, "After structural variant filter")

# ── 6. Save ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 6: Saving")
print("=" * 60)

out = df.drop(columns=["d_ins_count", "var_arm_length", "struct_variant"])
out.to_csv(args.output, index=False)
print(f"  Final dataset: {len(out):,} sequences → {args.output}", flush=True)

summary_df = (
    pd.DataFrame(summary_rows)
    .sort_values(["isotype", "domain", "n_seqs"], ascending=[True, True, False])
)
summary_df.to_csv(args.summary, index=False)
print(f"  Variant summary  → {args.summary}", flush=True)

print("\n  Per-isotype counts (final):")
iso_counts = out["Anticodon_predicted_isotype"].value_counts().sort_index()
for iso, n in iso_counts.items():
    print(f"    {iso:<8} {n:>8,}")
print(f"  {'TOTAL':<8} {len(out):>8,}")

print("\n  Domain breakdown:")
for domain, n in out["domain"].value_counts().items():
    print(f"    {domain:<12} {n:>8,}")
