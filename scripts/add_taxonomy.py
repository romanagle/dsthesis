"""
add_taxonomy.py
---------------
Joins trnascan_GTDB_r226.csv with GTDB r226 taxonomy, adding one column
per taxonomic rank (domain → species) to the full ~6M-row table.

Output: /data/roma/trnascan_GTDB_r226_taxonomy.csv
"""

import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(description="Add GTDB taxonomy columns to a trnascan CSV")
parser.add_argument("csv", nargs="?", default="/data/roma/trnascan_GTDB_r226.csv",
                    help="Input trnascan CSV (default: trnascan_GTDB_r226.csv)")
args = parser.parse_args()

TRNASCAN  = args.csv
TAXONOMY  = "/data/kate/software/release226/taxonomy/gtdb_taxonomy.tsv"
_p = Path(TRNASCAN)
OUTPUT    = str(_p.parent / (_p.stem + "_taxonomy" + _p.suffix))

TAX_RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
TAX_PREFIXES = ["d__", "p__", "c__", "o__", "f__", "g__", "s__"]


# ── 1. Load and parse taxonomy ────────────────────────────────────────────────
print("Loading taxonomy …")
tax = pd.read_csv(TAXONOMY, sep="\t", header=None, names=["raw_id", "taxonomy"])

# Strip RS_ / GB_ prefix to get bare accession (e.g. GCF_000016525.1)
tax["accession"] = tax["raw_id"].str.replace(r"^(RS_|GB_)", "", regex=True)

# Split semicolon-delimited string into rank columns, strip prefixes
split = tax["taxonomy"].str.split(";", expand=True)
for i, (rank, prefix) in enumerate(zip(TAX_RANKS, TAX_PREFIXES)):
    tax[rank] = split[i].str.replace(prefix, "", regex=False)

tax = tax[["accession"] + TAX_RANKS]
print(f"  {len(tax):,} genomes in taxonomy table")


# ── 2. Load trnascan ──────────────────────────────────────────────────────────
print("Loading trnascan CSV …")
df = pd.read_csv(TRNASCAN, low_memory=False, on_bad_lines="skip")
df = df[df["tRNAscanID"] != "tRNAscanID"].copy()   # drop repeated headers
print(f"  {len(df):,} tRNA rows, {df['GenomeID'].nunique():,} unique genomes")


# ── 3. Normalize GenomeID → accession ────────────────────────────────────────
# e.g. GCF_014170335.1_genomic → GCF_014170335.1
df["accession"] = df["GenomeID"].str.replace(r"_genomic$", "", regex=True)


# ── 4. Merge ──────────────────────────────────────────────────────────────────
print("Merging …")
df = df.merge(tax, on="accession", how="left")

matched = df["domain"].notna().sum()
print(f"  {matched:,} / {len(df):,} rows matched ({matched/len(df)*100:.2f}%)")

unmatched_genomes = df.loc[df["domain"].isna(), "GenomeID"].nunique()
if unmatched_genomes:
    print(f"  WARNING: {unmatched_genomes:,} genomes had no taxonomy match")


# ── 5. Clean up and save ──────────────────────────────────────────────────────
df = df.drop(columns=["accession"])

print(f"Saving to {OUTPUT} …")
df.to_csv(OUTPUT, index=False)
print("Done.")
print(f"  Output columns: {list(df.columns[:5])} … {TAX_RANKS}")
