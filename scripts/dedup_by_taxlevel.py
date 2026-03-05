"""
dedup_by_taxlevel.py
---------------------
Deduplicates tRNA sequences at each taxonomic level independently.

Logic:
  - For each level (domain → species), keep one representative row per
    unique (primary_sequence, taxon) pair.
  - A sequence shared across two phyla is kept once per phylum.
  - A sequence that appears 100 times within the same genus keeps only one.

Input:  trnascan_GTDB_r226_taxonomy.csv  (2.8 GB, read in chunks)
Output: results/tax_level_dedup/dedup_{level}.csv  (one file per level)

Usage:
    python scripts/dedup_by_taxlevel.py
    python scripts/dedup_by_taxlevel.py --input /path/to/file.csv --outdir /path/to/out
"""

import argparse
import csv
import sys
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument(
    "--input", default="/data/roma/trnascan_GTDB_r226_taxonomy.csv",
    help="Path to taxonomy-enriched tRNAscan CSV"
)
parser.add_argument(
    "--outdir", default="/data/roma/results/tax_level_dedup",
    help="Output directory for per-level deduplicated CSVs"
)
parser.add_argument(
    "--chunksize", type=int, default=500_000,
    help="Rows to read per chunk (default 500000)"
)
args = parser.parse_args()

LEVELS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
SEQ_COL = "primary_sequence"
CHUNK = args.chunksize
INPUT = Path(args.input)
OUTDIR = Path(args.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"Input:   {INPUT}")
print(f"Output:  {OUTDIR}")
print(f"Levels:  {', '.join(LEVELS)}\n")

# ── open one output file per level ───────────────────────────────────────────
# We'll write headers after reading the first chunk.
out_files   = {}   # level -> file handle
out_writers = {}   # level -> csv.DictWriter
seen        = {level: set() for level in LEVELS}   # (seq, taxon) already written

# ── stream through the input CSV in chunks ───────────────────────────────────
total_in = 0
kept     = {level: 0 for level in LEVELS}

with open(INPUT, newline="", encoding="utf-8") as fh:
    reader = csv.DictReader(fh)
    fieldnames = reader.fieldnames

    # validate columns
    missing = [c for c in LEVELS + [SEQ_COL] if c not in fieldnames]
    if missing:
        sys.exit(f"ERROR: columns not found in CSV: {missing}")

    # open output writers now that we have fieldnames
    for level in LEVELS:
        out_path = OUTDIR / f"dedup_{level}.csv"
        f = open(out_path, "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        out_files[level]   = f
        out_writers[level] = writer

    chunk = []
    for row in reader:
        # skip repeated header rows (artifact of concatenated CSVs)
        if row["tRNAscanID"] == "tRNAscanID":
            continue

        chunk.append(row)
        total_in += 1

        if len(chunk) >= CHUNK:
            for row in chunk:
                seq = row[SEQ_COL]
                for level in LEVELS:
                    taxon = row.get(level, "")
                    key = (seq, taxon)
                    if key not in seen[level]:
                        seen[level].add(key)
                        out_writers[level].writerow(row)
                        kept[level] += 1
            chunk = []

            print(f"  processed {total_in:>12,} rows  |  "
                  + "  ".join(f"{l[0]}:{kept[l]:,}" for l in LEVELS),
                  flush=True)

    # flush remaining chunk
    for row in chunk:
        seq = row[SEQ_COL]
        for level in LEVELS:
            taxon = row.get(level, "")
            key = (seq, taxon)
            if key not in seen[level]:
                seen[level].add(key)
                out_writers[level].writerow(row)
                kept[level] += 1

# ── close all output files ────────────────────────────────────────────────────
for f in out_files.values():
    f.close()

# ── summary ──────────────────────────────────────────────────────────────────
print(f"\n{'─'*65}")
print(f"  Total input rows:  {total_in:,}")
print(f"{'─'*65}")
print(f"  {'Level':<10}  {'Kept':>12}  {'Removed':>12}  {'% kept':>8}")
print(f"  {'─'*10}  {'─'*12}  {'─'*12}  {'─'*8}")
for level in LEVELS:
    k = kept[level]
    r = total_in - k
    pct = k / total_in * 100 if total_in else 0
    print(f"  {level:<10}  {k:>12,}  {r:>12,}  {pct:>7.1f}%")
print(f"{'─'*65}")
print(f"\nFiles written to: {OUTDIR}/")
