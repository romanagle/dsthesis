#!/usr/bin/env bash
# dedup_mmseqs2.sh
# ─────────────────────────────────────────────────────────────────────────────
# Deduplicates tRNA sequences at 100% identity using MMseqs2 linclust,
# then filters trnascan_GTDB_r226.csv (or the taxonomy-enriched version)
# to keep one representative row per unique sequence.
#
# Output: /data/roma/trnascan_GTDB_r226_dedup.csv  (or _taxonomy_dedup.csv)
#
# Usage:  bash dedup_mmseqs2.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── config ────────────────────────────────────────────────────────────────────
CSV="/data/roma/trnascan_GTDB_r226.csv"          # change to _taxonomy.csv if ready
WORKDIR="/data/roma/mmseqs2_dedup"
FASTA="$WORKDIR/sequences.fasta"
DB="$WORKDIR/seqDB"
CLUST="$WORKDIR/clustDB"
REPR="$WORKDIR/reprDB"
REPR_FASTA="$WORKDIR/representatives.fasta"
REPR_IDS="$WORKDIR/representative_ids.txt"
TMPDIR="$WORKDIR/tmp"
THREADS=16

mkdir -p "$WORKDIR" "$TMPDIR"

# ── Step 1: extract sequences → FASTA ────────────────────────────────────────
echo "[1/5] Extracting sequences from CSV → FASTA …"
python3 - <<'PYEOF'
import csv, sys

csv_path  = "/data/roma/trnascan_GTDB_r226.csv"
fasta_out = "/data/roma/mmseqs2_dedup/sequences.fasta"

seen_headers = False
written = 0
with open(csv_path) as fh, open(fasta_out, "w") as out:
    reader = csv.DictReader(fh)
    for row in reader:
        if row["tRNAscanID"] == "tRNAscanID":   # skip repeated header rows
            continue
        seq = row["primary_sequence"].strip()
        if not seq:
            continue
        # MMseqs2 needs valid FASTA IDs (no spaces)
        fasta_id = row["tRNAscanID"].replace(" ", "_")
        out.write(f">{fasta_id}\n{seq}\n")
        written += 1

print(f"  Wrote {written:,} sequences to FASTA", flush=True)
PYEOF

# ── Step 2: create MMseqs2 sequence database ──────────────────────────────────
echo "[2/5] Creating MMseqs2 sequence database …"
mmseqs createdb "$FASTA" "$DB" --dbtype 1 -v 1

# ── Step 3: cluster at 100% identity ─────────────────────────────────────────
# --min-seq-id 1.0  : 100% identity
# -c 1.0            : 100% coverage of shorter sequence
# --cov-mode 1      : coverage of shorter sequence
# --cluster-mode 2  : connected component (handles transitivity)
echo "[3/5] Clustering at 100% identity …"
mmseqs linclust "$DB" "$CLUST" "$TMPDIR" \
    --min-seq-id 1.0 \
    -c 1.0 \
    --cov-mode 1 \
    --threads "$THREADS" \
    -v 1

# ── Step 4: extract representative sequences ──────────────────────────────────
echo "[4/5] Extracting cluster representatives …"
mmseqs createsubdb "$CLUST" "$DB" "$REPR" -v 1
mmseqs convert2fasta "$REPR" "$REPR_FASTA" -v 1

# pull representative IDs (FASTA header lines, strip '>')
grep "^>" "$REPR_FASTA" | sed 's/^>//' > "$REPR_IDS"
echo "  $(wc -l < "$REPR_IDS") unique sequences (representatives)"

# ── Step 5: filter original CSV ───────────────────────────────────────────────
echo "[5/5] Filtering CSV to representative rows …"
python3 - <<'PYEOF'
import csv

repr_ids_path = "/data/roma/mmseqs2_dedup/representative_ids.txt"
csv_in_path   = "/data/roma/trnascan_GTDB_r226.csv"
csv_out_path  = "/data/roma/trnascan_GTDB_r226_dedup.csv"

with open(repr_ids_path) as fh:
    repr_ids = set(line.strip() for line in fh)

kept = skipped = dup_headers = 0
with open(csv_in_path) as fh, open(csv_out_path, "w", newline="") as out:
    reader = csv.DictReader(fh)
    writer = csv.DictWriter(out, fieldnames=reader.fieldnames)
    writer.writeheader()
    for row in reader:
        tid = row["tRNAscanID"]
        if tid == "tRNAscanID":
            dup_headers += 1
            continue
        fasta_id = tid.replace(" ", "_")
        if fasta_id in repr_ids:
            writer.writerow(row)
            kept += 1
        else:
            skipped += 1

total = kept + skipped
print(f"  Input rows:          {total:,}")
print(f"  Kept (unique seqs):  {kept:,}  ({kept/total*100:.1f}%)")
print(f"  Removed (duplicates):{skipped:,}  ({skipped/total*100:.1f}%)")
print(f"  Output: {csv_out_path}")
PYEOF

echo "Done."
