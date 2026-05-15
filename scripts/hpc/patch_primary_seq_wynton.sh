#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
#$ -N patch_primary_seq
#$ -l h_rt=4:00:00
#$ -l mem_free=32G
#$ -pe smp 1

DATA_ROOT="${TRNA_DATA_ROOT:-/data/roma/dsthesis}"
# Usage: qsub patch_primary_seq_wynton.sh <struct_dir_or_file> [input.csv] [output.csv]
#   $1  directory containing *.struct.out files, OR a single struct.out file
#   $2  input CSV  (default: ~/Roma/trnascan_GTDB_r226_v2_final.csv)
#   $3  output CSV (default: ~/Roma/trnascan_GTDB_r226_v2.csv)

set -euo pipefail

TRNASCAN_DIR="/wynton/scratch/c-langeberg/trnascan"
STRUCT_INPUT="${1:-${TRNASCAN_DIR}}"
CSV_IN="${2:-${HOME}/Roma/trnascan_GTDB_r226_v2_final.csv}"
CSV_OUT="${3:-${TRNASCAN_DIR}/trnascan_GTDB_r226_v2.csv}"

echo "Date:         $(date)"
echo "Host:         $(hostname)"
echo "STRUCT_INPUT: ${STRUCT_INPUT}"
echo "CSV_IN:       ${CSV_IN}"
echo "CSV_OUT:      ${CSV_OUT}"

[[ -f "$CSV_IN" ]] || { echo "ERROR: CSV not found: $CSV_IN" >&2; exit 1; }

python3 - "$STRUCT_INPUT" "$CSV_IN" "$CSV_OUT" <<'PYEOF'
import csv, os, sys

struct_input, csv_in, csv_out = sys.argv[1], sys.argv[2], sys.argv[3]

# --- Collect struct.out paths ---
if os.path.isdir(struct_input):
    struct_files = [
        os.path.join(struct_input, f)
        for f in os.listdir(struct_input)
        if f.endswith(".struct.out")
    ]
    print(f"Found {len(struct_files):,} struct.out files in {struct_input}", flush=True)
else:
    struct_files = [struct_input]
    print(f"Using single struct file: {struct_input}", flush=True)

# --- Parse struct.out files: grab tRNAscanID and raw Seq: for each entry ---
# Format (from tRNAscan-SE -f / --struct):
#   CELF22B7.trna1 (12619-12738)   Length: 120 bp   <- ID = first token
#   ...
#   Seq: GCACGGATGGCCGAGTGGTctAAGG...               <- raw sequence, case preserved
#   Str: >>>>>>>..>>>...                             <- dot-bracket structure

seqs = {}
for path in struct_files:
    current_id = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if "Length:" in line and line.split():
                current_id = line.split()[0]
            elif line.startswith("Seq:") and current_id:
                seqs[current_id] = line.split("Seq:", 1)[1].strip()
                current_id = None

print(f"Sequences parsed: {len(seqs):,}", flush=True)

# --- Stream CSV and replace primary_sequence ---
# Write to a temp file first, then atomically replace the target.
tmp_out = csv_out + ".tmp"
matched = total = 0
with (
    open(csv_in,  newline="", encoding="utf-8") as fin,
    open(tmp_out, "w", newline="", encoding="utf-8") as fout,
):
    reader = csv.reader(fin)
    writer = csv.writer(fout)

    header = next(reader)
    id_col  = header.index("tRNAscanID")
    seq_col = header.index("primary_sequence")
    writer.writerow(header)

    for row in reader:
        total += 1
        seq = seqs.get(row[id_col])
        if seq is not None:
            row[seq_col] = seq
            matched += 1
        writer.writerow(row)
        if total % 500_000 == 0:
            print(f"  {total:,} rows processed...", flush=True)

os.replace(tmp_out, csv_out)
print(f"Total rows : {total:,}")
print(f"Matched    : {matched:,}")
print(f"Output     : {csv_out}")
PYEOF

echo "Done: $(date)"
[[ -n "${JOB_ID:-}" ]] && qstat -j "$JOB_ID"
