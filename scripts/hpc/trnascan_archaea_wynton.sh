#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
#$ -N trnascan_archaea
#$ -l h_rt=24:00:00
#$ -l mem_free=4G
#$ -l scratch=5G
#$ -pe smp 1
#$ -t 1-6968
#$ -tc 50

DATA_ROOT="${TRNA_DATA_ROOT:-/data/roma/dsthesis}"
set -euo pipefail

# ---- Reset inherited environments (avoid colabfold/conda PATH contamination) ----
export PATH="/usr/bin:/bin:/usr/sbin:/sbin"
unset PERL5LIB PERL_LOCAL_LIB_ROOT PERL_MB_OPT PERL_MM_OPT PERL5OPT
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL

# ---- Paths ----
GTDB_ROOT="/wynton/group/doudna/GTDB/gtdb_genomes_reps_r226/database"
PATHS_FILE="/wynton/home/doudna/c-langeberg/Roma/scripts/archaea_genome_paths.txt"

SCRATCH_BASE="/wynton/scratch/$USER/trnascan"
OUT_DIR="$SCRATCH_BASE/results"
LOG_DIR="$SCRATCH_BASE/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# ---- Environment ----
CONF="/wynton/home/doudna/c-langeberg/software/tRNAscan-SE/tRNAscan-SE.conf"
TRNASCANSE_ROOT="/wynton/home/doudna/c-langeberg/software/tRNAscan-SE"
TRNASCANSE_BIN="${TRNASCANSE_ROOT}/tRNAscan-SE"
export PERL5LIB="${TRNASCANSE_ROOT}/lib"

TASK_ID="${SGE_TASK_ID:?Run as an array job}"

# ---- Get this task's genome path from the file list ----
FASTA_GZ="$(sed -n "${TASK_ID}p" "$PATHS_FILE")"

if [[ -z "$FASTA_GZ" ]]; then
  echo "No path at line ${TASK_ID} of ${PATHS_FILE}" >&2
  exit 1
fi

if [[ ! -f "$FASTA_GZ" ]]; then
  echo "File not found: $FASTA_GZ" >&2
  exit 1
fi

# ---- Output names ----
BASENAME="$(basename "$FASTA_GZ")"
SAFE_ID="${BASENAME%.fna.gz}"
SAFE_ID="${SAFE_ID//[^A-Za-z0-9._-]/_}"

OUT_MAIN="${OUT_DIR}/${SAFE_ID}.trnascan.out"
OUT_ISO="${OUT_DIR}/${SAFE_ID}.isomodels.out"
OUT_STATS="${OUT_DIR}/${SAFE_ID}.stats.out"
OUT_STRUCT="${OUT_DIR}/${SAFE_ID}.struct.out"

if [[ -s "$OUT_MAIN" && -s "$OUT_ISO" && -s "$OUT_STATS" && -s "$OUT_STRUCT" ]]; then
  echo "Outputs exist for ${SAFE_ID}, skipping."
  exit 0
fi

# ---- Scratch ----
WORKDIR="${TMPDIR:-/tmp}"
FASTA_LOCAL="${WORKDIR}/${SAFE_ID}.fna"

# ---- Decompress genome ----
if command -v pigz >/dev/null 2>&1; then
  pigz -dc "$FASTA_GZ" > "$FASTA_LOCAL"
else
  gzip -dc "$FASTA_GZ" > "$FASTA_LOCAL"
fi

[[ -s "$FASTA_LOCAL" ]] || { echo "Empty FASTA after decompression: $FASTA_GZ" >&2; exit 1; }

echo "Date: $(date)"
echo "Host: $(hostname)"
echo "JOB_ID: ${JOB_ID:-}"
echo "TASK_ID: ${TASK_ID} / 6968"
echo "Genome: ${FASTA_GZ}"
echo "Local FASTA: ${FASTA_LOCAL}"
echo "Perl in job: $(which perl)"
perl -v | head -n 2

# ---- Run tRNAscan-SE in archaea mode ----
"$TRNASCANSE_BIN" -q -A "$FASTA_LOCAL" \
  --conf "$CONF" \
  -o "$OUT_MAIN" \
  -s "$OUT_ISO" \
  -m "$OUT_STATS" \
  -f "$OUT_STRUCT" \
  --detail

[[ -n "${JOB_ID:-}" ]] && qstat -j "$JOB_ID"
