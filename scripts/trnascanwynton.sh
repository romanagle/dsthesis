#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
#$ -N trnascan
#$ -l h_rt=24:00:00
#$ -l mem_free=4G
#$ -l scratch=5G
#$ -pe smp 1
#$ -t 1-150000
#$ -tc 50

set -euo pipefail

# ---- Reset inherited environments (avoid colabfold/conda PATH contamination) ----
export PATH="/usr/bin:/bin:/usr/sbin:/sbin"

# ---- Paths (UNCHANGED ROOT) ----
FASTA_GLOB="/wynton/group/doudna/GTDB/gtdb_genomes_reps_r226/database/**/**/**/**/*.fna.gz"

BASE="/wynton/home/doudna/c-langeberg/Roma"
OUT_DIR="${BASE}/results/runs"
LOG_DIR="${BASE}/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# ---- Environment (only what we need) ----
export PATH="/wynton/home/${USER}/.local/bin:${PATH}"
export PERL5LIB="/global/home/users/${USER}/.local/lib/tRNAscan-SE:${PERL5LIB:-}"

TASK_ID="${SGE_TASK_ID:?Run as an array job}"

# ---- Build deterministic list of genome FASTAs ----
shopt -s globstar nullglob
FILES=( $FASTA_GLOB )
shopt -u globstar

N="${#FILES[@]}"
if (( N == 0 )); then
  echo "No FASTA files found at: $FASTA_GLOB" >&2
  exit 1
fi

if (( TASK_ID < 1 || TASK_ID > N )); then
  echo "Task ${TASK_ID} out of range (N=${N}). Exiting."
  exit 0
fi

FASTA_GZ="${FILES[$((TASK_ID - 1))]}"

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
echo "TASK_ID: ${TASK_ID} / ${N}"
echo "Genome: ${FASTA_GZ}"
echo "Local FASTA: ${FASTA_LOCAL}"
echo "Perl in job: $(which perl)"
perl -v | head -n 2

# ---- Run tRNAscan-SE ----
tRNAscan-SE -q -B "$FASTA_LOCAL" \
  -o "$OUT_MAIN" \
  -s "$OUT_ISO" \
  -m "$OUT_STATS" \
  -f "$OUT_STRUCT" \
  --detail

[[ -n "${JOB_ID:-}" ]] && qstat -j "$JOB_ID"