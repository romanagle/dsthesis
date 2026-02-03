#!/bin/bash -l

#SBATCH --account=pc_rnallm
#SBATCH --job-name=trnascan
#SBATCH --partition=es1
#SBATCH --gres=gpu:A40:2
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --array=0-200%100
#SBATCH --qos=es_normal
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --output=logs/trnascan_%A_%a.out
#SBATCH --error=/global/scratch/users/romanagle/trnascan_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --mail-user=romanagle@berkeley.edu # Email notifications
#SBATCH --mail-type=ALL # Email notifications for job start, end, and failures

set -euo pipefail



BASE=/global/home/groups/pc_rnallm/roma_trnascan/dsthesis
SPLIT_DIR=/global/scratch/users/romanagle/226splits_clean
OUT_DIR=$BASE/results/runs

OUT_MAIN="$OUT_DIR/test_${SLURM_ARRAY_TASK_ID}.out"
OUT_ISO="$OUT_DIR/isomodels_${SLURM_ARRAY_TASK_ID}.out"
OUT_STATS="$OUT_DIR/stats_${SLURM_ARRAY_TASK_ID}.out"

if [[ -s "$OUT_MAIN" && -s "$OUT_ISO" && -s "$OUT_STATS" ]]; then
  echo "All outputs exist for task ${SLURM_ARRAY_TASK_ID}, skipping"
  exit 0
fi


export PERL5LIB=/global/home/users/romanagle/.local/lib/tRNAscan-SE:${PERL5LIB:-}

export PATH=/global/home/users/romanagle/.local/bin:$PATH

mkdir -p "$OUT_DIR" logs

shopt -s nullglob
files=("$SPLIT_DIR"/*)
if (( ${#files[@]} == 0 )); then
  echo "No split files found in $SPLIT_DIR" >&2
  exit 1
fi

# Sort for stable ordering
IFS=$'\n' files=($(printf '%s\n' "${files[@]}" | sort))
unset IFS

if (( SLURM_ARRAY_TASK_ID >= ${#files[@]} )); then
  echo "Task id $SLURM_ARRAY_TASK_ID out of range (nfiles=${#files[@]}). Exiting." >&2
  exit 0
fi

INPUT="${files[$SLURM_ARRAY_TASK_ID]}"

echo "Host: $(hostname)"
echo "Task: $SLURM_ARRAY_TASK_ID"
echo "Input: $INPUT"

tRNAscan-SE -B "$INPUT" \
  -o "$OUT_DIR/test_${SLURM_ARRAY_TASK_ID}.out" \
  -s "$OUT_DIR/isomodels_${SLURM_ARRAY_TASK_ID}.out" \
  -m "$OUT_DIR/stats_${SLURM_ARRAY_TASK_ID}.out" \
  --detail -f --struct --progress
