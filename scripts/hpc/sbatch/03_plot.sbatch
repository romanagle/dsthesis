#!/bin/bash
#SBATCH --job-name=dedup_plot
#SBATCH --output=logs/plot_%j.out
#SBATCH --error=logs/plot_%j.err
#SBATCH --partition=cpu               # <-- adjust to your cluster's CPU partition
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

DATA_ROOT="${TRNA_DATA_ROOT:-/data/roma/dsthesis}"
set -euo pipefail
mkdir -p $DATA_ROOT/logs
mkdir -p $DATA_ROOT/results/figures

# ── Load modules (adjust for your cluster) ────────────────────────────────────
# module load python/3.10
# source /path/to/venv/bin/activate

echo "Started: $(date)"

python $DATA_ROOT/scripts/figures/plot_dedup_accuracy.py \
    --indir  $DATA_ROOT/results/dedup_accuracy \
    --outdir $DATA_ROOT/results/figures

echo "Finished: $(date)"
