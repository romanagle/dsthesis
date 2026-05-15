#!/bin/bash
#SBATCH --job-name=rinalmo_embed
#SBATCH --output=logs/embed_full_%j.out
#SBATCH --error=logs/embed_full_%j.err
#SBATCH --partition=gpu               # <-- adjust to your cluster's GPU partition
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

DATA_ROOT="${TRNA_DATA_ROOT:-/data/roma/dsthesis}"
# Computes RiNALMo mean-pool embeddings for every unique sequence
# in trnascan_GTDB_r226_v2_final_matched_taxonomy.csv.
# Output reused by all classifier jobs -- only needs to run once.

set -euo pipefail
mkdir -p $DATA_ROOT/logs
mkdir -p $DATA_ROOT/results/embeddings/full

# ── Load modules (adjust for your cluster) ────────────────────────────────────
# module load cuda/12.1
# module load python/3.10
# source /path/to/venv/bin/activate

echo "Host:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Started: $(date)"

python $DATA_ROOT/scripts/embeddings/extract_rinalmo_full.py \
    --input      $DATA_ROOT/trnascan_GTDB_r226_v2_final_matched_taxonomy.csv \
    --outdir     $DATA_ROOT/results/embeddings/full \
    --batch-size 512 \
    --gpu        0 \
    --layer      12 \
    --resume

echo "Finished: $(date)"
