#!/bin/bash
#SBATCH --job-name=cmalign_probe
#SBATCH --output=logs/cmalign_probe_%j.out
#SBATCH --error=logs/cmalign_probe_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00

DATA_ROOT="${TRNA_DATA_ROOT:-/data/roma/dsthesis}"
# Scores all dedup tRNA sequences against isotype-specific CMs using cmalign.
# Produces a dense (n_seqs x n_isotypes) bit-score matrix per domain for
# logistic probing — analogous to neural LM embedding probing.
# cmalign is CPU-only; GPU node used here for access to 32 cores.

set -euo pipefail
mkdir -p $DATA_ROOT/logs

echo "Host:    $(hostname)"
echo "CPUs:    $SLURM_CPUS_PER_TASK"
echo "Started: $(date)"

conda run -n myenv python $DATA_ROOT/exploratory/run_cmscan.py \
    --csv     $DATA_ROOT/data/dedup/trnascan_GTDB_r226_dedup_taxonomy.csv \
    --cm-dir  /data/conner/software/tRNAscan-SE/lib/models \
    --outdir  $DATA_ROOT/results/cmscan \
    --cpus    $SLURM_CPUS_PER_TASK

echo "Finished: $(date)"
