#!/bin/bash
DATA_ROOT="${TRNA_DATA_ROOT:-/data/roma/dsthesis}"
# submit_pipeline.sh
# ------------------
# Submits the three-stage dedup accuracy pipeline with SLURM dependencies.
#
#   Stage 1 (GPU):  embed all unique sequences          -- ~4-12h depending on hardware
#   Stage 2 (CPU):  classify each dedup level (array)   -- ~1-4h per level
#   Stage 3 (CPU):  plot accuracy vs dedup level        -- <1 min
#
# Usage:
#   bash scripts/sbatch/submit_pipeline.sh
#
#   Skip stage 1 if embeddings already exist:
#   bash scripts/sbatch/submit_pipeline.sh --skip-embed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_EMBED=0

for arg in "$@"; do
    [[ "$arg" == "--skip-embed" ]] && SKIP_EMBED=1
done

# ── Stage 1: embed ─────────────────────────────────────────────────────────────
if [[ $SKIP_EMBED -eq 0 ]]; then
    JOB1=$(sbatch --parsable "$SCRIPT_DIR/01_embed_full.sbatch")
    echo "Submitted embed job:    $JOB1"
    EMBED_DEP="--dependency=afterok:$JOB1"
else
    echo "Skipping embed (--skip-embed set); assuming embeddings exist."
    EMBED_DEP=""
fi

# ── Stage 2: classify (job array, 8 tasks) ────────────────────────────────────
JOB2=$(sbatch --parsable $EMBED_DEP "$SCRIPT_DIR/02_classify_array.sbatch")
echo "Submitted classify array: $JOB2"

# ── Stage 3: plot ─────────────────────────────────────────────────────────────
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 "$SCRIPT_DIR/03_plot.sbatch")
echo "Submitted plot job:       $JOB3"

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f $DATA_ROOT/logs/embed_full_${JOB1:-<id>}.out"
echo ""
echo "Output:"
echo "  Embeddings:  $DATA_ROOT/embeddings/full/"
echo "  Results:     $DATA_ROOT/dedup_accuracy/result_<level>.json"
echo "  Figure:      $DATA_ROOT/figures/dedup_accuracy.png"
