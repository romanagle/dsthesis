#!/bin/bash
# run_film_pipeline.sh
# Full pipeline: rebuild training dataset → train FiLM GNN → ISM logos → comparison figure
# Outputs go to gnn_film/ and figures/ism_logos_film/ to avoid overwriting gnn_final/
set -euo pipefail

cd /data/roma/dsthesis
export TRNA_DATA_ROOT=/data/roma/dsthesis

LOG=logs/film_pipeline_$(date +%Y%m%d_%H%M%S).log
mkdir -p logs figures/ism_logos_film figures/film_comparison
exec > >(tee -a "$LOG") 2>&1

# Email on exit (success or failure) — requires NOTIFY_FROM + NOTIFY_PASSWORD env vars
_notify_exit() {
    code=$?
    if [ $code -eq 0 ]; then
        python scripts/notify.py "FiLM pipeline done" \
            "run_film_pipeline.sh completed successfully. Log: $LOG" || true
    else
        python scripts/notify.py "FiLM pipeline FAILED (exit $code)" \
            "run_film_pipeline.sh failed with exit code $code. Log: $LOG" || true
    fi
}
trap _notify_exit EXIT

echo "================================================================"
echo " FiLM pipeline  $(date)"
echo "================================================================"

# ── Step 1: Rebuild training dataset ─────────────────────────────────────────
echo ""
echo "── Step 1: build_training_dataset.py ──────────────────────────"
python scripts/pipeline/build_training_dataset.py \
    --csv  data/trnascan_GTDB_r226_dedup_taxonomy.csv \
    --sprinzl data/sprinzl_filtered.csv \
    --output  data/training_dataset.csv \
    --summary data/training_dataset_variant_summary.csv
echo "   Done: data/training_dataset.csv"

# ── Step 2: Train FiLM GNN + compute ISM for all isotypes ────────────────────
echo ""
echo "── Step 2: train_gnn.py --film ─────────────────────────────────"
python scripts/train/train_gnn.py \
    --emb-dir        embeddings/gnn \
    --csv            data/trnascan_GTDB_r226_dedup_taxonomy.csv \
    --outdir         gnn_film \
    --film \
    --phylo-encoder  gnn_final/phylo_encoder.pkl \
    --film-epochs    10 \
    --film-lr        1e-3 \
    --ism-n-seqs     100 \
    --ism-orders     Pelagibacterales
echo "   Done: gnn_film/"

# ── Step 3: Plot ISM logos for FiLM model ────────────────────────────────────
echo ""
echo "── Step 3: plot_ism_logos.py (FiLM) ───────────────────────────"
python scripts/figures/plot_ism_logos.py \
    --emb-dir  embeddings/gnn \
    --ism-dir  gnn_film \
    --csv      data/trnascan_GTDB_r226_dedup_taxonomy.csv \
    --outdir   figures/ism_logos_film
echo "   Done: figures/ism_logos_film/"

# ── Step 4: Comparison figure ─────────────────────────────────────────────────
echo ""
echo "── Step 4: compare_film_vs_base.py ────────────────────────────"
python scripts/figures/compare_film_vs_base.py \
    --base-dir gnn_final \
    --film-dir gnn_film \
    --outdir   figures/film_comparison
echo "   Done: figures/film_comparison/"

# ── Step 5: FiLM-delta ISM (per order) ───────────────────────────────────────
echo ""
echo "── Step 5: compute_film_delta_ism.py ──────────────────────────"
python scripts/ism/compute_film_delta_ism.py \
    --isotypes His Ala Gly \
    --min-seqs 5 \
    --n-seqs   20 \
    --model    gnn_film/gnn_model.pt \
    --outdir   gnn_film/film_delta_ism
echo "   Done: gnn_film/film_delta_ism/"

# ── Step 6: Divergence figures ────────────────────────────────────────────────
echo ""
echo "── Step 6: plot_film_divergence.py ────────────────────────────"
python scripts/figures/plot_film_divergence.py \
    --isotypes His Ala Gly \
    --outdir   figures/film_divergence
echo "   Done: figures/film_divergence/"

echo ""
echo "================================================================"
echo " Pipeline complete  $(date)"
echo "================================================================"
echo " Key outputs:"
echo "   gnn_film/gnn_model.pt              FiLM model checkpoint"
echo "   gnn_film/gnn_results.json          accuracy + IE scores"
echo "   gnn_film/gnn_ism_saliency_*.npy    ISM maps (42 files)"
echo "   gnn_film/film_delta_ism/           per-order FiLM-delta ISM"
echo "   figures/ism_logos_film/            per-isotype logos"
echo "   figures/film_comparison/           baseline vs FiLM comparison"
echo "   figures/film_divergence/           cross-order divergence maps"
