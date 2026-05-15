#!/bin/bash
# run_raw_film_prep.sh
# Trains Raw GCN +SS +FiLM and evaluates Emb GCN -SS +FiLM per-domain accuracy
# in parallel on GPUs 0 and 2.  GPU 1 is left alone.
set -euo pipefail

cd /data/roma/dsthesis
export TRNA_DATA_ROOT=/data/roma/dsthesis

LOG_DIR=logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)

RAW_LOG="$LOG_DIR/raw_film_${STAMP}.log"
EVAL_LOG="$LOG_DIR/eval_film_per_domain_${STAMP}.log"

echo "================================================================"
echo " Raw FiLM prep  $(date)"
echo " GPU 0 → train_gnn_raw_film.py   ($RAW_LOG)"
echo " GPU 2 → eval_film_per_domain.py ($EVAL_LOG)"
echo "================================================================"

# ── Job 1 (GPU 0): Raw GCN +SS +FiLM ─────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 conda run -n myenv python scripts/train/train_gnn_raw_film.py \
    --npz             embeddings/gnn/perpos_embeddings.npz \
    --csv             data/trnascan_GTDB_r226_dedup_taxonomy.csv \
    --outdir          gnn_raw_film \
    --film \
    --phylo-encoder   gnn_final/phylo_encoder.pkl \
    --film-epochs     10 \
    --film-lr         1e-3 \
    --ism-isotypes    Ala Gly \
    --ism-n-seqs      100 \
    > "$RAW_LOG" 2>&1 &
PID_RAW=$!
echo "  Raw model PID: $PID_RAW"

# ── Job 2 (GPU 2): per-domain eval of existing Emb GCN -SS +FiLM ─────────────
CUDA_VISIBLE_DEVICES=2 conda run -n myenv python scripts/eval_film_per_domain.py \
    --results-dir   gnn_film \
    --npz           embeddings/gnn/perpos_embeddings.npz \
    --phylo-encoder gnn_final/phylo_encoder.pkl \
    > "$EVAL_LOG" 2>&1 &
PID_EVAL=$!
echo "  Eval model PID: $PID_EVAL"

echo ""
echo "Both jobs running.  Waiting …"

# Wait for both and capture exit codes
FAIL=0
wait "$PID_RAW"  || { echo "ERROR: raw film job failed (exit $?)"; FAIL=1; }
wait "$PID_EVAL" || { echo "ERROR: eval job failed (exit $?)";     FAIL=1; }

if [ "$FAIL" -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo " Both jobs complete  $(date)"
    echo " Raw results : gnn_raw_film/gnn_results.json"
    echo " FiLM eval   : gnn_film/gnn_results.json (patched)"
    echo "================================================================"
    python scripts/notify.py "Raw FiLM prep done" \
        "run_raw_film_prep.sh finished. Logs: $RAW_LOG  $EVAL_LOG" || true
else
    echo ""
    echo "================================================================"
    echo " One or more jobs FAILED  $(date)"
    echo " Check logs: $RAW_LOG  $EVAL_LOG"
    echo "================================================================"
    python scripts/notify.py "Raw FiLM prep FAILED" \
        "run_raw_film_prep.sh had failures. Check: $RAW_LOG  $EVAL_LOG" || true
    exit 1
fi
