#!/bin/bash
# run_arch_rerun.sh
# Retrains all 10 architecture variants as 23-class combined-domain classifiers
# with domain-specific 5-isotype (Ala, Trp, Gly, Gln, Pro) IE scoring.
# Also supplements the two existing FiLM models with Trp/Gln/Pro ISM.
#
# GPU assignment:
#   GPU 0 : raw_shallow, raw_gnn, raw_gnn_ss, raw_deep, emb_shallow
#   GPU 2 : emb_se, emb_gnn_no_ss, emb_gnn, emb_transformer, emb_deep
#   GPU 0 : (after arch group) ISM supplement for gnn_raw_film  (fast, raw)
#   GPU 2 : (after arch group) ISM supplement for gnn_film       (slow, RiNALMo)
set -euo pipefail

cd /data/roma/dsthesis
export TRNA_DATA_ROOT=/data/roma/dsthesis

LOG_DIR=logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
OUTBASE=arch_ablation_5iso

ISM_ISOS="Ala Trp Gly Gln Pro"
ISM_N=100

echo "================================================================"
echo " Arch ablation (5-iso)  $(date)"
echo " Outputs → ${OUTBASE}/"
echo "================================================================"

# ── GPU 0 group ───────────────────────────────────────────────────────────────
run_gpu0() {
    local ARCHS=("raw_shallow" "raw_gnn" "raw_gnn_ss" "raw_deep" "emb_shallow")
    for arch in "${ARCHS[@]}"; do
        LOG="$LOG_DIR/arch_${arch}_${STAMP}.log"
        echo "  [GPU 0] starting $arch → $LOG"
        CUDA_VISIBLE_DEVICES=0 conda run -n myenv \
            python scripts/train/train_gnn_all_archs.py \
                --arch       "$arch" \
                --npz        embeddings/gnn/perpos_embeddings.npz \
                --csv        data/trnascan_GTDB_r226_dedup_taxonomy.csv \
                --outdir     "${OUTBASE}/${arch}" \
                --ism-isotypes $ISM_ISOS \
                --ism-n-seqs $ISM_N \
            > "$LOG" 2>&1 \
            || { echo "  ERROR: $arch failed (see $LOG)"; exit 1; }
        echo "  [GPU 0] finished $arch"
    done

    # Supplement raw FiLM model with Trp/Gln/Pro ISM (fast: one-hot)
    SUPP_LOG="$LOG_DIR/supp_raw_film_${STAMP}.log"
    echo "  [GPU 0] ISM supplement for gnn_raw_film → $SUPP_LOG"
    CUDA_VISIBLE_DEVICES=0 conda run -n myenv \
        python scripts/eval_film_ism_supplement.py \
            --results-dir   gnn_raw_film \
            --npz           embeddings/gnn/perpos_embeddings.npz \
            --phylo-encoder gnn_final/phylo_encoder.pkl \
            --isotypes      Trp Gln Pro \
            --raw \
        > "$SUPP_LOG" 2>&1 \
        || echo "  WARNING: raw film ISM supplement failed (see $SUPP_LOG)"
    echo "  [GPU 0] done"
}

# ── GPU 2 group ───────────────────────────────────────────────────────────────
run_gpu2() {
    local ARCHS=("emb_se" "emb_gnn_no_ss" "emb_gnn" "emb_transformer" "emb_deep")
    for arch in "${ARCHS[@]}"; do
        LOG="$LOG_DIR/arch_${arch}_${STAMP}.log"
        echo "  [GPU 2] starting $arch → $LOG"
        CUDA_VISIBLE_DEVICES=2 conda run -n myenv \
            python scripts/train/train_gnn_all_archs.py \
                --arch       "$arch" \
                --npz        embeddings/gnn/perpos_embeddings.npz \
                --csv        data/trnascan_GTDB_r226_dedup_taxonomy.csv \
                --outdir     "${OUTBASE}/${arch}" \
                --ism-isotypes $ISM_ISOS \
                --ism-n-seqs $ISM_N \
            > "$LOG" 2>&1 \
            || { echo "  ERROR: $arch failed (see $LOG)"; exit 1; }
        echo "  [GPU 2] finished $arch"
    done

    # Supplement emb FiLM model with Trp/Gln/Pro ISM (slow: RiNALMo)
    SUPP_LOG="$LOG_DIR/supp_emb_film_${STAMP}.log"
    echo "  [GPU 2] ISM supplement for gnn_film → $SUPP_LOG"
    CUDA_VISIBLE_DEVICES=2 conda run -n myenv \
        python scripts/eval_film_ism_supplement.py \
            --results-dir   gnn_film \
            --npz           embeddings/gnn/perpos_embeddings.npz \
            --phylo-encoder gnn_final/phylo_encoder.pkl \
            --isotypes      Trp Gln Pro \
        > "$SUPP_LOG" 2>&1 \
        || echo "  WARNING: emb film ISM supplement failed (see $SUPP_LOG)"
    echo "  [GPU 2] done"
}

# Launch both GPU groups in parallel
run_gpu0 &
PID0=$!
run_gpu2 &
PID2=$!

echo ""
echo "Both GPU groups running.  Waiting …"

FAIL=0
wait "$PID0" || { echo "ERROR: GPU 0 group failed"; FAIL=1; }
wait "$PID2" || { echo "ERROR: GPU 2 group failed"; FAIL=1; }

if [ "$FAIL" -eq 0 ]; then
    echo ""
    echo "================================================================"
    echo " All jobs complete  $(date)"
    echo " Arch results : ${OUTBASE}/**/arch_results.json"
    echo " FiLM patches : gnn_film/gnn_results.json"
    echo "               gnn_raw_film/gnn_results.json"
    echo "================================================================"
    python scripts/notify.py "Arch rerun done" \
        "run_arch_rerun.sh finished. Results in ${OUTBASE}/" || true
else
    echo ""
    echo "================================================================"
    echo " One or more jobs FAILED  $(date)"
    echo " Check logs in ${LOG_DIR}/"
    echo "================================================================"
    python scripts/notify.py "Arch rerun FAILED" \
        "run_arch_rerun.sh had failures. Check ${LOG_DIR}/" || true
    exit 1
fi
