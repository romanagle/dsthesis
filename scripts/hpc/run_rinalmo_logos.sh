#!/usr/bin/env bash
DATA_ROOT="${TRNA_DATA_ROOT:-/data/roma/dsthesis}"
# run_rinalmo_logos.sh
# --------------------
# Run all model-based identity logo scripts after compute_order_saliency_rinalmo.py
# finishes.  Execute from /data/roma:
#   bash scripts/run_rinalmo_logos.sh 2>&1 | tee logs/run_rinalmo_logos.log

set -euo pipefail
cd "$(dirname "$0")/.."

SAL="order_identity_embeddings/order_saliency_rinalmo.npz"

if [ ! -f "$SAL" ]; then
    echo "ERROR: $SAL not found — run compute_order_saliency_rinalmo.py first." >&2
    exit 1
fi

echo "=== 1. Global per-isotype saliency logos ==="
python scripts/figures/plot_global_saliency_logos_rinalmo.py

echo ""
echo "=== 2. GMM cluster model-based logos ==="

echo ""
echo "All done."
