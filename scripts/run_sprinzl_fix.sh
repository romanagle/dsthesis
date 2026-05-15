#!/usr/bin/env bash
# run_sprinzl_fix.sh
# ------------------
# Applies the Sprinzl-space ISM alignment fix and regenerates:
#   1. gnn_film_sprinzl/   — Sprinzl-aligned domain-level FiLM saliency
#   2. figures/ism_logos_film/
#   3. figures/film_comparison/
#   4. figures/gnn_eda_5iso/  (film comparison panels only)
#   5. figures/film_divergence/
# Then sends an ntfy notification.

set -euo pipefail
cd /data/roma

ROOT="dsthesis"
SCRIPTS="$ROOT/scripts"
FIGS="$ROOT/figures"
DATA="$ROOT/data/trnascan_GTDB_r226_dedup_taxonomy.csv"
EMB="$ROOT/embeddings/gnn"
DELTA="$ROOT/gnn_film/film_delta_ism"
SP_DIR="$ROOT/gnn_film_sprinzl"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 1: Create Sprinzl-aligned domain-level FiLM saliency ─────────────────
log "=== Step 1: create_sprinzl_saliency.py ==="
conda run -n myenv python "$SCRIPTS/ism/create_sprinzl_saliency.py" \
    --delta-dir "$DELTA" \
    --csv       "$DATA"  \
    --outdir    "$SP_DIR" \
    --max-len   194

# ── Step 2: FiLM ISM logos ────────────────────────────────────────────────────
log "=== Step 2: FiLM ISM logos ==="
mkdir -p "$FIGS/ism_logos_film"
conda run -n myenv python "$SCRIPTS/figures/plot_ism_logos.py" \
    --emb-dir  "$EMB"              \
    --ism-dir  "$SP_DIR"           \
    --csv      "$DATA"             \
    --outdir   "$FIGS/ism_logos_film"

# ── Step 3: Film comparison → figures/film_comparison ────────────────────────
log "=== Step 3: film_comparison ==="
mkdir -p "$FIGS/film_comparison"
conda run -n myenv python "$SCRIPTS/figures/compare_film_vs_base.py" \
    --base-dir     "$ROOT/gnn_final"   \
    --film-dir     "$ROOT/gnn_film"    \
    --film-sal-dir "$SP_DIR"           \
    --outdir       "$FIGS/film_comparison"

# ── Step 4: Film comparison → figures/gnn_eda_5iso ───────────────────────────
log "=== Step 4: gnn_eda_5iso (film panels) ==="
mkdir -p "$FIGS/gnn_eda_5iso"
conda run -n myenv python "$SCRIPTS/figures/compare_film_vs_base.py" \
    --base-dir     "$ROOT/gnn_final"   \
    --film-dir     "$ROOT/gnn_film"    \
    --film-sal-dir "$SP_DIR"           \
    --outdir       "$FIGS/gnn_eda_5iso"

# ── Step 5: Film divergence ───────────────────────────────────────────────────
log "=== Step 5: film_divergence ==="
mkdir -p "$FIGS/film_divergence"
conda run -n myenv python "$SCRIPTS/figures/plot_film_divergence.py" \
    --delta-dir "$DELTA"           \
    --csv       "$DATA"            \
    --outdir    "$FIGS/film_divergence"

log "=== All done ==="

# ── ntfy notification ─────────────────────────────────────────────────────────
curl -s -o /dev/null \
    -H "Title: roma: Sprinzl fix complete" \
    -H "Priority: default" \
    -d "film ISM logos, film_comparison, gnn_eda_5iso, film_divergence regenerated with per-order Sprinzl alignment" \
    ntfy.sh/romanagle || true
