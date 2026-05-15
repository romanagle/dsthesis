#!/usr/bin/env bash
# setup_env.sh
# Creates the 'roma' conda environment with all dependencies needed to run
# the tRNA embedding pipeline (RNA-FM + RiNALMo).
#
# Usage:
#   bash setup_env.sh
#
# After setup, activate with:
#   conda activate roma

set -euo pipefail

ENV_NAME="roma"
PYTHON_VERSION="3.11"
CONDA_BASE="$(conda info --base)"
ENV_DIR="${CONDA_BASE}/envs/${ENV_NAME}"
PIP="${ENV_DIR}/bin/pip"
PYTHON="${ENV_DIR}/bin/python"

# PyTorch >= 2.6 required for flash-attn 2.8 compatibility.
# Driver is backwards-compatible with cu124 wheels.
TORCH_INDEX="https://download.pytorch.org/whl/cu124"

echo "============================================"
echo " Setting up conda environment: ${ENV_NAME}"
echo "============================================"
echo "  Conda base: ${CONDA_BASE}"
echo "  Env dir:    ${ENV_DIR}"
echo ""

# ── 1. Create environment ─────────────────────────────────────────────────────
if [ -d "${ENV_DIR}" ]; then
    echo "[1/5] Environment '${ENV_NAME}' already exists — skipping creation."
    echo "      To recreate from scratch: conda env remove -n ${ENV_NAME}"
else
    echo "[1/5] Creating conda environment (Python ${PYTHON_VERSION}) …"
    conda create -y -n "${ENV_NAME}" python="${PYTHON_VERSION}"
fi

# ── 2. PyTorch ────────────────────────────────────────────────────────────────
echo "[2/5] Installing PyTorch (cu124) …"
"${PIP}" install --quiet \
    "torch>=2.6" torchvision torchaudio \
    --index-url "${TORCH_INDEX}"

# ── 3. RNA-FM ─────────────────────────────────────────────────────────────────
echo "[3/5] Installing RNA-FM …"
"${PIP}" install --quiet rna-fm

# ── 4. RiNALMo ───────────────────────────────────────────────────────────────
echo "[4/5] Installing RiNALMo …"
"${PIP}" install --quiet git+https://github.com/lbcb-sci/RiNALMo.git

# flash-attn is required by the RiNALMo giga model (weights use fused QKV).
# Builds CUDA kernels from source — takes ~15 min.
echo "      Installing flash-attn (compiling CUDA kernels, ~15 min) …"
"${PIP}" install flash-attn --no-build-isolation

# flash-attn 2.8+ dropped the pos_idx_in_fp32 arg from RotaryEmbedding.
# Patch rinalmo's attention.py to remove it.
ATTN_FILE="${ENV_DIR}/lib/python${PYTHON_VERSION}/site-packages/rinalmo/model/attention.py"
"${PYTHON}" - "${ATTN_FILE}" <<'PYEOF'
import sys
path = sys.argv[1]
text = open(path).read()
# flash-attn 2.8+: RotaryEmbedding dropped pos_idx_in_fp32
text = text.replace("                pos_idx_in_fp32=True,  # fp32 RoPE precision\n", "")
# flash-attn 2.8+: unpad_input now returns 5 values instead of 4
text = text.replace(
    "x_unpad, indices, cu_seqlens, max_s = unpad_input(qkv, key_padding_mask)",
    "x_unpad, indices, cu_seqlens, max_s, _ = unpad_input(qkv, key_padding_mask)",
)
open(path, "w").write(text)
print("      Patched rinalmo attention.py.")
PYEOF

# ── 5. Other dependencies ─────────────────────────────────────────────────────
echo "[5/5] Installing remaining dependencies …"
"${PIP}" install --quiet numpy pandas umap-learn matplotlib

# ── Verify ────────────────────────────────────────────────────────────────────
echo ""
echo "Verifying installs …"
"${PYTHON}" - <<'EOF'
import torch, fm, rinalmo
print(f"  torch    {torch.__version__}  (CUDA available: {torch.cuda.is_available()})")
print(f"  fm       {fm.__version__}")
print(f"  rinalmo  ok")
EOF

echo ""
echo "Done. Activate with:  conda activate ${ENV_NAME}"
echo "Then run the pipeline:"
echo "  python scripts/mask_anticodon.py"
echo "  python scripts/extract_rnafm_full.py"
echo "  python scripts/extract_rnafm_full.py --masked"
echo "  python scripts/extract_rinalmo_full.py"
echo "  python scripts/extract_rinalmo_full.py --masked"
