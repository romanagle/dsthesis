#!/usr/bin/env bash
set -euo pipefail

# Build and install Infernal (cmsearch/cmscan) and tRNAscan-SE from source.
# Assumes you already have the source directories unpacked.
# Optional: install Infernal via Homebrew if requested.

PREFIX="${PREFIX:-$HOME/.local}"
NPROC="${NPROC:-1}"

INFERNAL_SRC_DIR="${INFERNAL_SRC_DIR:-infernal-1.1.2}"
TRNASCAN_SRC_DIR="${TRNASCAN_SRC_DIR:-tRNAscan-SE-master}"
USE_HOMEBREW_INFERNAL="${USE_HOMEBREW_INFERNAL:-0}"

echo "Using PREFIX=$PREFIX"
echo "Using INFERNAL_SRC_DIR=$INFERNAL_SRC_DIR"
echo "Using TRNASCAN_SRC_DIR=$TRNASCAN_SRC_DIR"
echo "Using USE_HOMEBREW_INFERNAL=$USE_HOMEBREW_INFERNAL"

if [[ ! -d "$INFERNAL_SRC_DIR" ]]; then
  if [[ "$USE_HOMEBREW_INFERNAL" == "1" ]]; then
    if command -v brew >/dev/null 2>&1; then
      echo "Installing Infernal via Homebrew..."
      brew install infernal
    else
      echo "Homebrew not found. Install it or set INFERNAL_SRC_DIR." >&2
      exit 1
    fi
  else
    echo "Missing Infernal source dir: $INFERNAL_SRC_DIR" >&2
    echo "Set INFERNAL_SRC_DIR or run with USE_HOMEBREW_INFERNAL=1." >&2
    exit 1
  fi
fi

if [[ -d "$INFERNAL_SRC_DIR" ]]; then
  echo "Building Infernal..."
  pushd "$INFERNAL_SRC_DIR" >/dev/null
  ./configure --prefix="$PREFIX"
  make -j "$NPROC"
  make install
  popd >/dev/null
fi

if [[ ! -d "$TRNASCAN_SRC_DIR" ]]; then
  echo "Missing tRNAscan-SE source dir: $TRNASCAN_SRC_DIR" >&2
  exit 1
fi

echo "Building tRNAscan-SE..."
pushd "$TRNASCAN_SRC_DIR" >/dev/null
./configure --prefix="$PREFIX"
make -j "$NPROC"
make install
popd >/dev/null

echo "Done. Ensure $PREFIX/bin is on your PATH."
echo "Verify with: command -v cmsearch cmscan tRNAscan-SE"
