#!/usr/bin/env bash
set -euo pipefail

# Build and install Infernal (cmsearch/cmscan) and tRNAscan-SE from source.
# Assumes you already have the source directories unpacked.

PREFIX="${PREFIX:-$HOME/.local}"
NPROC="${NPROC:-1}"

INFERNAL_SRC_DIR="${INFERNAL_SRC_DIR:-infernal-1.1.2}"
TRNASCAN_SRC_DIR="${TRNASCAN_SRC_DIR:-tRNAscan-SE-master}"

echo "Using PREFIX=$PREFIX"
echo "Using INFERNAL_SRC_DIR=$INFERNAL_SRC_DIR"
echo "Using TRNASCAN_SRC_DIR=$TRNASCAN_SRC_DIR"

if [[ ! -d "$INFERNAL_SRC_DIR" ]]; then
  echo "Missing Infernal source dir: $INFERNAL_SRC_DIR" >&2
  exit 1
fi

echo "Building Infernal..."
pushd "$INFERNAL_SRC_DIR" >/dev/null
./configure --prefix="$PREFIX"
make -j "$NPROC"
make install
popd >/dev/null

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
