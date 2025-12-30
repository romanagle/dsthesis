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
TRNASCAN_TARBALL="${TRNASCAN_TARBALL:-}"

echo "Using PREFIX=$PREFIX"
echo "Using INFERNAL_SRC_DIR=$INFERNAL_SRC_DIR"
echo "Using TRNASCAN_SRC_DIR=$TRNASCAN_SRC_DIR"
echo "Using USE_HOMEBREW_INFERNAL=$USE_HOMEBREW_INFERNAL"
if [[ -n "$TRNASCAN_TARBALL" ]]; then
  echo "Using TRNASCAN_TARBALL=$TRNASCAN_TARBALL"
fi

if [[ ! -d "$INFERNAL_SRC_DIR" ]]; then
  if [[ "$USE_HOMEBREW_INFERNAL" == "1" ]]; then
    if command -v brew >/dev/null 2>&1; then
      echo "Installing Infernal via Homebrew..."
      brew tap brewsci/bio
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

if [[ -n "$TRNASCAN_TARBALL" ]]; then
  if [[ -f "$TRNASCAN_TARBALL" ]]; then
    trna_top_dir="$(tar -tzf "$TRNASCAN_TARBALL" | head -1 | cut -d/ -f1)"
    tar -xzf "$TRNASCAN_TARBALL"
    TRNASCAN_SRC_DIR="$trna_top_dir"
    echo "Extracted tRNAscan-SE to $TRNASCAN_SRC_DIR"
  else
    echo "Missing tRNAscan-SE tarball: $TRNASCAN_TARBALL" >&2
    exit 1
  fi
fi

if [[ ! -d "$TRNASCAN_SRC_DIR" ]]; then
  echo "Missing tRNAscan-SE source dir: $TRNASCAN_SRC_DIR" >&2
  echo "Set TRNASCAN_SRC_DIR or TRNASCAN_TARBALL to a release tarball." >&2
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
