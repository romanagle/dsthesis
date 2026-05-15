#!/usr/bin/env python3
"""
run_family_ism.py
-----------------
Generate batched compute_ism_clade.py commands to compute ISM saliency maps
for all qualifying bacterial clades at a given taxonomic level.
Prints one command per batch to stdout.

Usage
-----
    python scripts/run_family_ism.py | bash
    python scripts/run_family_ism.py --clade-col order | bash
    python scripts/run_family_ism.py --clade-col family --batch-size 20 > jobs.sh
    python scripts/run_family_ism.py --dry-run
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

parser = argparse.ArgumentParser()
parser.add_argument("--csv",        default=f"{DATA_ROOT}/data/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--clade-col",  default="order",
                    help="Taxonomic column to use (order, family, class, …)")
parser.add_argument("--outdir",     default=None,
                    help="Output directory (default: gnn_final/{clade_col}_ism)")
parser.add_argument("--emb-dir",    default=f"{DATA_ROOT}/embeddings/gnn")
parser.add_argument("--model-path", default=f"{DATA_ROOT}/gnn_final/gnn_model.pt")
parser.add_argument("--domain",     default="Bacteria")
parser.add_argument("--min-count",  type=int, default=50,
                    help="Min sequences per clade (total) to include")
parser.add_argument("--batch-size", type=int, default=20,
                    help="Clades per compute_ism_clade.py call")
parser.add_argument("--ism-n-seqs", type=int, default=100)
parser.add_argument("--dry-run",    action="store_true",
                    help="Print summary without emitting commands")
args = parser.parse_args()

if args.outdir is None:
    args.outdir = f"{DATA_ROOT}/gnn_final/{args.clade_col}_ism"

df = pd.read_csv(args.csv, usecols=["domain", args.clade_col], low_memory=False)
if args.domain != "All":
    df = df[df["domain"] == args.domain]

clade_counts = df[args.clade_col].value_counts()
qualifying   = [c for c in clade_counts.index if clade_counts[c] >= args.min_count]
n_batches    = (len(qualifying) + args.batch_size - 1) // args.batch_size

print(f"# {len(qualifying)} qualifying {args.domain} {args.clade_col}s (>= {args.min_count} seqs)")
print(f"# {n_batches} batches of {args.batch_size}")
print(f"# Output: {args.outdir}")
print(f"# Largest:  {qualifying[0]} ({clade_counts[qualifying[0]]:,} seqs)")
print(f"# Smallest: {qualifying[-1]} ({clade_counts[qualifying[-1]]:,} seqs)")

if args.dry_run:
    raise SystemExit(0)

print()
os.makedirs(args.outdir, exist_ok=True)
script = os.path.join(DATA_ROOT, "scripts", "analysis", "compute_ism_clade.py")

for i in range(0, len(qualifying), args.batch_size):
    batch      = qualifying[i:i + args.batch_size]
    clades_str = " ".join(f"'{c}'" for c in batch)
    print(
        f"python {script} "
        f"--clade-col {args.clade_col} "
        f"--emb-dir {args.emb_dir} "
        f"--model-path {args.model_path} "
        f"--outdir {args.outdir} "
        f"--ism-n-seqs {args.ism_n_seqs} "
        f"--use-all "
        f"--skip-existing "
        f"--clades {clades_str}"
    )
