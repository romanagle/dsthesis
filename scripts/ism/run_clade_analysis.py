#!/usr/bin/env python3
"""
run_clade_analysis.py
---------------------
One-command wrapper: train clade GNN → compute ISM → plot logos.

Usage
-----
    python scripts/run_clade_analysis.py \\
        --clade Acidimicrobiales \\
        --clade-col order \\
        --isotypes Thr Asp

    python scripts/run_clade_analysis.py \\
        --clade Pelagibacterales \\
        --clade-col order \\
        --isotypes His \\
        --parent-col class
"""

from __future__ import annotations
import argparse, subprocess, sys, os
import os
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")


parser = argparse.ArgumentParser()
parser.add_argument("--clade",           required=True)
parser.add_argument("--clade-col",       default="order")
parser.add_argument("--isotypes",        nargs="+", required=True,
                    help="Isotypes to plot (all isotypes are still computed for "
                         "cross-isotype normalization)")
parser.add_argument("--emb-dir",         default="embeddings/gnn")
parser.add_argument("--results-dir",     default="results",
                    help="Root directory for outputs (default: .)")
parser.add_argument("--parent-col",      default="class",
                    help="Taxonomic level for parent-class augmentation (default: class)")
parser.add_argument("--parent-weight",   default=None, type=float)
parser.add_argument("--parent-max-seqs", default=5000,  type=int)
parser.add_argument("--ism-n-seqs",      default=100,   type=int)
parser.add_argument("--skip-train",      action="store_true",
                    help="Skip training if model already exists")
parser.add_argument("--skip-existing",   action="store_true",
                    help="Skip ISM files that already exist")
parser.add_argument("--csv",  default="data/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--min-sal", default=0.002, type=float)
args = parser.parse_args()

clade_slug = args.clade.lower().replace(" ", "_")
model_dir  = os.path.join(args.results_dir, f"gnn_{clade_slug}")
ism_dir    = os.path.join(model_dir, "clade_ism")
logo_dir   = os.path.join(ism_dir, "logos")
model_path = os.path.join(model_dir, "gnn_model.pt")

def run(cmd: list[str], label: str) -> None:
    print(f"\n{'='*60}", flush=True)
    print(f"[{label}]", flush=True)
    print("  " + " ".join(cmd), flush=True)
    print('='*60, flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(f"\nERROR: [{label}] failed (exit {result.returncode})")

# ── 1. Train ──────────────────────────────────────────────────────────────────
if args.skip_train and os.path.exists(model_path):
    print(f"Skipping training — model exists: {model_path}", flush=True)
else:
    train_cmd = [
        sys.executable, "scripts/train_gnn_clade.py",
        "--clade",           args.clade,
        "--clade-col",       args.clade_col,
        "--emb-dir",         args.emb_dir,
        "--outdir",          model_dir,
        "--csv",             args.csv,
    ]
    if args.parent_col:
        train_cmd += ["--parent-col", args.parent_col]
    if args.parent_weight is not None:
        train_cmd += ["--parent-weight", str(args.parent_weight)]
    train_cmd += ["--parent-max-seqs", str(args.parent_max_seqs)]
    run(train_cmd, "train")

# ── 2. ISM — all isotypes (needed for cross-isotype normalization) ────────────
ism_cmd_base = [
    sys.executable, "scripts/compute_ism_clade.py",
    "--emb-dir",    args.emb_dir,
    "--model-path", model_path,
    "--outdir",     ism_dir,
    "--clades",     args.clade,
    "--clade-col",  args.clade_col,
    "--csv",        args.csv,
    "--ism-n-seqs", str(args.ism_n_seqs),
    "--use-all",
]
if args.skip_existing:
    ism_cmd_base.append("--skip-existing")

run(ism_cmd_base, "ISM (all isotypes)")

# ── 3. Plot ───────────────────────────────────────────────────────────────────
plot_cmd = [
    sys.executable, "scripts/plot_ism_logos.py",
    "--emb-dir",   args.emb_dir,
    "--ism-dir",   ism_dir,
    "--outdir",    logo_dir,
    "--csv",       args.csv,
    "--clade-col", args.clade_col,
    "--isotypes",  *args.isotypes,
    "--min-sal",   str(args.min_sal),
]
run(plot_cmd, "plot")

print(f"\nDone. Logos in: {logo_dir}/")
