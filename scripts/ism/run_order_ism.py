#!/usr/bin/env python3
"""
run_order_ism.py
----------------
Launch compute_ism_clade.py across multiple GPUs for culturable orders.

Queries the taxonomy CSV + GTDB metadata to find eligible orders, splits them
across --n-gpus processes, and launches them in parallel. Safe to re-run: already-
completed (iso, order) pairs are skipped automatically via --skip-existing.

To start with bacteria >=200 seqs, then later extend to >=100:
    python scripts/run_order_ism.py --min-seqs-bac 200
    python scripts/run_order_ism.py --min-seqs-bac 100   # new orders appended

Usage
-----
    python scripts/run_order_ism.py
    python scripts/run_order_ism.py --min-seqs-bac 100 --dry-run
    python scripts/run_order_ism.py --domain Archaea --min-seqs-arc 20
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

parser = argparse.ArgumentParser()
parser.add_argument("--domain",         default="Both", choices=["Bacteria", "Archaea", "Both"])
parser.add_argument("--min-seqs-bac",   type=int, default=200,
                    help="Min culturable sequences per bacterial order (default: 200)")
parser.add_argument("--min-seqs-arc",   type=int, default=50,
                    help="Min culturable sequences per archaeal order (default: 50)")
parser.add_argument("--csv",            default=f"{DATA_ROOT}/data/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--bac-metadata",   default="/data/kate/group_II/bac120_metadata_r226.tsv")
parser.add_argument("--arc-metadata",   default="/data/conner/GII/seq_2_taxonomy/GTDB_metadata/ar53_metadata.tsv")
parser.add_argument("--outdir",         default=f"{DATA_ROOT}/gnn_final/order_ism_culturable")
parser.add_argument("--emb-dir",        default=f"{DATA_ROOT}/embeddings/gnn")
parser.add_argument("--model-path",     default=f"{DATA_ROOT}/gnn_final/gnn_model.pt")
parser.add_argument("--n-gpus",         type=int, default=2)
parser.add_argument("--gpu-offset",     type=int, default=0,
                    help="First CUDA device index to use (default: 0)")
parser.add_argument("--python",         default=sys.executable,
                    help="Python executable to use for subprocesses")
parser.add_argument("--isotypes",       nargs="+", default=None,
                    help="Restrict to these isotypes (default: all)")
parser.add_argument("--ism-n-seqs",     type=int, default=100)
parser.add_argument("--dry-run",        action="store_true",
                    help="Print the commands that would be launched without running them")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

# ── Build culturable order list ───────────────────────────────────────────────
print("Loading taxonomy …", flush=True)
df = pd.read_csv(args.csv, usecols=["GenomeID", "domain", "order"], low_memory=False)

meta_frames = []
for meta_path in [args.bac_metadata, args.arc_metadata]:
    if os.path.exists(meta_path):
        meta_frames.append(pd.read_csv(
            meta_path, sep="\t",
            usecols=["accession", "ncbi_genome_category"],
            low_memory=False,
        ))
if not meta_frames:
    sys.exit("ERROR: no GTDB metadata files found")
meta_df = pd.concat(meta_frames, ignore_index=True)
meta_df["acc"] = meta_df["accession"].str.replace(r"^RS_|^GB_", "", regex=True)
culturable_accs = set(meta_df.loc[meta_df["ncbi_genome_category"] == "none", "acc"])

df["acc"] = df["GenomeID"].str.replace(r"_genomic$", "", regex=True)
df = df[df["acc"].isin(culturable_accs)]

eligible_orders: list[str] = []
for domain, min_seqs in [("Bacteria", args.min_seqs_bac), ("Archaea", args.min_seqs_arc)]:
    if args.domain not in (domain, "Both"):
        continue
    counts = df[df["domain"] == domain]["order"].value_counts()
    orders = list(counts[counts >= min_seqs].index)
    print(f"  {domain} >=  {min_seqs} seqs: {len(orders)} orders", flush=True)
    eligible_orders.extend(orders)

print(f"  Total eligible orders: {len(eligible_orders)}", flush=True)

# ── Skip orders already fully complete ───────────────────────────────────────
# An order is complete when every (iso, order) has a saliency file.
# Incomplete orders are re-queued; compute_ism_clade --skip-existing handles
# skipping individual (iso, order) pairs that are already done within a re-queued order.
import numpy as np

npz_data  = np.load(os.path.join(args.emb_dir, "perpos_embeddings.npz"), allow_pickle=True)
all_isos  = list(npz_data["classes"].astype(str))
check_isos = args.isotypes if args.isotypes else all_isos

def is_order_complete(order: str) -> bool:
    return os.path.exists(os.path.join(args.outdir, f"{order}.done"))

pending = [o for o in eligible_orders if not is_order_complete(o)]
done    = len(eligible_orders) - len(pending)
print(f"  Already complete: {done}  Pending: {len(pending)}", flush=True)

if not pending:
    print("All orders complete. Nothing to do.")
    sys.exit(0)

# ── Split pending orders across GPUs (round-robin for even load) ──────────────
chunks: list[list[str]] = [[] for _ in range(args.n_gpus)]
for i, order in enumerate(pending):
    chunks[i % args.n_gpus].append(order)

script = os.path.join(DATA_ROOT, "scripts/analysis/compute_ism_clade.py")

# ── Launch ────────────────────────────────────────────────────────────────────
procs = []
for gpu_i, chunk in enumerate(chunks):
    if not chunk:
        continue
    gpu_id  = args.gpu_offset + gpu_i
    cmd = [
        args.python, script,
        "--clades",        *chunk,
        "--clade-col",     "order",
        "--culturable-only",
        "--use-all",
        "--skip-existing",
        "--emb-dir",       args.emb_dir,
        "--model-path",    args.model_path,
        "--outdir",        args.outdir,
        "--ism-n-seqs",    str(args.ism_n_seqs),
    ]
    if args.isotypes:
        cmd += ["--isotypes", *args.isotypes]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = os.path.join(args.outdir, f"gpu{gpu_id}.log")
    print(f"\nGPU {gpu_id}: {len(chunk)} orders → {log_path}", flush=True)
    if args.dry_run:
        print("  " + " ".join(cmd))
        continue

    log_fh = open(log_path, "a")
    procs.append(subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=log_fh))

if args.dry_run:
    sys.exit(0)

print(f"\nLaunched {len(procs)} processes. Waiting …", flush=True)
for p in procs:
    p.wait()

# ── Final status report ───────────────────────────────────────────────────────
n_complete = sum(is_order_complete(o) for o in eligible_orders)
n_total    = len(eligible_orders)
print(f"\nDone. {n_complete}/{n_total} orders fully complete.")
if n_complete < n_total:
    still_pending = [o for o in eligible_orders if not is_order_complete(o)]
    print(f"Still incomplete ({len(still_pending)}): {still_pending[:10]}"
          + (" …" if len(still_pending) > 10 else ""))
    print("Re-run this script to resume.")
