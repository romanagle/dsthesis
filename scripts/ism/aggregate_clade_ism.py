#!/usr/bin/env python3
"""
aggregate_clade_ism.py
----------------------
Aggregate order-level per-sequence ISM files to higher taxonomic levels
(family, class, phylum) without re-running the model.

For each (higher_clade, isotype):
  1. Find all completed order-level perseq files whose order maps to this clade
  2. Concatenate their per-sequence stacks
  3. Compute mean saliency and SEM

Writes the same file format as compute_ism_clade.py:
  {outdir}/gnn_ism_saliency_{iso}_{clade}.npy   (max_len, 4)
  {outdir}/gnn_ism_perseq_{iso}_{clade}.npy     (n_seqs, max_len, 4)
  {outdir}/gnn_ism_sem_{iso}_{clade}.npy        (max_len, 4)
  {outdir}/aggregate_metadata.csv               which orders went into each clade

So all downstream plotting scripts (plot_ism_logos.py etc.) work unchanged.

Usage
-----
    # Aggregate to family level
    python scripts/analysis/aggregate_clade_ism.py --clade-col family

    # Aggregate to class or phylum
    python scripts/analysis/aggregate_clade_ism.py --clade-col class
    python scripts/analysis/aggregate_clade_ism.py --clade-col phylum

    # Restrict to specific clades or isotypes
    python scripts/analysis/aggregate_clade_ism.py --clade-col class \\
        --clades Gammaproteobacteria Actinomycetes
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

parser = argparse.ArgumentParser()
parser.add_argument("--clade-col",  required=True,
                    choices=["family", "class", "phylum"],
                    help="Taxonomic level to aggregate to")
parser.add_argument("--perseq-dir", default=f"{DATA_ROOT}/gnn_final/order_ism_culturable",
                    help="Directory containing order-level gnn_ism_perseq_*.npy files")
parser.add_argument("--csv",        default=f"{DATA_ROOT}/data/trnascan_GTDB_r226_dedup_taxonomy.csv")
parser.add_argument("--outdir",     default=None,
                    help="Output directory (default: {perseq-dir}/../{clade_col}_ism_culturable)")
parser.add_argument("--clades",     nargs="+", default=None,
                    help="Restrict to these higher-level clades (default: all with >=1 order)")
parser.add_argument("--isotypes",   nargs="+", default=None,
                    help="Restrict to these isotypes (default: infer from perseq files)")
parser.add_argument("--min-orders", type=int, default=1,
                    help="Skip higher clades with fewer than this many contributing orders")
parser.add_argument("--skip-existing", action="store_true")
args = parser.parse_args()

if args.outdir is None:
    base = os.path.dirname(args.perseq_dir)
    args.outdir = os.path.join(base, f"{args.clade_col}_ism_culturable")
os.makedirs(args.outdir, exist_ok=True)

# ── Discover available (iso, order) perseq files ─────────────────────────────
print(f"Scanning {args.perseq_dir} for perseq files …", flush=True)
available: dict[tuple[str, str], str] = {}   # (iso, order) → path
for fname in os.listdir(args.perseq_dir):
    if not (fname.startswith("gnn_ism_perseq_") and fname.endswith(".npy")):
        continue
    stem  = fname[len("gnn_ism_perseq_"):-len(".npy")]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        available[(parts[0], parts[1])] = os.path.join(args.perseq_dir, fname)

completed_orders = sorted(set(order for _, order in available))
completed_isos   = sorted(set(iso   for iso, _   in available))
print(f"  Found {len(available)} perseq files: "
      f"{len(completed_orders)} orders × {len(completed_isos)} isotypes", flush=True)

if not available:
    raise SystemExit("No perseq files found. Run compute_ism_clade.py first.")

iso_list = args.isotypes if args.isotypes else completed_isos

# ── Build order → higher_clade map from CSV ───────────────────────────────────
print(f"Loading order → {args.clade_col} map …", flush=True)
tax_df = pd.read_csv(
    args.csv,
    usecols=["order", args.clade_col],
    low_memory=False,
).dropna().drop_duplicates("order")
order_to_higher = dict(zip(tax_df["order"], tax_df[args.clade_col]))

# Map completed orders to their higher clade
higher_to_orders: dict[str, list[str]] = {}
for order in completed_orders:
    higher = order_to_higher.get(order)
    if higher:
        higher_to_orders.setdefault(higher, []).append(order)

if args.clades:
    higher_to_orders = {k: v for k, v in higher_to_orders.items() if k in args.clades}

if args.min_orders > 1:
    before = len(higher_to_orders)
    higher_to_orders = {k: v for k, v in higher_to_orders.items()
                        if len(v) >= args.min_orders}
    print(f"  min-orders filter: {before} → {len(higher_to_orders)} clades", flush=True)

print(f"  {len(higher_to_orders)} {args.clade_col}-level clades to aggregate", flush=True)

# ── Aggregate ─────────────────────────────────────────────────────────────────
meta_rows = []

for higher_clade, orders in sorted(higher_to_orders.items()):
    print(f"\n[{higher_clade}]  {len(orders)} orders: {orders}", flush=True)

    for iso in iso_list:
        sal_path    = os.path.join(args.outdir, f"gnn_ism_saliency_{iso}_{higher_clade}.npy")
        perseq_path = os.path.join(args.outdir, f"gnn_ism_perseq_{iso}_{higher_clade}.npy")
        sem_path    = os.path.join(args.outdir, f"gnn_ism_sem_{iso}_{higher_clade}.npy")

        if args.skip_existing and os.path.exists(sal_path) and os.path.exists(perseq_path):
            print(f"  [{iso}] exists — skipping", flush=True)
            continue

        # Collect perseq arrays from all contributing orders
        arrays = []
        contributing = []
        for order in orders:
            path = available.get((iso, order))
            if path is None:
                continue
            arr = np.load(path)   # (n_seqs, max_len, 4)
            arrays.append(arr)
            contributing.append(order)

        if not arrays:
            print(f"  [{iso}] no perseq files found — skipping", flush=True)
            continue

        stack   = np.concatenate(arrays, axis=0)   # (total_seqs, max_len, 4)
        n_seqs  = stack.shape[0]
        mean_sal = stack.mean(axis=0)               # (max_len, 4)
        sem      = stack.std(axis=0) / np.sqrt(n_seqs)

        np.save(sal_path,    mean_sal)
        np.save(perseq_path, stack)
        np.save(sem_path,    sem)

        meta_rows.append({
            args.clade_col: higher_clade,
            "isotype":       iso,
            "n_orders":      len(contributing),
            "n_seqs":        n_seqs,
            "contributing_orders": "|".join(contributing),
        })
        print(f"  [{iso}] {len(contributing)} orders, {n_seqs} seqs → saved", flush=True)

# ── Save metadata ─────────────────────────────────────────────────────────────
if meta_rows:
    meta_df = pd.DataFrame(meta_rows)
    meta_path = os.path.join(args.outdir, "aggregate_metadata.csv")
    if os.path.exists(meta_path):
        existing = pd.read_csv(meta_path)
        meta_df  = pd.concat([existing, meta_df]).drop_duplicates(
            subset=[args.clade_col, "isotype"], keep="last"
        )
    meta_df.to_csv(meta_path, index=False)
    print(f"\nSaved metadata → {meta_path}", flush=True)

print(f"\nDone. Results in: {args.outdir}/")
print(f"\nPlot with:")
print(f"  python scripts/analysis/plot_ism_logos.py \\")
print(f"    --emb-dir embeddings/gnn \\")
print(f"    --ism-dir {args.outdir} \\")
print(f"    --outdir {args.outdir}/logos \\")
print(f"    --clade-col {args.clade_col}")
