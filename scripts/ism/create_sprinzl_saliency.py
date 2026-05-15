#!/usr/bin/env python3
"""
create_sprinzl_saliency.py
--------------------------
Re-aggregate FiLM per-order ISM saliency into domain-level arrays using
per-order Sprinzl maps, eliminating the raw-position misalignment that
occurs when sequences from different orders have different lengths/structures.

The existing gnn_film/gnn_ism_saliency_{iso}_{domain}.npy files average ISM
in raw-position space across all sequences in a domain. For orders whose
tRNAs have non-modal lengths, this misaligns positions — e.g. for Ala/Archaea,
Bathyarchaeales (modal 75 nt) has its discriminator at raw index 74, not 71,
so raw-space averaging smears their T-stem signal onto Sprinzl 73.

Fix: for each order, use the per-order modal secondary structure to build
the correct Sprinzl map, extract values at the right raw positions, and
accumulate in Sprinzl space weighted by sequence count. The resulting
Sprinzl-space averages are written back into a (max_len, 4) array using the
global (iso, domain) modal Sprinzl map so that downstream plot_ism_logos.py
requires no changes.

Output: gnn_film_sprinzl/gnn_ism_saliency_{iso}_{domain}.npy
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")
sys.path.insert(0, os.path.join(DATA_ROOT, "scripts"))
from utils.iso_classes import normalize_isotypes

parser = argparse.ArgumentParser()
parser.add_argument("--delta-dir", default=os.path.join(DATA_ROOT, "gnn_film/film_delta_ism"))
parser.add_argument("--csv",       default=os.path.join(DATA_ROOT, "data/trnascan_GTDB_r226_dedup_taxonomy.csv"))
parser.add_argument("--outdir",    default=os.path.join(DATA_ROOT, "gnn_film_sprinzl"))
parser.add_argument("--max-len",   type=int, default=194)
parser.add_argument("--force",     action="store_true")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
MAX_LEN = args.max_len


# ── Sprinzl utilities ─────────────────────────────────────────────────────────
def parse_pairs(ss):
    pairs, stack = {}, []
    for i, c in enumerate(ss):
        if c == ">": stack.append(i)
        elif c == "<":
            j = stack.pop(); pairs[j] = i; pairs[i] = j
    return pairs


def build_sprinzl_map(ss):
    pairs = parse_pairs(ss)
    n     = len(ss)
    fps   = sorted(i for i in pairs if i < pairs[i])
    stems, cur = [], []
    for p in fps:
        if not cur: cur = [p]
        elif p == cur[-1]+1 and pairs[p] == pairs[cur[-1]]-1: cur.append(p)
        else: stems.append(cur); cur = [p]
    if cur: stems.append(cur)
    if len(stems) < 4: return {}
    sp = {}
    acc5 = stems[0]; acc3 = sorted(pairs[p] for p in acc5)
    overhang = [p for p in range(acc5[0]) if p not in pairs]
    for i, p in enumerate(reversed(overhang)): sp[p] = str(-(i+1))
    for i, p in enumerate(acc5): sp[p] = str(i+1)
    for i, p in enumerate(acc3): sp[p] = str(66+i)
    disc = sorted(p for p in range(acc3[-1]+1, n) if p not in pairs)
    for i, p in enumerate(disc): sp[p] = str(73+i)
    d5 = stems[1]
    for i, p in enumerate([p for p in range(acc5[-1]+1, d5[0]) if p not in pairs]):
        sp[p] = str(8+i)
    d3 = sorted(pairs[p] for p in d5); nd = len(d5)
    for i, p in enumerate(d5): sp[p] = str(10+i)
    for i, p in enumerate(d3): sp[p] = str(25-nd+1+i)
    d_loop = [p for p in range(d5[-1]+1, d3[0]) if p not in pairs]
    for i, p in enumerate(d_loop):
        labs = ["14","15","16","17","17a","18","19","20","20a","20b","21"]
        if i < len(labs): sp[p] = labs[i]
    ac5 = stems[2]
    lk2 = [p for p in range(d3[-1]+1, ac5[0]) if p not in pairs]
    if lk2: sp[lk2[-1]] = "26"
    ac3 = sorted(pairs[p] for p in ac5)
    for i, p in enumerate(ac5): sp[p] = str(27+i)
    for i, p in enumerate(ac3): sp[p] = str(39+i)
    for i, p in enumerate([p for p in range(ac5[-1]+1, ac3[0]) if p not in pairs]):
        sp[p] = str(32+i)
    t5 = stems[-1]; t3 = sorted(pairs[p] for p in t5)
    for i, p in enumerate(t5): sp[p] = str(49+i)
    for i, p in enumerate(t3): sp[p] = str(61+i)
    for i, p in enumerate([p for p in range(t5[-1]+1, t3[0]) if p not in pairs]):
        sp[p] = str(54+i)
    var = [p for p in range(ac3[-1]+1, t5[0])]
    if len(stems) == 5:
        for i, p in enumerate(var): sp[p] = f"e{i+1}"
    else:
        for i, p in enumerate([p for p in var if p not in pairs]): sp[p] = str(44+i)
    return sp


# ── Load CSV ──────────────────────────────────────────────────────────────────
print("Loading CSV …", flush=True)
df = pd.read_csv(
    args.csv,
    usecols=["primary_sequence", "secondary_structure",
             "Anticodon_predicted_isotype", "order", "domain"],
    low_memory=False,
).dropna(subset=["primary_sequence", "secondary_structure",
                 "Anticodon_predicted_isotype", "order", "domain"])
df = normalize_isotypes(df)

order_counts = df.groupby(["Anticodon_predicted_isotype", "order"]).size().to_dict()
order_domain = df.groupby("order")["domain"].agg(lambda x: x.mode()[0]).to_dict()

print(f"  {len(df):,} sequences, {df['order'].nunique()} orders", flush=True)


# ── Sprinzl map caches ────────────────────────────────────────────────────────
_order_sp_cache: dict = {}
def get_order_sp(iso, order) -> dict:
    key = (iso, order)
    if key not in _order_sp_cache:
        sub = df[(df["Anticodon_predicted_isotype"] == iso) &
                 (df["order"] == order)]
        if sub.empty:
            _order_sp_cache[key] = {}
        else:
            ml = sub["primary_sequence"].str.len().mode()[0]
            ss = sub[sub["primary_sequence"].str.len() == ml]["secondary_structure"].mode()[0]
            _order_sp_cache[key] = build_sprinzl_map(ss)
    return _order_sp_cache[key]


_global_sp_cache: dict = {}
def get_global_sp(iso, domain) -> dict:
    key = (iso, domain)
    if key not in _global_sp_cache:
        sub = df[(df["Anticodon_predicted_isotype"] == iso) &
                 (df["domain"] == domain)]
        if sub.empty:
            _global_sp_cache[key] = {}
        else:
            ml = sub["primary_sequence"].str.len().mode()[0]
            ss = sub[sub["primary_sequence"].str.len() == ml]["secondary_structure"].mode()[0]
            _global_sp_cache[key] = build_sprinzl_map(ss)
    return _global_sp_cache[key]


# ── Discover film_ism_on files ────────────────────────────────────────────────
film_on: dict[str, dict[str, str]] = {}   # iso -> {order -> path}
for fn in os.listdir(args.delta_dir):
    m = re.match(r"film_ism_on_(\w+)_(.+)\.npy$", fn)
    if m:
        iso, order = m.group(1), m.group(2)
        film_on.setdefault(iso, {})[order] = os.path.join(args.delta_dir, fn)

print(f"Found {sum(len(v) for v in film_on.values())} film_ism_on files "
      f"for {len(film_on)} isotypes", flush=True)


# ── Aggregate per (iso, domain) in Sprinzl space ────────────────────────────
for iso, order_paths in sorted(film_on.items()):
    domain_orders: dict[str, list[str]] = {}
    for order in order_paths:
        dom = order_domain.get(order)
        if dom:
            domain_orders.setdefault(dom, []).append(order)

    global_sp_all = {}
    for dom in domain_orders:
        gsp = get_global_sp(iso, dom)
        for raw, sp in gsp.items():
            global_sp_all[sp] = raw  # last write wins, but same iso so consistent

    for domain, orders in sorted(domain_orders.items()):
        out_path = os.path.join(args.outdir, f"gnn_ism_saliency_{iso}_{domain}.npy")
        if os.path.exists(out_path) and not args.force:
            print(f"  [{iso}/{domain}] exists — skipping (use --force to overwrite)",
                  flush=True)
            continue

        print(f"  [{iso}/{domain}]  {len(orders)} orders …", flush=True)

        sp_sum: dict[str, np.ndarray] = {}
        sp_wt:  dict[str, float]      = {}

        for order in orders:
            arr   = np.load(order_paths[order])          # (max_len, 4)
            n_seq = order_counts.get((iso, order), 1)
            sp_map = get_order_sp(iso, order)
            if not sp_map:
                continue
            for raw_pos, sp_lbl in sp_map.items():
                if raw_pos >= arr.shape[0]:
                    continue
                val = arr[raw_pos].astype(np.float64)
                if sp_lbl not in sp_sum:
                    sp_sum[sp_lbl] = np.zeros(4, dtype=np.float64)
                    sp_wt[sp_lbl]  = 0.0
                sp_sum[sp_lbl] += val * n_seq
                sp_wt[sp_lbl]  += n_seq

        # Write back to (max_len, 4) array at global Sprinzl raw positions
        global_sp = get_global_sp(iso, domain)
        global_sp_inv = {v: k for k, v in global_sp.items()}   # sp_lbl -> raw

        out_arr = np.zeros((MAX_LEN, 4), dtype=np.float32)
        n_placed, n_missing = 0, 0
        for sp_lbl, wt in sp_wt.items():
            if wt == 0:
                continue
            if sp_lbl not in global_sp_inv:
                n_missing += 1
                continue
            raw = global_sp_inv[sp_lbl]
            if raw >= MAX_LEN:
                continue
            out_arr[raw] = (sp_sum[sp_lbl] / wt).astype(np.float32)
            n_placed += 1

        np.save(out_path, out_arr)
        print(f"    placed {n_placed} Sprinzl positions, {n_missing} had no global mapping",
              flush=True)

print("\nDone.", flush=True)
