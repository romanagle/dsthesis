#!/usr/bin/env python3
"""
classify_dedup.py
-----------------
Train a logistic regression classifier (4-mer frequency features) on each
taxonomically-deduplicated dataset and save accuracy results as JSON.

Run once per level; results are read by plot_dedup_sprinzl_accuracy.py.

Usage:
    python scripts/pipeline/classify_dedup.py
    python scripts/pipeline/classify_dedup.py --outdir figures/dedup/results
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")

parser = argparse.ArgumentParser()
parser.add_argument("--dedup-dir", default="/data/roma/data/dedup",
                    help="Directory containing dedup_{level}.csv files")
parser.add_argument("--raw-csv",   default="/data/roma/archive/trnascan_GTDB_r226_taxonomy.csv",
                    help="Full raw taxonomy CSV (used for the 'none' level)")
parser.add_argument("--outdir",    default=f"{DATA_ROOT}/figures/dedup/results")
parser.add_argument("--n-cap",     type=int, default=200_000,
                    help="Max sequences to use per level (default: 200000)")
parser.add_argument("--seed",      type=int, default=42)
args = parser.parse_args()

OUTDIR = Path(args.outdir)
OUTDIR.mkdir(parents=True, exist_ok=True)

LEVELS     = ["none", "species", "genus", "family", "order", "class", "phylum", "domain"]
DROP_ISOS  = {"SeC", "Undet", "Sup"}
MERGE_ISOS = {"Ile2": "Ile"}
TEST_SIZE  = 0.2

# ── 4-mer frequency features ──────────────────────────────────────────────────
NUCS  = "ACGT"
KMERS = [a+b+c+d for a in NUCS for b in NUCS for c in NUCS for d in NUCS]
KMER_IDX = {k: i for i, k in enumerate(KMERS)}
K = 4


def kmer_features(seqs):
    X = np.zeros((len(seqs), len(KMERS)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        seq = seq.upper().replace("U", "T")
        n   = len(seq) - K + 1
        if n <= 0:
            continue
        for j in range(n):
            mer = seq[j:j+K]
            if mer in KMER_IDX:
                X[i, KMER_IDX[mer]] += 1
        X[i] /= n
    return X


# ── Per-level loop ─────────────────────────────────────────────────────────────
for level in LEVELS:
    out_path = OUTDIR / f"result_{level}.json"

    csv_path = (args.raw_csv if level == "none"
                else os.path.join(args.dedup_dir, f"dedup_{level}.csv"))

    if not os.path.exists(csv_path):
        print(f"  Missing {csv_path} — skipping {level}")
        continue

    print(f"\n=== {level} ===", flush=True)
    df = pd.read_csv(csv_path,
                     usecols=["primary_sequence", "Anticodon_predicted_isotype"],
                     low_memory=False)
    df = df.dropna(subset=["primary_sequence", "Anticodon_predicted_isotype"])
    df = df[~df["Anticodon_predicted_isotype"].isin(DROP_ISOS)].copy()
    df["Anticodon_predicted_isotype"] = df["Anticodon_predicted_isotype"].replace(MERGE_ISOS)

    n_total  = len(df)
    n_classes = df["Anticodon_predicted_isotype"].nunique()
    print(f"  {n_total:,} rows  {n_classes} classes", flush=True)

    n_used = min(n_total, args.n_cap)
    if n_total > args.n_cap:
        df = df.sample(args.n_cap, random_state=args.seed)

    print("  Computing features …", flush=True)
    X = kmer_features(df["primary_sequence"].tolist())
    le = LabelEncoder()
    y  = le.fit_transform(df["Anticodon_predicted_isotype"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=args.seed, stratify=y
    )

    print(f"  Training  (train={len(X_tr):,}  test={len(X_te):,}) …", flush=True)
    t0  = time.time()
    clf = LogisticRegression(max_iter=2000, random_state=args.seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    fit_time = time.time() - t0

    acc = accuracy_score(y_te, clf.predict(X_te))
    print(f"  Test acc: {acc:.4f}   fit_time: {fit_time:.1f}s", flush=True)

    result = {
        "label":        level,
        "n_rows_total": n_total,
        "n_used":       n_used,
        "n_train":      len(X_tr),
        "n_test":       len(X_te),
        "n_classes":    n_classes,
        "test_acc":     round(float(acc), 6),
        "fit_time_s":   round(fit_time, 3),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved → {out_path}", flush=True)

print("\nDone.")
