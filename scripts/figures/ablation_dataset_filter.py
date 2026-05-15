#!/usr/bin/env python3
"""
ablation_dataset_filter.py
---------------------------
Ablation experiment: compare isotype classification accuracy across four
progressively filtered versions of the dataset, using a fixed classifier
(one-hot encoded anticodon-masked sequence → logistic regression softmax).

Dataset variants (each built from scratch):
  1. baseline      — all sequences passing sprinzl_ok + valid isotype (~4.7M)
  2. dedup         — baseline + family-level dedup (keep highest conf_score)
  3. dedup+variant — dedup + drop 10 least frequent structural variants per isotype
  4. dedup+variant+conf — full pipeline + confidence score filter (±2 SD per domain)

For each variant the same 80/20 stratified train/test split is used and a
multinomial logistic regression is trained.  Top-1 accuracy and per-isotype
F1 are reported.  Results saved to --outdir.

Usage
-----
    python scripts/ablation_dataset_filter.py
    python scripts/ablation_dataset_filter.py --max-samples 200000 --outdir data
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")


# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--sprinzl",         default="trnascan_GTDB_r226_v2_sprinzl.csv")
parser.add_argument("--taxonomy",        default=f"{DATA_ROOT}/trnascan_GTDB_r226_v2_final_matched_taxonomy.csv")
parser.add_argument("--outdir",          default="data")
parser.add_argument("--drop-n-variants", type=int,   default=10)
parser.add_argument("--conf-sd",         type=float, default=2.0)
parser.add_argument("--max-samples",     type=int,   default=0,
                    help="Cap per variant for speed (0 = use all)")
parser.add_argument("--test-size",       type=float, default=0.2)
parser.add_argument("--seed",            type=int,   default=42)
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
np.random.seed(args.seed)

D_INSERT   = ["17a", "20a", "20b"]
VAR_INSERT = ["47a","47b","47c","47d","47e","47f","47g",
              "47h","47i","47j","47k","47l","47m","47n","47o","47p"]
ALPHABET   = ["A", "C", "G", "T", "N"]
CHAR_IDX   = {c: i for i, c in enumerate(ALPHABET)}
MAX_LEN    = 200


def mask_anticodon(seq: str, codon: str) -> str:
    """Replace anticodon triplet with NNN in the sequence."""
    seq = seq.upper().replace("U", "T")
    codon = str(codon).upper().replace("U", "T")
    if len(codon) == 3 and codon in seq:
        pos = seq.index(codon)
        seq = seq[:pos] + "NNN" + seq[pos + 3:]
    return seq


def ohe_sequences(seqs: pd.Series) -> np.ndarray:
    """One-hot encode sequences to (N, MAX_LEN * 5) float32 array."""
    out = np.zeros((len(seqs), MAX_LEN * len(ALPHABET)), dtype=np.float32)
    for i, seq in enumerate(seqs):
        seq = seq.upper().replace("U", "T")
        for pos, ch in enumerate(seq[:MAX_LEN]):
            idx = CHAR_IDX.get(ch, CHAR_IDX["N"])
            out[i, pos * len(ALPHABET) + idx] = 1.0
    return out


def build_base(sprinzl_df: pd.DataFrame, tax: pd.DataFrame) -> pd.DataFrame:
    """Merge sprinzl + taxonomy, apply sprinzl_ok + valid isotype filter."""
    df = sprinzl_df.merge(tax, on="tRNAscanID", how="left")
    df = df.dropna(subset=["family", "domain", "conf_score", "Codon"]).copy()
    return df


def add_struct_variant(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for p in D_INSERT + VAR_INSERT:
        col = f"pos_{p}"
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace({"": "__", "nan": "__"})
    d_cols   = [f"pos_{p}" for p in D_INSERT   if f"pos_{p}" in df.columns]
    var_cols = [f"pos_{p}" for p in VAR_INSERT if f"pos_{p}" in df.columns]
    df["d_ins_count"]    = (df[d_cols]   != "__").sum(axis=1).astype(np.int8)
    df["var_arm_length"] = (df[var_cols] != "__").sum(axis=1).astype(np.int8)
    df["struct_variant"] = (
        df["sprinzl_type"].astype(str) + "_d" +
        df["d_ins_count"].astype(str)  + "_v" +
        df["var_arm_length"].astype(str)
    )
    return df


def apply_variant_filter(df: pd.DataFrame, n_drop: int) -> pd.DataFrame:
    keep = pd.Series(True, index=df.index)
    for iso, grp in df.groupby("Anticodon_predicted_isotype"):
        vc = grp["struct_variant"].value_counts()
        n_to_drop     = min(n_drop, len(vc) - 1)
        drop_variants = set(vc.tail(n_to_drop).index) if n_to_drop > 0 else set()
        if drop_variants:
            keep.loc[grp[grp["struct_variant"].isin(drop_variants)].index] = False
    return df[keep].copy()


def apply_conf_filter(df: pd.DataFrame, n_sd: float) -> pd.DataFrame:
    keep = pd.Series(False, index=df.index)
    for domain, grp in df.groupby("domain"):
        mu, sd = grp["conf_score"].mean(), grp["conf_score"].std()
        mask   = (grp["conf_score"] >= mu - n_sd * sd) & (grp["conf_score"] <= mu + n_sd * sd)
        keep.loc[grp.index[mask]] = True
    return df[keep].copy()


def train_eval(df: pd.DataFrame, label: str) -> dict:
    """Encode sequences, split, train logistic regression, return metrics."""
    print(f"\n  [{label}] n={len(df):,}  encoding sequences ...", flush=True)

    # Mask anticodon
    df = df.copy()
    df["masked_seq"] = df.apply(
        lambda r: mask_anticodon(str(r["primary_sequence"]), str(r["Codon"])), axis=1
    )

    # Optionally cap
    if args.max_samples > 0 and len(df) > args.max_samples:
        df = df.sample(args.max_samples, random_state=args.seed)
        print(f"  [{label}] Capped to {args.max_samples:,}", flush=True)

    X = ohe_sequences(df["masked_seq"])
    le = LabelEncoder()
    y  = le.fit_transform(df["Anticodon_predicted_isotype"].values)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size,
                                  random_state=args.seed)
    train_idx, test_idx = next(sss.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    print(f"  [{label}] train={len(y_train):,}  test={len(y_test):,}  fitting ...", flush=True)
    t0  = time.time()
    clf = LogisticRegression(
        max_iter=1000, solver="lbfgs",
        C=1.0, random_state=args.seed
    )
    clf.fit(X_train, y_train)
    elapsed = time.time() - t0

    preds   = clf.predict(X_test)
    acc     = accuracy_score(y_test, preds)
    report  = classification_report(y_test, preds,
                                     target_names=le.classes_, output_dict=True)
    print(f"  [{label}] accuracy={acc:.4f}  ({elapsed:.0f}s)", flush=True)

    per_class = {iso: report[iso]["f1-score"] for iso in le.classes_}
    return {
        "label":       label,
        "n_total":     len(df),
        "n_train":     len(y_train),
        "n_test":      len(y_test),
        "accuracy":    round(acc, 6),
        "fit_seconds": round(elapsed, 1),
        **{f"f1_{iso}": round(v, 6) for iso, v in per_class.items()},
    }


# ── Load source data ──────────────────────────────────────────────────────────
print("=" * 60)
print("Loading source data")
print("=" * 60)

print("  Loading taxonomy ...", flush=True)
tax = pd.read_csv(
    args.taxonomy,
    usecols=["tRNAscanID", "domain", "family", "conf_score", "Codon"],
    low_memory=False,
).drop_duplicates("tRNAscanID")
tax["conf_score"] = pd.to_numeric(tax["conf_score"], errors="coerce")

print("  Loading sprinzl CSV ...", flush=True)
sprinzl_cols = (
    ["tRNAscanID", "primary_sequence", "sprinzl_ok",
     "Anticodon_predicted_isotype", "sprinzl_type"]
    + [f"pos_{p}" for p in D_INSERT + VAR_INSERT]
)
sprinzl = pd.read_csv(args.sprinzl, usecols=sprinzl_cols, low_memory=False)
sprinzl = sprinzl[sprinzl["sprinzl_ok"].astype(float) == 1].copy()
sprinzl = sprinzl[~sprinzl["Anticodon_predicted_isotype"].isin(["Undet", "Sup", ""])].copy()
sprinzl = sprinzl.dropna(subset=["Anticodon_predicted_isotype"]).copy()
print(f"  Sprinzl base: {len(sprinzl):,}", flush=True)

# ── Build the 4 dataset variants ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Building dataset variants")
print("=" * 60)

# 1. Baseline
base = build_base(sprinzl, tax)
print(f"  1. baseline:        {len(base):,}")

# 2. Dedup (keep highest conf_score per sequence+family)
dedup = (
    base.sort_values("conf_score", ascending=False)
        .drop_duplicates(subset=["primary_sequence", "family"])
        .copy()
)
print(f"  2. dedup:           {len(dedup):,}")

# 3. Dedup + variant filter
dedup_sv = add_struct_variant(dedup)
dedup_sv = apply_variant_filter(dedup_sv, args.drop_n_variants)
print(f"  3. dedup+variant:   {len(dedup_sv):,}")

# 4. Dedup + variant + conf filter
full = apply_conf_filter(dedup_sv, args.conf_sd)
print(f"  4. dedup+var+conf:  {len(full):,}")

# ── Train and evaluate each variant ──────────────────────────────────────────
print("\n" + "=" * 60)
print("Training classifiers")
print("=" * 60)

results = []
for label, dataset in [
    ("baseline",       base),
    ("dedup",          dedup),
    ("dedup+variant",  dedup_sv),
    ("dedup+var+conf", full),
]:
    res = train_eval(dataset, label)
    results.append(res)

# ── Save and print summary ────────────────────────────────────────────────────
results_df = pd.DataFrame(results)
out_path   = os.path.join(args.outdir, "ablation_results.csv")
results_df.to_csv(out_path, index=False)

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
summary_cols = ["label", "n_total", "n_train", "n_test", "accuracy", "fit_seconds"]
print(results_df[summary_cols].to_string(index=False))
print(f"\nFull results saved to {out_path}")
