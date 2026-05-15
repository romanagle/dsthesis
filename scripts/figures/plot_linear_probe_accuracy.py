#!/usr/bin/env python3
"""
plot_linear_probe_accuracy.py
-----------------------------
Trains a logistic regression (linear probe) on mean-pooled embeddings from
four conditions (RNA-FM unmasked/masked, RiNALMo unmasked/masked) and plots
overall test accuracy as a bar chart.

Usage:
    python scripts/figures/plot_linear_probe_accuracy.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")
EMB_ROOT  = os.path.join(DATA_ROOT, "embeddings")
OUT_DIR   = os.path.join(DATA_ROOT, "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Each tuple: (display label, emb subdir, colour, label_csv, seq_col_in_csv)
CONDITIONS = [
    ("RNA-FM unmasked",
     "rnafm_unmasked", "#4C9BE8",
     "archive/trnascan_GTDB_r226_taxonomy.csv",
     "primary_sequence"),
    ("RNA-FM masked",
     "rnafm_masked", "#E8694C",
     "trnascan_GTDB_r226_v2_final_matched_taxonomy.csv",
     "anticodon_masked_sequence"),
    ("RiNALMo unmasked",
     "rinalmo_unmasked", "#55A868",
     "data/training_dataset.csv",
     "primary_sequence"),
    ("RiNALMo masked",
     "rinalmo_masked", "#E8B84C",
     "trnascan_GTDB_r226_v2_final_matched_taxonomy.csv",
     "anticodon_masked_sequence"),
]


def build_label_map(csv_rel: str, seq_col: str) -> dict:
    path = os.path.join(DATA_ROOT, csv_rel)
    df = pd.read_csv(path, usecols=[seq_col, "Anticodon_predicted_isotype"])
    return dict(zip(df[seq_col], df["Anticodon_predicted_isotype"]))


def run_probe(emb_dir: str, label_map: dict) -> float:
    """Load embeddings, match labels, train logistic regression, return test acc."""
    emb_path  = os.path.join(EMB_ROOT, emb_dir, "embeddings.npy")
    seq_path  = os.path.join(EMB_ROOT, emb_dir, "unique_sequences.npy")

    X    = np.load(emb_path, mmap_mode="r")
    seqs = np.load(seq_path, allow_pickle=True)

    labels = np.array([label_map[s] for s in seqs])

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = LogisticRegression(
        multi_class="multinomial", solver="saga",
        max_iter=300, C=1.0, random_state=42, n_jobs=-1
    )
    clf.fit(X_tr, y_tr)
    acc = clf.score(X_te, y_te)
    print(f"  {emb_dir:<22}  test acc = {acc:.4f}")
    return acc


# ── Run probes ────────────────────────────────────────────────────────────────
print("\nRunning linear probes …")
accs   = []
labels = []
colors = []

for label, emb_dir, color, csv_rel, seq_col in CONDITIONS:
    print(f"\n[{label}]  loading labels …")
    label_map = build_label_map(csv_rel, seq_col)
    acc = run_probe(emb_dir, label_map)
    accs.append(acc)
    labels.append(label)
    colors.append(color)


# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.5))

x = np.arange(len(labels))
bars = ax.bar(x, accs, color=colors, width=0.55, alpha=0.88, edgecolor="white",
              linewidth=1.2)

for bar, acc in zip(bars, accs):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{acc:.1%}",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Overall accuracy", fontsize=13)
ax.set_ylim(0, 1.08)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

out_path = os.path.join(OUT_DIR, "linear_probe_accuracy_bar.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved -> {out_path}")
