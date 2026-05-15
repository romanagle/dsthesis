#!/usr/bin/env python3
"""
plot_softmax_phylo_accuracy.py
------------------------------
Per-taxon softmax accuracy at Domain/Phylum/Class/Order for
RiNALMo-masked vs RNA-FM-masked embeddings.

Caches test predictions to avoid retraining on subsequent runs:
  {DATA_ROOT}/figures/classification/cache/{emb_dir}_test_preds.parquet

Delete cache files to retrain from scratch.

Usage:
    python scripts/figures/plot_softmax_phylo_accuracy.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_ROOT = os.getenv("TRNA_DATA_ROOT", "/data/roma/dsthesis")
EMB_ROOT  = os.path.join(DATA_ROOT, "embeddings")
OUT_DIR   = os.path.join(DATA_ROOT, "figures", "classification")
CACHE_DIR = os.path.join(OUT_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

CSV_PATH     = os.path.join(DATA_ROOT, "data", "trnascan_GTDB_r226_dedup_taxonomy.csv")
N_SAMPLE     = 500_000   # sequences per model; None = use all
RANDOM_STATE = 42
MIN_TEST     = 10        # min test seqs required per taxon
PHY_LEVELS   = ["domain", "phylum", "class", "order"]
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = [
    ("RiNALMo", "rinalmo_masked", "#5B9BD5"),
    ("RNA-FM",  "rnafm_masked",   "#E8694C"),
]


# ── Fast GPU linear classifier ────────────────────────────────────────────────

def train_linear(X_tr: np.ndarray, y_tr: np.ndarray, n_classes: int,
                 epochs: int = 25, lr: float = 3e-3, batch: int = 4096) -> nn.Linear:
    in_dim = X_tr.shape[1]
    head   = nn.Linear(in_dim, n_classes).to(DEVICE)
    opt    = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit   = nn.CrossEntropyLoss()

    Xt = torch.from_numpy(X_tr).float().to(DEVICE)
    yt = torch.from_numpy(y_tr).long().to(DEVICE)
    ds = TensorDataset(Xt, yt)
    dl = DataLoader(ds, batch_size=batch, shuffle=True)

    head.train()
    for ep in range(epochs):
        total_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(head(xb), yb)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()
        avg = total_loss / len(dl)
        print(f"    epoch {ep+1:2d}/{epochs}  loss={avg:.4f}", flush=True)

    return head


@torch.no_grad()
def predict(head: nn.Linear, X_te: np.ndarray, batch: int = 8192) -> np.ndarray:
    head.eval()
    preds = []
    Xt = torch.from_numpy(X_te).float()
    for i in range(0, len(Xt), batch):
        xb = Xt[i:i+batch].to(DEVICE)
        preds.append(head(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ── Per-model pipeline ────────────────────────────────────────────────────────

def train_and_cache(label: str, emb_dir: str) -> pd.DataFrame:
    cache = os.path.join(CACHE_DIR, f"{emb_dir}_test_preds.csv.gz")
    if os.path.exists(cache):
        print(f"[{label}] loading cached predictions …")
        return pd.read_csv(cache)

    print(f"\n[{label}] running classifier pipeline …")

    print("  loading CSV …")
    df = pd.read_csv(
        CSV_PATH,
        usecols=["anticodon_masked_sequence", "Anticodon_predicted_isotype",
                 "domain", "phylum", "class", "order"],
        low_memory=False,
    )
    df = df.dropna(subset=["anticodon_masked_sequence", "Anticodon_predicted_isotype"])

    print("  loading seq_to_idx …")
    seq_to_idx = np.load(
        os.path.join(EMB_ROOT, emb_dir, "seq_to_idx.npy"), allow_pickle=True
    ).item()

    df = df[df["anticodon_masked_sequence"].isin(seq_to_idx)].copy()
    print(f"  {len(df):,} sequences matched")

    if N_SAMPLE and len(df) > N_SAMPLE:
        df = df.sample(n=N_SAMPLE, random_state=RANDOM_STATE).reset_index(drop=True)
        print(f"  sampled to {len(df):,}")

    emb_full = np.load(os.path.join(EMB_ROOT, emb_dir, "embeddings.npy"), mmap_mode="r")
    emb_idx  = np.array([seq_to_idx[s] for s in df["anticodon_masked_sequence"]])

    le  = LabelEncoder()
    y   = le.fit_transform(df["Anticodon_predicted_isotype"].values)
    row = np.arange(len(df))

    tr_i, te_i, y_tr, y_te, r_tr, r_te = train_test_split(
        emb_idx, y, row, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"  loading {len(tr_i):,} train embeddings …")
    X_tr = emb_full[tr_i].astype(np.float32)
    print(f"  loading {len(te_i):,} test embeddings …")
    X_te = emb_full[te_i].astype(np.float32)

    n_classes = int(y.max() + 1)
    print(f"  training on {DEVICE}  (n_classes={n_classes}) …")
    head = train_linear(X_tr, y_tr, n_classes)
    y_pred = predict(head, X_te)

    acc = (y_pred == y_te).mean()
    print(f"  test accuracy: {acc:.4f}")

    test_df = df.iloc[r_te].copy().reset_index(drop=True)
    test_df["correct"] = (y_pred == y_te)

    test_df.to_csv(cache, index=False)
    print(f"  cached → {cache}")
    return test_df


# ── Per-taxon accuracy ────────────────────────────────────────────────────────

def taxon_accs(df: pd.DataFrame, level: str) -> np.ndarray:
    return np.array([
        grp["correct"].mean()
        for _, grp in df.groupby(level)
        if len(grp) >= MIN_TEST
    ])


# ── Run ───────────────────────────────────────────────────────────────────────

results = {
    label: {"df": train_and_cache(label, emb_dir), "color": color}
    for label, emb_dir, color in MODELS
}


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, len(PHY_LEVELS), figsize=(14, 5))

for ax, level in zip(axes, PHY_LEVELS):
    plot_data, n_taxa_list = [], []

    for label, info in results.items():
        accs = taxon_accs(info["df"], level)
        plot_data.append(accs)
        n_taxa_list.append(len(accs))

    bp = ax.boxplot(
        plot_data,
        patch_artist=True,
        widths=0.55,
        medianprops=dict(color="black", linewidth=2.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker=".", markersize=3, alpha=0.4),
    )
    for patch, (_, info) in zip(bp["boxes"], results.items()):
        patch.set_facecolor(info["color"])
        patch.set_alpha(0.75)
    for flier, (_, info) in zip(bp["fliers"], results.items()):
        flier.set_markerfacecolor(info["color"])
        flier.set_markeredgecolor(info["color"])

    # Median annotations ABOVE the upper cap (not on the box)
    upper_caps = bp["caps"][1::2]
    for i, (accs, cap) in enumerate(zip(plot_data, upper_caps)):
        med   = float(np.median(accs))
        cap_y = float(cap.get_ydata()[0])
        ax.text(
            i + 1, cap_y + 0.020,
            f"{med:.3f}",
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold",
        )

    ax.set_xticks([1, 2])
    ax.set_xticklabels([lbl for lbl, _, _ in MODELS], rotation=0, fontsize=10)
    ax.set_title(level.capitalize(), fontsize=12, pad=6)
    ax.set_xlabel(f"n = {n_taxa_list[0]}", fontsize=8.5, color="#555555", labelpad=4)

    ax.set_ylim(0, 1.20)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if ax is axes[0]:
        ax.set_ylabel("Per-taxon accuracy", fontsize=11)

plt.tight_layout()

out_png = os.path.join(OUT_DIR, "softmax_phylo_accuracy.png")
out_svg = os.path.join(OUT_DIR, "softmax_phylo_accuracy.svg")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
print(f"\nSaved → {out_png}, {out_svg}")
