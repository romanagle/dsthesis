#!/usr/bin/env python3
"""
Comparison plots: one-hot (unmasked/masked) vs RNA-FM (unmasked/masked).

Produces:
  results/figures/fig1_accuracy_summary.png   — CV + test accuracy bar chart
  results/figures/fig2_per_class_f1.png       — per-class F1 heatmap
  results/figures/fig3_precision_recall.png   — per-class precision & recall grouped bars
  results/figures/fig4_confusion_matrices.png — 2×2 grid of test-set confusion matrices
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── colour palette ──────────────────────────────────────────────────────────
COLORS = {
    "One-hot\nUnmasked": "#4C72B0",
    "One-hot\nMasked":   "#DD8452",
    "RNA-FM\nUnmasked":  "#55A868",
    "RNA-FM\nMasked":    "#C44E52",
}
LABELS = list(COLORS.keys())

# ── helpers ──────────────────────────────────────────────────────────────────
CSV_PATH = "results/runs/isomodels_first100_with_test.csv"
AMINO_ACIDS = [
    "Ala","Arg","Asn","Asp","Cys","Gln","Glu","Gly","His",
    "Ile","Ile2","Leu","Lys","Met","Phe","Pro","SeC","Ser",
    "Thr","Trp","Tyr","Val","fMet",
]


def load_result_csv(path):
    """Return (per_class_df, cv_mean, cv_std, test_acc) from a saved results CSV."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    meta_rows = ["CV_mean", "CV_std", "test_acc"]
    per_class = df[~df["class"].isin(meta_rows)].copy()
    per_class = per_class.set_index("class").reindex(AMINO_ACIDS)
    per_class = per_class[["precision", "recall", "f1_score", "support"]].apply(
        pd.to_numeric, errors="coerce"
    )
    cv_mean  = float(df.loc[df["class"] == "CV_mean",  "precision"].values[0])
    cv_std   = float(df.loc[df["class"] == "CV_std",   "precision"].values[0])
    test_acc = float(df.loc[df["class"] == "test_acc", "precision"].values[0])
    return per_class, cv_mean, cv_std, test_acc


def run_embedding_classifier(emb_path, lbl_path):
    """
    Train/evaluate softmax on saved .npy embeddings.
    Returns (per_class_df, cv_mean, cv_std, test_acc, y_test, y_pred, classes).
    """
    X = np.load(emb_path).astype(np.float32)
    y_raw = np.load(lbl_path)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = le.classes_

    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs",
        max_iter=1000, C=1.0, random_state=42
    )
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = []
    for tr, va in skf.split(X, y):
        clf.fit(X[tr], y[tr])
        cv_scores.append(clf.score(X[va], y[va]))
    cv_mean, cv_std = np.mean(cv_scores), np.std(cv_scores)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    test_acc = clf.score(X_te, y_te)

    report = classification_report(
        y_te, y_pred, labels=range(len(classes)),
        target_names=classes, output_dict=True, zero_division=0
    )
    rows = []
    for aa in AMINO_ACIDS:
        if aa in report:
            rows.append({
                "class": aa,
                "precision": report[aa]["precision"],
                "recall":    report[aa]["recall"],
                "f1_score":  report[aa]["f1-score"],
                "support":   report[aa]["support"],
            })
        else:
            rows.append({"class": aa, "precision": 0.0, "recall": 0.0,
                         "f1_score": 0.0, "support": 0.0})
    per_class = pd.DataFrame(rows).set_index("class")
    return per_class, cv_mean, cv_std, test_acc, y_te, y_pred, classes


# ── load / compute all four conditions ──────────────────────────────────────
print("Loading results …")
pc_sf_un, cv_sf_un, std_sf_un, acc_sf_un = load_result_csv(
    "results/runs/softmax_classifier_results.csv"
)
pc_sf_ma, cv_sf_ma, std_sf_ma, acc_sf_ma = load_result_csv(
    "results/runs/softmax_classifier_masked_results.csv"
)
pc_fm_un, cv_fm_un, std_fm_un, acc_fm_un, yte_fm_un, ypred_fm_un, cls_fm_un = \
    run_embedding_classifier(
        "results/embeddings/rnafm_embeddings.npy",
        "results/embeddings/rnafm_labels.npy",
    )
pc_fm_ma, cv_fm_ma, std_fm_ma, acc_fm_ma, yte_fm_ma, ypred_fm_ma, cls_fm_ma = \
    run_embedding_classifier(
        "results/embeddings/rnafm_embeddings_masked.npy",
        "results/embeddings/rnafm_labels_masked.npy",
    )

per_class = {
    "One-hot\nUnmasked": pc_sf_un,
    "One-hot\nMasked":   pc_sf_ma,
    "RNA-FM\nUnmasked":  pc_fm_un,
    "RNA-FM\nMasked":    pc_fm_ma,
}
cv_means  = [cv_sf_un, cv_sf_ma, cv_fm_un, cv_fm_ma]
cv_stds   = [std_sf_un, std_sf_ma, std_fm_un, std_fm_ma]
test_accs = [acc_sf_un, acc_sf_ma, acc_fm_un, acc_fm_ma]

print("  CV:  ", [f"{m:.3f}±{s:.3f}" for m, s in zip(cv_means, cv_stds)])
print("  Test:", test_accs)


# ── FIG 1: accuracy summary bar chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(LABELS))
w = 0.35

bars_cv   = ax.bar(x - w/2, cv_means,  w, yerr=cv_stds, capsize=5,
                   color=[COLORS[l] for l in LABELS], alpha=0.85, label="CV accuracy")
bars_test = ax.bar(x + w/2, test_accs, w,
                   color=[COLORS[l] for l in LABELS], alpha=0.45, label="Test accuracy",
                   edgecolor=[COLORS[l] for l in LABELS], linewidth=1.5)

# value labels — CV labels sit above the error bar cap, test labels above the bar top
for bar, val, std in zip(bars_cv, cv_means, cv_stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.025,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9)
for bar, val in zip(bars_test, test_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
            f"{val:.2f}", ha="center", va="bottom", fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=11)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Classifier Accuracy: One-hot vs RNA-FM\n(unmasked vs anticodon-masked)", fontsize=13)
ax.set_ylim(0, 1.05)
ax.axhline(1/23, color="grey", linestyle="--", linewidth=0.8, label="Random (1/23)")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_accuracy_summary.png", dpi=150)
plt.close()
print("Saved fig1_accuracy_summary.png")


# ── FIG 2: per-class F1 heatmap ─────────────────────────────────────────────
# only rows with support > 0 in at least one condition
has_support = [
    aa for aa in AMINO_ACIDS
    if any(per_class[l].loc[aa, "support"] > 0 for l in LABELS)
]

f1_matrix = pd.DataFrame(
    {l: per_class[l].loc[has_support, "f1_score"] for l in LABELS}
)

fig, ax = plt.subplots(figsize=(9, len(has_support) * 0.55 + 1.5))
sns.heatmap(
    f1_matrix, annot=True, fmt=".2f", vmin=0, vmax=1,
    cmap="RdYlGn", linewidths=0.4, linecolor="white",
    ax=ax, annot_kws={"size": 9}
)
ax.set_title("Per-class F1 Score by Condition", fontsize=13, pad=12)
ax.set_xlabel("")
ax.set_ylabel("Isotype", fontsize=11)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=9, rotation=0)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_per_class_f1.png", dpi=150)
plt.close()
print("Saved fig2_per_class_f1.png")


# ── FIG 3: per-class precision & recall grouped bars ────────────────────────
# focus on classes with support > 0
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
x = np.arange(len(has_support))
n = len(LABELS)
width = 0.18
offsets = np.linspace(-(n-1)/2 * width, (n-1)/2 * width, n)

for metric, ax in zip(["precision", "recall"], axes):
    for i, label in enumerate(LABELS):
        vals = per_class[label].loc[has_support, metric].fillna(0).values
        ax.bar(x + offsets[i], vals, width,
               label=label.replace("\n", " "),
               color=list(COLORS.values())[i], alpha=0.85)
    ax.set_ylabel(metric.capitalize(), fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(1.0, color="grey", linewidth=0.6, linestyle="--")

axes[0].set_title("Per-class Precision & Recall by Condition", fontsize=13)
axes[0].legend(ncol=4, fontsize=9, loc="upper right")
axes[1].set_xticks(x)
axes[1].set_xticklabels(has_support, rotation=45, ha="right", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_precision_recall.png", dpi=150)
plt.close()
print("Saved fig3_precision_recall.png")


# ── FIG 4: confusion matrices for RNA-FM conditions ─────────────────────────
# (one-hot confusion matrix requires re-running the classifier; do RNA-FM pair)

def make_cm_data(y_te, y_pred, classes):
    """Return square confusion matrix aligned to AMINO_ACIDS."""
    idx = [i for i, c in enumerate(classes) if c in AMINO_ACIDS]
    present = [classes[i] for i in idx]
    cm = confusion_matrix(y_te, y_pred, labels=idx)
    df = pd.DataFrame(cm, index=present, columns=present)
    # reindex to full AMINO_ACIDS (zero-fill absent)
    df = df.reindex(index=AMINO_ACIDS, columns=AMINO_ACIDS, fill_value=0)
    return df


def run_onehot_classifier(masked=False):
    """Re-run one-hot classifier to capture y_test/y_pred for confusion matrix."""
    alphabet = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}
    max_len = 95

    def one_hot(seq):
        arr = np.zeros((max_len, 5), dtype=np.float32)
        for i, ch in enumerate(seq[:max_len]):
            arr[i, alphabet.get(ch.upper(), 4)] = 1.0
        return arr.flatten()

    if masked:
        mdf = pd.read_csv("results/runs/isomodels_first100_anticodon_masked.csv")
        mdf.columns = [c.strip() for c in mdf.columns]
        mdf["Anticodon_predicted_isotype"] = mdf["Anticodon_predicted_isotype"].str.strip()
        seqs   = mdf["anticodon_masked_sequence"].str.strip().values
        labels = mdf["Anticodon_predicted_isotype"].values
    else:
        df = pd.read_csv(CSV_PATH)
        df.columns = [c.strip() for c in df.columns]
        df["Anticodon_predicted_isotype"] = df["Anticodon_predicted_isotype"].str.strip()
        seqs   = df["primary_sequence"].str.strip().values
        labels = df["Anticodon_predicted_isotype"].values

    X = np.array([one_hot(s) for s in seqs])
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    classes = le.classes_

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=None
    )
    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs",
        max_iter=1000, C=1.0, random_state=42
    )
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    return y_te, y_pred, classes


print("Re-running one-hot classifiers for confusion matrices …")
yte_oh_un, ypred_oh_un, cls_oh_un = run_onehot_classifier(masked=False)
yte_oh_ma, ypred_oh_ma, cls_oh_ma = run_onehot_classifier(masked=True)

cm_data = {
    "One-hot Unmasked": make_cm_data(yte_oh_un, ypred_oh_un, cls_oh_un),
    "One-hot Masked":   make_cm_data(yte_oh_ma, ypred_oh_ma, cls_oh_ma),
    "RNA-FM Unmasked":  make_cm_data(yte_fm_un, ypred_fm_un, cls_fm_un),
    "RNA-FM Masked":    make_cm_data(yte_fm_ma, ypred_fm_ma, cls_fm_ma),
}

# only show rows/cols that appear in at least one test set
active = sorted({
    aa for aa in AMINO_ACIDS
    if any(cm.loc[aa].sum() > 0 or cm[aa].sum() > 0
           for cm in cm_data.values())
})

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, (title, cm) in zip(axes.flatten(), cm_data.items()):
    sub = cm.loc[active, active].values.astype(int)
    sns.heatmap(
        sub, annot=True, fmt="d", cmap="Blues",
        xticklabels=active, yticklabels=active,
        linewidths=0.3, linecolor="white",
        ax=ax, cbar=False, annot_kws={"size": 8}
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

plt.suptitle("Confusion Matrices — Test Set (n=20)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4_confusion_matrices.png")

print(f"\nAll figures saved to {OUT_DIR}/")
