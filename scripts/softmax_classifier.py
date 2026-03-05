#!/usr/bin/env python3
"""
Softmax classifier for tRNA isotype prediction.

Input:  primary_sequence  (nucleotide string, length 71–95)
Target: Anticodon_predicted_isotype  (23 amino-acid isotype classes)

Approach
--------
1. One-hot encode each sequence over alphabet {A, C, G, T/U, N} padded to
   max_len, producing a flat feature vector of size max_len * 5.
2. Fit a multinomial logistic regression (softmax) classifier.
3. Report accuracy and a full classification report on a held-out test split.
   Because n=100, we also run stratified k-fold CV to get a stable estimate.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support


# ---------------------------------------------------------------------------
# Sequence encoding
# ---------------------------------------------------------------------------

ALPHABET = ["A", "C", "G", "T", "N"]  # T covers T and U
CHAR_TO_IDX = {ch: i for i, ch in enumerate(ALPHABET)}


def one_hot_encode(seq: str, max_len: int) -> np.ndarray:
    """Return a (max_len * len(ALPHABET),) float32 one-hot vector."""
    seq = seq.upper().replace("U", "T")          # normalise RNA → DNA
    vec = np.zeros((max_len, len(ALPHABET)), dtype=np.float32)
    for pos, ch in enumerate(seq[:max_len]):
        idx = CHAR_TO_IDX.get(ch, CHAR_TO_IDX["N"])
        vec[pos, idx] = 1.0
    return vec.ravel()


def encode_sequences(sequences: pd.Series, max_len: int) -> np.ndarray:
    return np.vstack([one_hot_encode(s, max_len) for s in sequences])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data_path = "results/runs/isomodels_first100_anticodon_masked.csv"
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]

    seqs   = df["anticodon_masked_sequence"].str.strip()
    labels = df["Anticodon_predicted_isotype"].str.strip()

    max_len = seqs.str.len().max()
    print(f"Dataset: {len(df)} samples, {labels.nunique()} classes, max_len={max_len}")

    # Encode sequences and labels
    X = encode_sequences(seqs, max_len)            # shape (100, max_len*5)
    le = LabelEncoder()
    y = le.fit_transform(labels)

    print(f"Feature matrix: {X.shape}")
    print(f"Classes: {list(le.classes_)}\n")

    # ------------------------------------------------------------------
    # Softmax classifier  (multinomial logistic regression)
    # ------------------------------------------------------------------
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        C=1.0,              # inverse regularisation strength
        random_state=42,
    )

    # Stratified k-fold CV for a stable accuracy estimate on n=100.
    # Use k=3 because some classes have only 1 sample.
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    print(f"3-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Per-fold: {np.round(cv_scores, 3)}\n")

    # Single 80/20 train-test split for the classification report.
    # Cannot stratify because Phe and SeC have only 1 sample each.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=None, random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"Hold-out test accuracy: {accuracy_score(y_test, y_pred):.3f}  "
          f"({len(y_test)} samples)\n")
    print("Classification report:")
    print(classification_report(
        y_test, y_pred,
        labels=list(range(len(le.classes_))),
        target_names=le.classes_,
        zero_division=0,
    ))

    # Show learned weight matrix shape for reference
    print(f"Weight matrix shape: {clf.coef_.shape}  "
          f"({clf.coef_.shape[0]} classes × {clf.coef_.shape[1]} features)")

    # ------------------------------------------------------------------
    # Save results to CSV
    # ------------------------------------------------------------------
    out_path = "results/runs/softmax_classifier_masked_results.csv"

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred,
        labels=list(range(len(le.classes_))),
        zero_division=0,
    )
    report_df = pd.DataFrame({
        "class":     le.classes_,
        "precision": precision.round(3),
        "recall":    recall.round(3),
        "f1_score":  f1.round(3),
        "support":   support,
    })

    # Summary rows
    test_acc = accuracy_score(y_test, y_pred)
    summary_rows = pd.DataFrame([
        {"class": "CV_mean",  "precision": round(cv_scores.mean(), 3),
         "recall": "",        "f1_score": "",  "support": ""},
        {"class": "CV_std",   "precision": round(cv_scores.std(), 3),
         "recall": "",        "f1_score": "",  "support": ""},
        {"class": "test_acc", "precision": round(test_acc, 3),
         "recall": "",        "f1_score": "",  "support": len(y_test)},
    ])

    pd.concat([report_df, summary_rows], ignore_index=True).to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
