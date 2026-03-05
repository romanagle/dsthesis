#!/usr/bin/env python3
"""
Softmax classifier on RNA-FM embeddings.

Uses the pre-computed 640-dim mean-pooled RNA-FM embeddings
(results/embeddings/rnafm_embeddings.npy) as features instead of
one-hot encoded sequences, then runs the same evaluation protocol
as softmax_classifier.py for a direct comparison.

Output: results/runs/softmax_rnafm_results.csv
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

EMB_PATH   = "results/embeddings/rnafm_embeddings.npy"
LBL_PATH   = "results/embeddings/rnafm_labels.npy"
OUT_PATH   = "results/runs/softmax_rnafm_results.csv"


def main():
    X      = np.load(EMB_PATH)          # (100, 640)
    labels = np.load(LBL_PATH)          # (100,)  str

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    print(f"Features: {X.shape}  |  Classes: {len(le.classes_)}")
    print(f"Classes: {list(le.classes_)}\n")

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=1000,
        C=1.0,
        random_state=42,
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")
    print(f"3-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"  Per-fold: {np.round(cv_scores, 3)}\n")

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

    # Save CSV
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
    test_acc = accuracy_score(y_test, y_pred)
    summary_rows = pd.DataFrame([
        {"class": "CV_mean",  "precision": round(cv_scores.mean(), 3),
         "recall": "",        "f1_score": "",  "support": ""},
        {"class": "CV_std",   "precision": round(cv_scores.std(), 3),
         "recall": "",        "f1_score": "",  "support": ""},
        {"class": "test_acc", "precision": round(test_acc, 3),
         "recall": "",        "f1_score": "",  "support": len(y_test)},
    ])
    pd.concat([report_df, summary_rows], ignore_index=True).to_csv(OUT_PATH, index=False)
    print(f"Results saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
