#!/usr/bin/env python3
"""
Run the softmax classifier on all embedding conditions and print a comparison.

Conditions
----------
  rnafm_unmasked   : RNA-FM, original sequences
  rnafm_masked     : RNA-FM, anticodon-masked sequences
  rinalmo_unmasked : RiNALMo, original sequences
  rinalmo_masked   : RiNALMo, anticodon-masked sequences

Output: results/runs/softmax_embeddings_comparison.csv
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

OUT_DIR  = "results/embeddings"
OUT_CSV  = "results/runs/softmax_embeddings_comparison.csv"

CONDITIONS = [
    ("rnafm_unmasked",   "rnafm_embeddings.npy",         "rnafm_labels.npy"),
    ("rnafm_masked",     "rnafm_embeddings_masked.npy",   "rnafm_labels_masked.npy"),
    ("rinalmo_unmasked", "rinalmo_embeddings.npy",        "rinalmo_labels.npy"),
    ("rinalmo_masked",   "rinalmo_embeddings_masked.npy", "rinalmo_labels_masked.npy"),
]


def run_condition(name, emb_file, lbl_file):
    import os
    emb_path = os.path.join(OUT_DIR, emb_file)
    lbl_path = os.path.join(OUT_DIR, lbl_file)
    if not os.path.exists(emb_path):
        print(f"  [{name}] SKIPPED — {emb_file} not found")
        return None

    X      = np.load(emb_path)
    labels = np.load(lbl_path)

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    clf = LogisticRegression(
        multi_class="multinomial", solver="lbfgs",
        max_iter=1000, C=1.0, random_state=42,
    )

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(clf, X, y, cv=skf, scoring="accuracy")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=None, random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"  [{name}]  CV={cv_scores.mean():.3f}±{cv_scores.std():.3f}  "
          f"test={test_acc:.3f}  feat={X.shape[1]}")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred,
        labels=list(range(len(le.classes_))),
        zero_division=0,
    )
    rows = []
    for cls, p, r, f, s in zip(le.classes_, precision, recall, f1, support):
        rows.append({"condition": name, "class": cls,
                     "precision": round(p, 3), "recall": round(r, 3),
                     "f1_score": round(f, 3), "support": s})
    rows.append({"condition": name, "class": "CV_mean",
                 "precision": round(cv_scores.mean(), 3), "recall": "",
                 "f1_score": "", "support": ""})
    rows.append({"condition": name, "class": "test_acc",
                 "precision": round(test_acc, 3), "recall": "",
                 "f1_score": "", "support": len(y_test)})
    return rows


def main():
    print("Softmax classifier on embeddings\n")
    all_rows = []
    for name, emb_file, lbl_file in CONDITIONS:
        result = run_condition(name, emb_file, lbl_file)
        if result:
            all_rows.extend(result)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nFull results saved to {OUT_CSV}")

    # Summary table
    print("\nSummary:")
    print(f"{'Condition':<22} {'CV acc':>8} {'Test acc':>10}")
    print("-" * 42)
    for name, emb_file, lbl_file in CONDITIONS:
        subset = df[(df["condition"] == name)]
        cv_row   = subset[subset["class"] == "CV_mean"]
        test_row = subset[subset["class"] == "test_acc"]
        if cv_row.empty:
            continue
        cv_val   = cv_row["precision"].values[0]
        test_val = test_row["precision"].values[0]
        print(f"{name:<22} {cv_val:>8.3f} {test_val:>10.3f}")


if __name__ == "__main__":
    main()
