#!/usr/bin/env python3
"""
Step 5 - Train the clinical model and evaluate it on the temporal hold-out.

Trains two models on the SAME temporal split, so the only thing that differs is
the feature set:

  FULL      - includes reporting-channel features (is_us, reporter qualification,
              report type). Reported ONLY to quantify what the leak was worth.
  CLINICAL  - channel features REMOVED. This is the model reported in the paper.

Then reports:
  * discrimination (AUROC, AUPRC) overall and by region
  * calibration (Brier score, reliability deciles)
  * the recall / alert-rate trade-off across thresholds, and the operating point
  * the equity consequence of applying a single global threshold

Expected: CLINICAL AUROC 0.896, Brier 0.139. Removing the leaking channel
features costs +0.054 AUROC (FULL 0.950) - a small price for a model that is
not predicting from reporting geography.

Input:  faers_temporal_{train,test}.parquet
Output: printed results (Results 3.2 and 3.3 in the paper)
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"   # <-- set to your data path

NUMERIC = ["n_drugs", "n_suspect", "n_concomitant", "n_reactions",
           "age_years", "weight_kg"]
SEED = 42


def build(numeric, categorical, binary):
    parts = []
    if numeric:
        parts.append(("num", SimpleImputer(strategy="median"), numeric))
    if categorical:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    if binary:
        parts.append(("bin", "passthrough", binary))
    # Hyperparameters fixed a priori; no tuning was performed.
    return Pipeline([("pre", ColumnTransformer(parts)), ("xgb", XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
        tree_method="hist", n_jobs=-1, random_state=SEED))])


def evaluate(tr, te, numeric, categorical, binary):
    feats = numeric + categorical + binary
    clf = build(numeric, categorical, binary)
    clf.fit(tr[feats], tr["label_serious"])
    p = clf.predict_proba(te[feats])[:, 1]
    y = te["label_serious"].values
    us = (te["is_us"] == 1).values
    return {"auroc": roc_auc_score(y, p),
            "auprc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p),
            "auroc_us": roc_auc_score(y[us], p[us]),
            "auroc_nonus": roc_auc_score(y[~us], p[~us]),
            "p": p, "y": y, "us": us}


def recall_at(y, p, th):
    pred = p >= th
    tp = ((pred) & (y == 1)).sum()
    fn = ((~pred) & (y == 1)).sum()
    return tp / (tp + fn) if (tp + fn) else np.nan


def alert_at(y, p, th):
    return (p >= th).mean()


def threshold_for(y, p, target):
    for th in np.round(np.arange(0.99, 0.0, -0.01), 2):
        if recall_at(y, p, th) >= target:
            return float(th)
    return 0.0


def main():
    tr = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_train.parquet"))
    te = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_test.parquet"))
    drug = [c for c in tr.columns if c.startswith("drug::")]
    pt = [c for c in tr.columns if c.startswith("pt::")]

    full = evaluate(tr, te, NUMERIC, ["sex", "qualification", "reporttype"],
                    ["is_us"] + drug + pt)
    clin = evaluate(tr, te, NUMERIC, ["sex"], drug + pt)

    print("=== Temporal validation (train 2025 Q1-Q3 -> test 2025 Q4) ===")
    print(f"{'':<10}{'AUROC':>8}{'AUPRC':>8}{'Brier':>8}{'US':>8}{'non-US':>9}")
    for name, r in [("FULL", full), ("CLINICAL", clin)]:
        print(f"{name:<10}{r['auroc']:>8.3f}{r['auprc']:>8.3f}{r['brier']:>8.3f}"
              f"{r['auroc_us']:>8.3f}{r['auroc_nonus']:>9.3f}")
    print(f"\ncost of removing the leaking channel features: "
          f"{full['auroc'] - clin['auroc']:+.3f} AUROC")
    print("The CLINICAL model is the one reported in the paper.\n")

    y, p, us = clin["y"], clin["p"], clin["us"]

    print("=== Calibration (CLINICAL model, by decile) ===")
    order = np.argsort(p)
    for i, idx in enumerate(np.array_split(order, 10), 1):
        print(f"  decile {i:>2}: predicted {p[idx].mean():.2f}   "
              f"observed {y[idx].mean():.2f}   (n={len(idx):,})")

    print("\n=== Recall / alert trade-off ===")
    print(f"{'threshold':>10}{'recall':>9}{'alert rate':>12}")
    for th in np.round(np.arange(0.1, 0.95, 0.1), 2):
        print(f"{th:>10}{recall_at(y, p, th):>9.2f}{alert_at(y, p, th):>12.1%}")

    gth = threshold_for(y, p, 0.90)
    print(f"\noperating point for 90% recall: threshold {gth:.2f} "
          f"(alert rate {alert_at(y, p, gth):.1%})")

    print("\n=== Equity: one global threshold, split by region ===")
    print(f"{'group':<10}{'n':>8}{'base':>8}{'recall':>9}{'alert':>9}{'own thr':>9}")
    for name, m in [("overall", np.ones(len(y), bool)), ("US", us), ("non-US", ~us)]:
        own = threshold_for(y[m], p[m], 0.90)
        print(f"{name:<10}{int(m.sum()):>8,}{y[m].mean():>8.1%}"
              f"{recall_at(y[m], p[m], gth):>9.2f}"
              f"{alert_at(y[m], p[m], gth):>9.1%}{own:>9.2f}")

    print("\nRanking is comparable across regions; recall at a single threshold")
    print("is not. The remedy is group-aware thresholds, not a different model.")
    print("\nNext: 06_subgroups.py")


if __name__ == "__main__":
    main()
