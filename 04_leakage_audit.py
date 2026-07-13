#!/usr/bin/env python3
"""
Step 4 - Behavioural leakage audit.

A blacklist cannot anticipate PROXY leakage: a feature that is upstream of the
outcome in principle, but encodes it in practice. This script hunts for such
features behaviourally, in three stages.

  A. UNIVARIATE SCAN - how well does each feature separate the classes ALONE?
     Anything far from 0.5 with no clinical reason to predict seriousness is a
     suspect.

  B. NESTED ABLATION - train on (a) structural + reporter features only, then
     (b) + drug indicators, then (c) + reaction indicators. A model with NO
     clinical content should be near-useless. If (a) is already strong, that is
     a leak, not a finding.

  C. LEAK HUNT - leave-one-out (drop each feature, measure the AUROC loss) and
     single-feature models. The feature whose removal costs most, and which
     alone scores highest, is the culprit.

Expected: (a) reaches ~0.887 with NO clinical information, which is impossible
without a leak. The hunt isolates report origin (is_us), which alone reaches
~0.775. Reporting-channel features are therefore excluded from the clinical
model in step 5.

Input:  faers_temporal_train.parquet
Output: printed audit (Table 3 in the paper)
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"   # <-- set to your data path

NUMERIC = ["n_drugs", "n_suspect", "n_concomitant", "n_reactions",
           "age_years", "weight_kg"]
CATEGORICAL = ["sex", "qualification", "reporttype"]
CHANNEL = ["is_us"]
SEED = 42


def build(numeric, categorical, binary):
    parts = []
    if numeric:
        parts.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler())]), numeric))
    if categorical:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))
    if binary:
        parts.append(("bin", "passthrough", binary))
    return Pipeline([("pre", ColumnTransformer(parts)),
                     ("lr", LogisticRegression(max_iter=2000,
                                               class_weight="balanced"))])


def score(df, y, feats, tr, te):
    num = [f for f in feats if f in NUMERIC]
    cat = [f for f in feats if f in CATEGORICAL]
    binr = [f for f in feats if f not in NUMERIC + CATEGORICAL]
    clf = build(num, cat, binr)
    clf.fit(df.loc[tr, feats], y.loc[tr])
    p = clf.predict_proba(df.loc[te, feats])[:, 1]
    return roc_auc_score(y.loc[te], p), average_precision_score(y.loc[te], p)


def main():
    df = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_train.parquet"))
    y = df["label_serious"]
    drug = [c for c in df.columns if c.startswith("drug::")]
    pt = [c for c in df.columns if c.startswith("pt::")]
    tr, te = train_test_split(df.index, test_size=0.25, random_state=SEED,
                              stratify=y)
    print(f"rows {len(df):,} | serious {y.mean():.1%} | "
          f"{len(pt)} reaction terms | {len(drug)} drug terms\n")

    # ---- A. univariate scan ------------------------------------------------
    print("=== A. Strongest univariate separators (|AUC - 0.5|) ===")
    rows = []
    for col in NUMERIC + CATEGORICAL + CHANNEL:
        s = df[col]
        if s.dtype == object:
            for val in s.value_counts().head(4).index:
                x = (s == val).astype(int).values
                if 0 < x.sum() < len(x):
                    rows.append((f"{col}={val}", roc_auc_score(y, x)))
        else:
            x = s.fillna(s.median()).values
            if np.nanstd(x) > 0:
                rows.append((col, roc_auc_score(y, x)))
    for name, auc in sorted(rows, key=lambda r: abs(r[1] - 0.5), reverse=True)[:8]:
        flag = "   <-- clinically implausible" if abs(auc - 0.5) > 0.2 else ""
        print(f"  AUC {auc:.3f}   {name}{flag}")

    # ---- B. nested ablation ------------------------------------------------
    print("\n=== B. Nested ablation (is the signal clinical, or a leak?) ===")
    configs = [
        ("(a) structural + reporter only", NUMERIC + CATEGORICAL + CHANNEL),
        ("(b) + drug indicators", NUMERIC + CATEGORICAL + CHANNEL + drug),
        ("(c) + reaction indicators", NUMERIC + CATEGORICAL + CHANNEL + drug + pt),
    ]
    print(f"{'model':<34}{'AUROC':>8}{'AUPRC':>8}")
    print("-" * 50)
    prev = None
    for name, feats in configs:
        auroc, auprc = score(df, y, feats, tr, te)
        delta = f"  (+{auroc - prev:.3f})" if prev is not None else ""
        print(f"{name:<34}{auroc:>8.3f}{auprc:>8.3f}{delta}")
        prev = auroc
    print("\n  A model with NO clinical content should be near-useless.")
    print("  If (a) is already strong, that is a proxy leak - hunt it below.")

    # ---- C. leak hunt ------------------------------------------------------
    struct = NUMERIC + CATEGORICAL + CHANNEL
    full, _ = score(df, y, struct, tr, te)
    print(f"\n=== C. Leak hunt (structural-only baseline AUROC {full:.3f}) ===")

    print("\n  Leave-one-out (larger loss = more leaky):")
    losses = []
    for f in struct:
        auc, _ = score(df, y, [x for x in struct if x != f], tr, te)
        losses.append((f, auc, full - auc))
    for f, auc, loss in sorted(losses, key=lambda r: r[2], reverse=True)[:5]:
        flag = "   <-- LEAK" if loss >= 0.05 else ""
        print(f"    drop {f:<16} -> {auc:.3f}   (loss {loss:+.3f}){flag}")

    print("\n  Single feature alone:")
    singles = [(f, score(df, y, [f], tr, te)[0]) for f in struct]
    for f, auc in sorted(singles, key=lambda r: r[1], reverse=True)[:5]:
        flag = "   <-- CULPRIT" if auc >= 0.70 else ""
        print(f"    {f:<16} alone -> {auc:.3f}{flag}")

    print("\nReporting-channel features are excluded from the clinical model.")
    print("Next: 05_train_evaluate.py")


if __name__ == "__main__":
    main()
