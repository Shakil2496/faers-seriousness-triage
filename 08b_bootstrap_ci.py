#!/usr/bin/env python3
"""
Step 8b - Bootstrap confidence intervals for the headline results.

Produces the confidence intervals reported in the abstract and Table 5:

  1. The temporal AUROC of the clinical model, with a stratified bootstrap CI.

  2. For each panel drug: the classical AUROC, the LLM AUROC, and - the number
     that actually matters - the GAP between them, with a PAIRED bootstrap CI.

Why paired. Both arms score the SAME cases. Bootstrapping each arm separately
and comparing the two intervals would ignore that pairing and inflate the
variance, making a real difference look uncertain. The paired procedure instead
resamples the cases once per iteration and scores BOTH arms on that same
resample, recording the difference. This is the correct test for a head-to-head
comparison, and it is what the paper reports.

A gap whose 95% CI excludes zero is statistically significant. Expected: five of
the six drugs are significant. The exception is seladelpar (n=187, the smallest
panel), where the interval marginally includes zero - reported honestly as
limited power rather than evidence of no effect.

Input:  faers_temporal_{train,test}.parquet, <slug>_suspect.json,
        llm_arm_full/<slug>_llm.json
Output: printed CIs (Table 5 in the paper)
Runtime: a few minutes at N_BOOT=2000
"""

import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"   # <-- set to your data path
LLM_DIR = os.path.join(DATA_DIR, "llm_arm_full")

N_BOOT = 2000
SEED = 42
RNG = np.random.default_rng(SEED)

NUMERIC = ["n_drugs", "n_suspect", "n_concomitant", "n_reactions",
           "age_years", "weight_kg"]
AGE_UNIT = {"800": 10, "801": 1, "802": 1 / 12, "803": 1 / 52,
            "804": 1 / 365, "805": 1 / 8760}

PANEL = [("suzetrigine (Journavx)", "journavx"),
         ("elafibranor (Iqirvo)", "iqirvo"),
         ("xanomeline-trospium (Cobenfy)", "cobenfy"),
         ("resmetirom (Rezdiffra)", "rezdiffra"),
         ("seladelpar (Livdelzi)", "livdelzi"),
         ("sotatercept (Winrevair)", "sotatercept")]


def boot_auroc(y, p, n_boot=N_BOOT):
    """Stratified bootstrap CI for a single AUROC."""
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    stats = []
    for _ in range(n_boot):
        i = np.concatenate([RNG.choice(pos, len(pos), replace=True),
                            RNG.choice(neg, len(neg), replace=True)])
        stats.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return roc_auc_score(y, p), lo, hi


def boot_gap(y, p_classical, p_llm, n_boot=N_BOOT):
    """PAIRED bootstrap CI for the difference (LLM - classical) on identical cases."""
    y = np.asarray(y)
    p_a, p_b = np.asarray(p_classical), np.asarray(p_llm)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    diffs = []
    for _ in range(n_boot):
        i = np.concatenate([RNG.choice(pos, len(pos), replace=True),
                            RNG.choice(neg, len(neg), replace=True)])
        diffs.append(roc_auc_score(y[i], p_b[i]) - roc_auc_score(y[i], p_a[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    gap = roc_auc_score(y, p_b) - roc_auc_score(y, p_a)
    return gap, lo, hi


def age_years(patient):
    a, u = patient.get("patientonsetage"), patient.get("patientonsetageunit")
    try:
        return float(a) * AGE_UNIT.get(u, 1) if a else None
    except (TypeError, ValueError):
        return None


def featurize(reports, top_pts, top_ings):
    """Same feature construction as the training matrix, applied to panel cases."""
    rows = []
    for rep in reports:
        p = rep.get("patient") or {}
        drugs = p.get("drug", []) or []
        reactions = p.get("reaction", []) or []
        roles = Counter(d.get("drugcharacterization") for d in drugs)
        w = p.get("patientweight")
        try:
            w = float(w) if w else None
        except (TypeError, ValueError):
            w = None
        row = {"n_drugs": len(drugs), "n_suspect": roles.get("1", 0),
               "n_concomitant": roles.get("2", 0), "n_reactions": len(reactions),
               "age_years": age_years(p), "weight_kg": w,
               "sex": p.get("patientsex") or "unk"}
        rx = {(r.get("reactionmeddrapt") or "").strip().lower()
              for r in reactions}
        ing = {((d.get("activesubstance") or {}).get("activesubstancename") or "")
               .strip().lower()
               for d in drugs if d.get("drugcharacterization") == "1"}
        for t in top_pts:
            row[f"pt::{t}"] = 1 if t in rx else 0
        for i in top_ings:
            row[f"drug::{i}"] = 1 if i in ing else 0
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    tr = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_train.parquet"))
    te = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_test.parquet"))
    top_pts = [c[4:] for c in tr.columns if c.startswith("pt::")]
    top_ings = [c[6:] for c in tr.columns if c.startswith("drug::")]
    binary = [c for c in tr.columns if "::" in c]
    feats = NUMERIC + ["sex"] + binary

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["sex"]),
        ("bin", "passthrough", binary)])
    clf = Pipeline([("pre", pre), ("xgb", XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
        tree_method="hist", n_jobs=-1, random_state=SEED))])
    clf.fit(tr[feats], tr["label_serious"])

    # ---- 1. temporal headline -------------------------------------------------
    y_te = te["label_serious"].values
    p_te = clf.predict_proba(te[feats])[:, 1]
    auc, lo, hi = boot_auroc(y_te, p_te)
    print("=" * 78)
    print("1. TEMPORAL VALIDATION (clinical model, held-out Q4)")
    print("=" * 78)
    print(f"   AUROC {auc:.3f}  (95% CI {lo:.3f}-{hi:.3f})   n={len(y_te):,}\n")

    # ---- 2. per-drug classical vs LLM ----------------------------------------
    print("=" * 78)
    print("2. PER-DRUG: classical vs LLM (paired bootstrap on identical cases)")
    print("=" * 78)
    print(f"{'drug':<30}{'classical (95% CI)':>24}{'LLM (95% CI)':>24}"
          f"{'gap (95% CI)':>26}")
    print("-" * 104)

    gaps, ns, cls_aucs, llm_aucs = [], [], [], []
    for label, slug in PANEL:
        with open(os.path.join(LLM_DIR, f"{slug}_llm.json"), encoding="utf-8") as f:
            llm = json.load(f)
        with open(os.path.join(DATA_DIR, f"{slug}_suspect.json"),
                  encoding="utf-8") as f:
            by_id = {r.get("safetyreportid"): r for r in json.load(f)}

        reports, y, p_llm = [], [], []
        for item in llm:
            rid = item["safetyreportid"]
            if item["llm_prob"] is None or item["label"] is None or rid not in by_id:
                continue
            reports.append(by_id[rid])
            y.append(item["label"])
            p_llm.append(item["llm_prob"])
        y = np.array(y)
        p_llm = np.array(p_llm)

        X = featurize(reports, top_pts, top_ings)
        for c in feats:
            if c not in X.columns:
                X[c] = 0
        p_cls = clf.predict_proba(X[feats])[:, 1]

        c_auc, c_lo, c_hi = boot_auroc(y, p_cls)
        l_auc, l_lo, l_hi = boot_auroc(y, p_llm)
        gap, g_lo, g_hi = boot_gap(y, p_cls, p_llm)
        sig = "*" if g_lo > 0 else " "
        print(f"{label:<30}{f'{c_auc:.3f} ({c_lo:.3f}-{c_hi:.3f})':>24}"
              f"{f'{l_auc:.3f} ({l_lo:.3f}-{l_hi:.3f})':>24}"
              f"{f'{gap:+.3f} ({g_lo:+.3f},{g_hi:+.3f}){sig}':>26}")

        gaps.append(gap)
        ns.append(len(y))
        cls_aucs.append(c_auc)
        llm_aucs.append(l_auc)

    print("-" * 104)
    print(f"{'MEAN (per-drug)':<30}{np.mean(cls_aucs):>24.3f}"
          f"{np.mean(llm_aucs):>24.3f}{np.mean(gaps):>+26.3f}")
    wc = np.average(cls_aucs, weights=ns)
    wl = np.average(llm_aucs, weights=ns)
    print(f"{'MEAN (n-weighted)':<30}{wc:>24.3f}{wl:>24.3f}{wl - wc:>+26.3f}")
    print(f"\n  n = {sum(ns):,} cases across the six-drug panel")
    print("  * = 95% CI for the gap excludes zero (LLM significantly better)")
    print("\nNext: 09_figures.py")


if __name__ == "__main__":
    main()
