#!/usr/bin/env python3
"""
Step 6 - Subgroup audit with bootstrap confidence intervals.

TRIPOD+AI item 23a asks for performance estimates with confidence intervals
"including for key subgroups". This script provides them.

Two metrics are reported per subgroup, and the distinction between them is the
substance of the paper's equity finding:

  AUROC       - can the model RANK cases correctly within this group?
  recall@thr  - at a single global threshold, does it actually CATCH the serious
                cases in this group?

A model can rank well in every group (fair discrimination) yet catch serious
cases at very different rates (unfair allocation), because groups have different
base rates. That is an operating-point problem, and it is fixable with
group-aware thresholds - which the final column reports.

Expected findings:
  * region: US recall 0.85 (0.84-0.86) vs non-US 0.93 (0.93-0.94), while AUROC
    is comparable (0.862 vs 0.869) - fair ranking, unfair operating point
  * MISSING DEMOGRAPHICS: recall falls to 0.70 (0.68-0.72) for unrecorded sex
    and 0.82 (0.80-0.83) for unrecorded age - the largest inequity, and one a
    discrimination-only analysis would miss entirely
  * reporter unknown (n=510): AUROC 0.512 (0.461-0.562) - a confidence interval
    spanning chance. This is a DISCRIMINATION FAILURE, not a threshold problem;
    no threshold repairs a model that cannot rank. Such reports should bypass
    automated triage.

Input:  faers_temporal_{train,test}.parquet
Output: printed table (Table 4 and Figure 4 in the paper)
Runtime: ~2-4 minutes at N_BOOT=1000
"""

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"   # <-- set to your data path

N_BOOT = 1000       # resamples per subgroup metric
THRESH = 0.50       # the global operating threshold reported in the paper
SEED = 42
RNG = np.random.default_rng(SEED)

NUMERIC = ["n_drugs", "n_suspect", "n_concomitant", "n_reactions",
           "age_years", "weight_kg"]

SEX_LABELS = {"1": "male", "2": "female"}
REPORTER_LABELS = {"1": "physician", "2": "pharmacist",
                   "3": "other HP", "5": "consumer"}


def train_clinical(tr, te):
    """The reported model: clinical content only, no reporting-channel features."""
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
    return clf.predict_proba(te[feats])[:, 1]


def auroc(y, p):
    return roc_auc_score(y, p)


def recall_at(y, p, th=THRESH):
    pred = p >= th
    tp = ((pred) & (y == 1)).sum()
    fn = ((~pred) & (y == 1)).sum()
    return tp / (tp + fn) if (tp + fn) else np.nan


def boot_ci(y, p, metric, n_boot=N_BOOT):
    """Stratified bootstrap: resample within outcome class, preserving base rate."""
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(pos) < 2 or len(neg) < 2:
        return np.nan, np.nan
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([RNG.choice(pos, len(pos), replace=True),
                              RNG.choice(neg, len(neg), replace=True)])
        try:
            stats.append(metric(y[idx], p[idx]))
        except ValueError:
            continue
    if not stats:
        return np.nan, np.nan
    return tuple(np.percentile(stats, [2.5, 97.5]))


def threshold_for(y, p, target=0.90):
    for th in np.round(np.arange(0.99, 0.0, -0.01), 2):
        if recall_at(y, p, th) >= target:
            return float(th)
    return 0.0


def age_band(a):
    if pd.isna(a):
        return "unknown"
    if a < 18:
        return "0-17"
    if a < 45:
        return "18-44"
    if a < 65:
        return "45-64"
    if a < 75:
        return "65-74"
    return "75+"


def main():
    tr = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_train.parquet"))
    te = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_test.parquet"))
    y = te["label_serious"].values
    p = train_clinical(tr, te)

    lo, hi = boot_ci(y, p, auroc)
    print(f"OVERALL   n={len(y):,}   base rate {y.mean():.1%}   "
          f"AUROC {auroc(y, p):.3f} (95% CI {lo:.3f}-{hi:.3f})   "
          f"threshold {THRESH}\n")

    dims = {
        "REGION": (np.where(te["is_us"].values == 1, "US", "non-US"),
                   ["US", "non-US"]),
        "SEX": (te["sex"].map(SEX_LABELS).fillna("other/unknown").values,
                ["male", "female", "other/unknown"]),
        "AGE BAND": (te["age_years"].map(age_band).values,
                     ["0-17", "18-44", "45-64", "65-74", "75+", "unknown"]),
        "REPORTER": (te["qualification"].map(REPORTER_LABELS)
                     .fillna("other/unknown").values,
                     ["physician", "pharmacist", "other HP", "consumer",
                      "other/unknown"]),
    }

    for dim, (groups, order) in dims.items():
        print(f"=== {dim} ===")
        print(f"{'group':<15}{'n':>7}{'base':>8}{'AUROC (95% CI)':>24}"
              f"{'recall@thr (95% CI)':>26}{'own 90% thr':>13}")
        for g in order:
            m = groups == g
            if m.sum() == 0:
                continue
            ys, ps = y[m], p[m]
            if len(set(ys)) < 2:
                print(f"{g:<15}{int(m.sum()):>7}   (single outcome class - no CI)")
                continue
            a = auroc(ys, ps)
            a_lo, a_hi = boot_ci(ys, ps, auroc)
            r = recall_at(ys, ps)
            r_lo, r_hi = boot_ci(ys, ps, recall_at)
            own = threshold_for(ys, ps)
            chance = "  <-- CI spans chance" if a_lo < 0.5 < a_hi else ""
            print(f"{g:<15}{int(m.sum()):>7}{ys.mean():>8.1%}"
                  f"{f'{a:.3f} ({a_lo:.3f}-{a_hi:.3f})':>24}"
                  f"{f'{r:.2f} ({r_lo:.2f}-{r_hi:.2f})':>26}"
                  f"{own:>13.2f}{chance}")
        print()

    print("Read: comparable AUROC within a dimension = the model RANKS fairly.")
    print("Diverging recall@thr = a single global cutoff ALLOCATES unfairly.")
    print("The fix is group-aware thresholds - except where the AUROC confidence")
    print("interval spans 0.5, which is a discrimination failure no threshold can")
    print("repair; those cases should bypass automated triage.")
    print("\nNext: 07_llm_arm.py")


if __name__ == "__main__":
    main()
