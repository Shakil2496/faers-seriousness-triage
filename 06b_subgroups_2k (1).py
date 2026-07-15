#!/usr/bin/env python3
"""
Step 6b - Subgroup audit at the resample count the manuscript states.

WHY THIS SCRIPT EXISTS
----------------------
06_subgroups.py uses N_BOOT = 1000. Section 2.8 of the manuscript tells the
reader the confidence intervals were computed with 2,000 resamples. Table 4's
sixteen intervals were therefore produced by a procedure the paper does not
describe.

This is the same class of gap as the missing paired bootstrap in Table 5:
the text and the artifact disagree. This script settles it by running the
subgroup audit at 2,000 resamples - the number Section 2.8 states - and
checking the result against Table 4 as currently printed.

Nothing else changes. Same model, same seed (42), same threshold (0.50),
same stratified bootstrap, same subgroup definitions as 06_subgroups.py.

Table 4 carries the equity finding the paper's reliability boundary is built
on (unknown reporter: AUROC 0.512, 95% CI 0.461-0.562). Those intervals need
to come from the procedure the Methods describe.

OUTPUT
------
outputs/subgroup_metrics.csv   <- consumed by 09_figures.py (Fig 4)
outputs/calibration_deciles.csv <- consumed by 09_figures.py (Fig 2)
outputs/clinical_metrics.json   <- overall AUROC / Brier

Runtime: ~5-10 minutes (2,000 resamples x 17 groups x 2 metrics).

    python 06b_subgroups_2k.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

# ---- identical configuration to 06_subgroups.py -------------------------
DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"

N_BOOT = 2000       # <-- CHANGED from 1000. Section 2.8 says "2,000 resamples".
THRESH = 0.50
SEED = 42
RNG = np.random.default_rng(SEED)

NUMERIC = ["n_drugs", "n_suspect", "n_concomitant", "n_reactions",
           "age_years", "weight_kg"]
SEX_LABELS = {"1": "male", "2": "female"}
REPORTER_LABELS = {"1": "physician", "2": "pharmacist",
                   "3": "other HP", "5": "consumer"}

OUT = Path(DATA_DIR) / "outputs"
OUT.mkdir(exist_ok=True)

# Table 4 as currently printed: (dimension, group) -> (auroc, lo, hi, recall, r_lo, r_hi)
TABLE4 = {
    ("REGION", "US"):              (.862, .856, .868, .85, .84, .86),
    ("REGION", "non-US"):          (.869, .853, .884, .93, .93, .94),
    ("SEX", "male"):               (.896, .889, .903, .96, .95, .96),
    ("SEX", "female"):             (.897, .891, .902, .92, .92, .93),
    ("SEX", "other/unknown"):      (.885, .876, .894, .70, .68, .72),
    ("AGE BAND", "0-17"):          (.951, .937, .964, .95, .93, .97),
    ("AGE BAND", "18-44"):         (.908, .896, .919, .94, .93, .95),
    ("AGE BAND", "45-64"):         (.925, .917, .933, .94, .94, .95),
    ("AGE BAND", "65-74"):         (.914, .903, .924, .94, .93, .95),
    ("AGE BAND", "75+"):           (.875, .861, .891, .95, .94, .96),
    ("AGE BAND", "unknown"):       (.846, .838, .853, .82, .80, .83),
    ("REPORTER", "physician"):     (.905, .894, .915, .91, .91, .92),
    ("REPORTER", "pharmacist"):    (.896, .879, .914, .91, .89, .93),
    ("REPORTER", "other HP"):      (.930, .924, .936, .91, .90, .92),
    ("REPORTER", "consumer"):      (.877, .870, .883, .88, .87, .89),
    ("REPORTER", "other/unknown"): (.512, .461, .562, .94, .91, .96),
}


def train_clinical(tr, te):
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
    """Stratified bootstrap, identical to 06_subgroups.py except n_boot."""
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

    print(f"Subgroup audit, {N_BOOT:,} resamples, seed {SEED}, threshold {THRESH}\n")

    o_auc = auroc(y, p)
    o_brier = brier_score_loss(y, p)
    lo, hi = boot_ci(y, p, auroc)
    print(f"OVERALL   n={len(y):,}   base {y.mean():.1%}   "
          f"AUROC {o_auc:.3f} ({lo:.3f}-{hi:.3f})   Brier {o_brier:.3f}\n")

    (OUT / "clinical_metrics.json").write_text(json.dumps(
        {"auroc": float(o_auc), "auroc_lo": float(lo), "auroc_hi": float(hi),
         "brier": float(o_brier), "n_test": int(len(y)),
         "base_rate": float(y.mean()), "threshold": THRESH}, indent=2))

    # calibration deciles -> Fig 2
    order = np.argsort(p)
    cal = [{"decile": i, "mean_pred": float(p[idx].mean()),
            "obs_frac": float(y[idx].mean()), "n": int(len(idx))}
           for i, idx in enumerate(np.array_split(order, 10), 1)]
    pd.DataFrame(cal).to_csv(OUT / "calibration_deciles.csv", index=False)

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

    recs = []
    for dim, (groups, order_) in dims.items():
        print(f"=== {dim} ===")
        print(f"{'group':<15}{'n':>7}{'base':>8}{'AUROC (95% CI)':>24}"
              f"{'recall@0.50 (95% CI)':>26}{'own 90% thr':>13}")
        for g in order_:
            m = groups == g
            if m.sum() == 0:
                continue
            ys, ps = y[m], p[m]
            if len(set(ys)) < 2:
                print(f"{g:<15}{int(m.sum()):>7}   (single outcome class)")
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
            recs.append(dict(dimension=dim, group=g, n=int(m.sum()),
                             base_rate=float(ys.mean()),
                             auroc=a, auroc_lo=a_lo, auroc_hi=a_hi,
                             recall=r, recall_lo=r_lo, recall_hi=r_hi,
                             own_threshold=own))
        print()

    pd.DataFrame(recs).to_csv(OUT / "subgroup_metrics.csv", index=False)
    print(f"wrote {OUT/'subgroup_metrics.csv'}, "
          f"{OUT/'calibration_deciles.csv'}, {OUT/'clinical_metrics.json'}")

    # ---- does this reproduce Table 4? -----------------------------------
    print("\n=== CHECK AGAINST TABLE 4 AS CURRENTLY PRINTED ===")
    bad = 0
    for r in recs:
        exp = TABLE4.get((r["dimension"], r["group"]))
        if not exp:
            continue
        e_a, e_alo, e_ahi, e_r, e_rlo, e_rhi = exp
        d = [("AUROC",     r["auroc"],     e_a,   0.0015),
             ("AUROC lo",  r["auroc_lo"],  e_alo, 0.0015),
             ("AUROC hi",  r["auroc_hi"],  e_ahi, 0.0015),
             ("recall",    r["recall"],    e_r,   0.006),
             ("recall lo", r["recall_lo"], e_rlo, 0.006),
             ("recall hi", r["recall_hi"], e_rhi, 0.006)]
        offs = [(k, got, want) for k, got, want, tol in d if abs(got - want) > tol]
        if offs:
            bad += 1
            print(f"  MISMATCH  {r['dimension']} / {r['group']}")
            for k, got, want in offs:
                print(f"      {k:<10} computed {got:.3f}   manuscript {want:.3f}")

    print()
    if bad == 0:
        print("  All 16 subgroups match Table 4.")
        print("  The intervals are stable at 2,000 resamples. Only the stated")
        print("  resample count needed reconciling, not the numbers.")
    else:
        print(f"  {bad} subgroup(s) DO NOT match. Table 4 must be updated to the")
        print("  computed values above, and Figure 4 redrawn.")

    # the finding the reliability boundary rests on
    unk = next((r for r in recs if r["dimension"] == "REPORTER"
                and r["group"] == "other/unknown"), None)
    if unk:
        spans = unk["auroc_lo"] < 0.5 < unk["auroc_hi"]
        print(f"\n  EQUITY FINDING (unknown reporter, n={unk['n']}):")
        print(f"    AUROC {unk['auroc']:.3f} "
              f"(95% CI {unk['auroc_lo']:.3f}-{unk['auroc_hi']:.3f})")
        print(f"    CI spans chance (0.50): {spans}")
        print("    -> reliability boundary HOLDS" if spans else
              "    -> WARNING: no longer spans chance. The reliability boundary,\n"
              "       the dashboard refusal panel and Section 4.7 all depend on\n"
              "       this. Do not submit without reconciling.")

    print("\nNext: 09_figures.py")


if __name__ == "__main__":
    main()
