
#!/usr/bin/env python3
"""
Phase C - Step 2 (FULL PANEL, consistent): LLM vs classical, same cases everywhere.
 
Both arms are evaluated on the IDENTICAL full sole-suspect panel per drug, at the
drug's natural base rate. No balanced subsample. This fixes the earlier
inconsistency (balanced-150 LLM vs full-panel classical) so every number in the
paper comes from one consistent evaluation set.
 
  - classical arm: panel-excluded XGBoost, reactions + demographics only
  - LLM arm: claude-sonnet-5 probs from llm_arm_full/*_llm.json (saw drug name)
 
Prereq: llm_arm_full/*_llm.json (full-panel harness), faers_temporal_train.parquet.
"""
 
import json
import os
from collections import Counter
 
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
 
DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"
LLM_DIR = os.path.join(DATA_DIR, "llm_arm_full")
NUMERIC = ["n_drugs", "n_suspect", "n_concomitant", "n_reactions",
           "age_years", "weight_kg"]
AGE_UNIT = {"800": 10, "801": 1, "802": 1/12, "803": 1/52, "804": 1/365, "805": 1/8760}
 
PANEL = [("suzetrigine (Journavx)", "journavx"),
         ("elafibranor (Iqirvo)", "iqirvo"),
         ("xanomeline-trospium (Cobenfy)", "cobenfy"),
         ("resmetirom (Rezdiffra)", "rezdiffra"),
         ("seladelpar (Livdelzi)", "livdelzi"),
         ("sotatercept (Winrevair)", "sotatercept")]
 
 
def age_years(p):
    a, u = p.get("patientonsetage"), p.get("patientonsetageunit")
    try:
        return float(a) * AGE_UNIT.get(u, 1) if a else None
    except (TypeError, ValueError):
        return None
 
 
def featurize(reports, top_pts, top_ings):
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
        rx = {(r.get("reactionmeddrapt") or "").strip().lower() for r in reactions}
        ing = {((d.get("activesubstance") or {}).get("activesubstancename") or "")
               .strip().lower() for d in drugs if d.get("drugcharacterization") == "1"}
        for t in top_pts:
            row[f"pt::{t}"] = 1 if t in rx else 0
        for i in top_ings:
            row[f"drug::{i}"] = 1 if i in ing else 0
        rows.append(row)
    return pd.DataFrame(rows)
 
 
def main():
    tr = pd.read_parquet(os.path.join(DATA_DIR, "faers_temporal_train.parquet"))
    top_pts = [c[4:] for c in tr.columns if c.startswith("pt::")]
    top_ings = [c[6:] for c in tr.columns if c.startswith("drug::")]
    feats = NUMERIC + ["sex"] + [c for c in tr.columns if "::" in c]
 
    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sex"]),
        ("bin", "passthrough", [c for c in feats if "::" in c])])
    clf = Pipeline([("pre", pre), ("xgb", XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
        tree_method="hist", n_jobs=-1, random_state=42))])
    clf.fit(tr[feats], tr["label_serious"])
 
    print(f"{'drug':<30}{'n':>6}{'base%':>7}{'classical':>11}{'LLM':>8}{'gap':>8}")
    print("-" * 70)
    ca, la, ns = [], [], []
    for label, slug in PANEL:
        with open(os.path.join(LLM_DIR, f"{slug}_llm.json"), encoding="utf-8") as f:
            llm = json.load(f)
        with open(os.path.join(DATA_DIR, f"{slug}_suspect.json"), encoding="utf-8") as f:
            allrep = {r.get("safetyreportid"): r for r in json.load(f)}
 
        rows, y, lp = [], [], []
        for item in llm:
            rid = item["safetyreportid"]
            if item["llm_prob"] is None or item["label"] is None or rid not in allrep:
                continue
            rows.append(allrep[rid])
            y.append(item["label"])
            lp.append(item["llm_prob"])
        y = np.array(y)
        lp = np.array(lp)
 
        Xdf = featurize(rows, top_pts, top_ings)
        for c in feats:
            if c not in Xdf.columns:
                Xdf[c] = 0
        cp = clf.predict_proba(Xdf[feats])[:, 1]
 
        if len(set(y)) < 2:
            print(f"{label:<30}{len(y):>6}  (one class only, skip)")
            continue
        c_auc = roc_auc_score(y, cp)
        l_auc = roc_auc_score(y, lp)
        print(f"{label:<30}{len(y):>6}{y.mean():>7.0%}{c_auc:>11.3f}{l_auc:>8.3f}"
              f"{l_auc - c_auc:>+8.3f}")
        ca.append(c_auc); la.append(l_auc); ns.append(len(y))
 
    print("-" * 70)
    print(f"{'MEAN (per-drug)':<30}{'':>6}{'':>7}{np.mean(ca):>11.3f}"
          f"{np.mean(la):>8.3f}{np.mean(la) - np.mean(ca):>+8.3f}")
    # pooled (n-weighted) mean too
    wc = np.average(ca, weights=ns)
    wl = np.average(la, weights=ns)
    print(f"{'MEAN (n-weighted)':<30}{sum(ns):>6}{'':>7}{wc:>11.3f}{wl:>8.3f}"
          f"{wl - wc:>+8.3f}")
    print("\nBoth arms scored on the SAME full-panel cases at natural base rates. "
          "This is the consistent, defensible comparison for the paper.")
 
 
if __name__ == "__main__":
    main()
 