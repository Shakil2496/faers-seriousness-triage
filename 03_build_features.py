#!/usr/bin/env python3
"""
Step 3 — Build the leakage-controlled feature matrix and temporal split.

Pools the four cached quarters, deduplicates, excludes the panel drugs and
unlabelled cases, then builds a feature matrix under two leakage defences:

  1. BLACKLIST — every field downstream of the seriousness determination is
     excluded: the seriousness sub-criteria, the expedited-reporting flag
     (which is triggered BY seriousness), reaction-outcome, and action-taken.
     A hard assertion fails the build if any banned field reaches the matrix.

  2. OUTCOME-EQUIVALENT TERMS — reaction PTs that are themselves seriousness
     criteria wearing a reaction label ("death", "hospitalisation") are removed.
     This exclusion is deliberately NARROW: clinically serious but non-circular
     events (seizure, pneumonia, acute kidney injury, neutropenia) are RETAINED,
     because they are known at intake and are legitimate predictive signal.

Vocabulary (top 800 reaction PTs, top 300 suspect ingredients) is derived from
the TRAINING quarters only, then applied unchanged to the test quarter — so no
information from the future leaks into feature selection.

Split: train = 2025 Q1-Q3, test = 2025 Q4 (chronological, not random).

Output: faers_temporal_train.parquet, faers_temporal_test.parquet
"""

import json
import os
from collections import Counter

import pandas as pd

DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"   # <-- set to your data path

TOP_PT = 800            # reaction-term vocabulary size
TOP_INGREDIENT = 300    # suspect-ingredient vocabulary size

TRAIN_QUARTERS = [("20250101", "20250331"), ("20250401", "20250630"),
                  ("20250701", "20250930")]
TEST_QUARTERS = [("20251001", "20251231")]

PANEL_TERMS = ("journavx", "suzetrigine", "cobenfy", "xanomeline", "rezdiffra",
               "resmetirom", "iqirvo", "elafibranor", "livdelzi", "seladelpar",
               "winrevair", "sotatercept")

# Fields downstream of the seriousness label — must never enter the matrix.
BANNED = {
    "serious", "seriousnessdeath", "seriousnesslifethreatening",
    "seriousnesshospitalization", "seriousnessdisabling",
    "seriousnesscongenitalanomali", "seriousnessother",
    "fulfillexpeditecriteria",          # triggered BY seriousness
    "reactionoutcome",                  # how the reaction resolved
    "actiondrug",                       # e.g. drug withdrawn
}

# Reaction PTs that ARE a seriousness criterion wearing a reaction label, or that
# are definitionally non-serious. Either way they encode the label rather than
# predict it, so they are removed.
#
# NOTE the narrowness of this list. Clinically serious but NON-CIRCULAR events —
# seizure, pneumonia, acute kidney injury, neutropenia, and the antineoplastic and
# immunosuppressant drug indicators — are deliberately RETAINED. They are recorded
# at intake, they are not themselves seriousness criteria, and they are precisely
# the signal a triage model should learn. Excluding them would destroy real signal;
# retaining the terms below would be circular.
OUTCOME_EQUIVALENT_PT = {
    # seriousness criteria expressed as reaction terms
    "death", "sudden death", "completed suicide",
    "hospitalisation", "hospitalization",
    "life threatening", "disability", "incapacity",
    "congenital anomaly", "foetal death", "fetal death",
    # definitionally non-serious
    "no adverse event", "no adverse drug effect",
}

AGE_UNIT = {"800": 10, "801": 1, "802": 1 / 12, "803": 1 / 52,
            "804": 1 / 365, "805": 1 / 8760}


def load(quarters):
    out = []
    for d0, d1 in quarters:
        path = os.path.join(DATA_DIR, f"raw_quarter_{d0}_{d1}.json")
        with open(path, encoding="utf-8") as f:
            out.extend(json.load(f))
        print(f"  loaded {d0}..{d1}")
    return out


def dedup_latest(reports):
    latest = {}
    for rep in reports:
        rid = rep.get("safetyreportid")
        ver = int(rep.get("safetyreportversion", 0) or 0)
        if rid not in latest or ver > latest[rid][0]:
            latest[rid] = (ver, rep)
    return [v[1] for v in latest.values()]


def is_panel(rep):
    for d in (rep.get("patient") or {}).get("drug", []) or []:
        fields = [d.get("medicinalproduct") or "",
                  (d.get("activesubstance") or {}).get("activesubstancename") or ""]
        openfda = d.get("openfda") or {}
        for key in ("brand_name", "generic_name", "substance_name"):
            fields.extend(openfda.get(key) or [])
        if any(t in " ".join(fields).lower() for t in PANEL_TERMS):
            return True
    return False


def label_of(rep):
    s = rep.get("serious")
    return 1 if s == "1" else 0 if s == "2" else None


def age_years(p):
    a, u = p.get("patientonsetage"), p.get("patientonsetageunit")
    try:
        return float(a) * AGE_UNIT.get(u, 1) if a else None
    except (TypeError, ValueError):
        return None


def reactions_of(rep):
    return {(r.get("reactionmeddrapt") or "").strip().lower()
            for r in (rep.get("patient") or {}).get("reaction", []) or []}


def ingredients_of(rep):
    return {((d.get("activesubstance") or {}).get("activesubstancename") or "")
            .strip().lower()
            for d in (rep.get("patient") or {}).get("drug", []) or []
            if d.get("drugcharacterization") == "1"}


def build_vocab(reports):
    """Vocabulary from TRAINING data only — prevents look-ahead leakage.

    Terms are counted by occurrence (a report listing the same reaction twice
    contributes two counts), then the outcome-equivalent terms are filtered out
    and the top-N retained.
    """
    pt, ing = Counter(), Counter()
    for rep in reports:
        p = rep.get("patient") or {}
        for rx in p.get("reaction", []) or []:
            t = (rx.get("reactionmeddrapt") or "").strip().lower()
            if t:
                pt[t] += 1
        for d in p.get("drug", []) or []:
            if d.get("drugcharacterization") == "1":
                i = ((d.get("activesubstance") or {}).get("activesubstancename")
                     or "").strip().lower()
                if i:
                    ing[i] += 1
    top_pt = [t for t, _ in pt.most_common()
              if t not in OUTCOME_EQUIVALENT_PT][:TOP_PT]
    top_ing = [i for i, _ in ing.most_common(TOP_INGREDIENT)]
    return top_pt, top_ing


def featurize(reports, top_pt, top_ing):
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
        src = rep.get("primarysource") or {}

        row = {
            "safetyreportid": rep.get("safetyreportid"),
            "n_drugs": len(drugs),
            "n_suspect": roles.get("1", 0),
            "n_concomitant": roles.get("2", 0),
            "n_reactions": len(reactions),
            "age_years": age_years(p),
            "weight_kg": w,
            "sex": p.get("patientsex") or "unk",
            # channel features: retained in the matrix for the leakage audit and
            # the subgroup analysis, but EXCLUDED from the clinical model (step 5)
            "qualification": src.get("qualification") or "unk",
            "is_us": 1 if (rep.get("occurcountry") or "").upper() == "US" else 0,
            "reporttype": rep.get("reporttype") or "unk",
            "label_serious": label_of(rep),
        }
        rx = reactions_of(rep)
        ing = ingredients_of(rep)
        for t in top_pt:
            row[f"pt::{t}"] = 1 if t in rx else 0
        for i in top_ing:
            row[f"drug::{i}"] = 1 if i in ing else 0
        rows.append(row)
    return pd.DataFrame(rows)


def assert_no_leakage(df):
    """Hard stop: fail the build rather than emit a contaminated matrix."""
    banned_present = [c for c in df.columns if c in BANNED and c != "label_serious"]
    if banned_present:
        raise RuntimeError(f"LEAKAGE: banned fields in matrix: {banned_present}")
    bad_pt = [c for c in df.columns
              if c.startswith("pt::") and c[4:] in OUTCOME_EQUIVALENT_PT]
    if bad_pt:
        raise RuntimeError(f"LEAKAGE: outcome-equivalent PTs in matrix: {bad_pt}")
    print("  leakage assertion passed: no banned fields, no outcome-equivalent PTs")


def prepare(reports):
    reports = dedup_latest(reports)
    reports = [r for r in reports if not is_panel(r) and label_of(r) is not None]
    return reports


def main():
    print("loading training quarters:")
    train_raw = load(TRAIN_QUARTERS)
    print("loading test quarter:")
    test_raw = load(TEST_QUARTERS)

    train = prepare(train_raw)
    test = prepare(test_raw)
    print(f"\nafter dedup + dropping panel drugs and unlabelled cases:")
    print(f"  train {len(train):,}   test {len(test):,}")

    top_pt, top_ing = build_vocab(train)     # TRAIN ONLY
    print(f"  vocabulary from training data only: "
          f"{len(top_pt)} reaction terms, {len(top_ing)} ingredients")

    for name, reports in [("train", train), ("test", test)]:
        df = featurize(reports, top_pt, top_ing)
        assert_no_leakage(df)
        path = os.path.join(DATA_DIR, f"faers_temporal_{name}.parquet")
        df.to_parquet(path, index=False)
        print(f"  {name}: {df.shape[0]:,} x {df.shape[1]} -> {path}")
        print(f"    serious {df['label_serious'].mean():.1%}")

    print("\nNext: 04_leakage_audit.py")


if __name__ == "__main__":
    main()
