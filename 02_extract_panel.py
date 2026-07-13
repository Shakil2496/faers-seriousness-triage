#!/usr/bin/env python3
"""
Step 2 — Build the held-out novel-drug panel.

For each of the six first-in-class panel drugs, retrieves all FAERS reports where
the drug is the SOLE PRIMARY SUSPECT (the only drug with drugcharacterization=1),
restricted to reports received on or after that drug's US approval date.

Two cleaning steps matter and are applied here:
  * sole-primary-suspect filter — removes attribution ambiguity from co-suspects.
    For sotatercept this is decisive: raw sole-suspect cases are contaminated by
    co-reported background prostacyclin therapies (treprostinil, etc.), and
    cleaning reduces 1,334 -> 258 cases and raises the serious base rate 42% -> 75%.
  * approval-date floor — excludes any report predating the drug's US approval.

Drugs screened but EXCLUDED during panel construction, with reasons:
  * nemolizumab   — 90% consumer-reported; non-serious class dominated by
                    administrative reports, not clinical non-serious events
  * lenacapavir   — same administrative skew
  * acoramidis    — minority class too thin to evaluate
  * gepotidacin   — minority class too thin to evaluate

Requires: secrets.txt
Output:   <slug>_suspect.json  x6
"""

import json
import os
import time
import urllib.parse

from curl_cffi import requests as creq

DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"   # <-- set to your data path
KEY_PATH = "secrets.txt"

ENDPOINT = "https://api.fda.gov/drug/event.json"
PAGE = 500
CAP = 10000

# (slug, match terms, US approval date) — approval dates verified against FDA sources
PANEL = [
    ("journavx",    ("suzetrigine", "journavx"),                 "20250130"),
    ("iqirvo",      ("elafibranor", "iqirvo"),                   "20240610"),
    ("cobenfy",     ("xanomeline", "trospium", "cobenfy"),       "20240926"),
    ("rezdiffra",   ("resmetirom", "rezdiffra"),                 "20240314"),
    ("livdelzi",    ("seladelpar", "livdelzi"),                  "20240814"),
    ("sotatercept", ("sotatercept", "winrevair"),                "20240326"),
]

with open(KEY_PATH) as f:
    API_KEY = f.read().strip()


def matches(drug, terms):
    """Case-insensitive match across reported name, substance, and openFDA fields."""
    fields = [drug.get("medicinalproduct") or "",
              (drug.get("activesubstance") or {}).get("activesubstancename") or ""]
    openfda = drug.get("openfda") or {}
    for key in ("brand_name", "generic_name", "substance_name"):
        fields.extend(openfda.get(key) or [])
    hay = " ".join(fields).lower()
    return any(t in hay for t in terms)


def fetch(slug, terms, approval_date):
    """All reports naming the drug, received on/after approval."""
    reports, skip = [], 0
    search = (f'patient.drug.medicinalproduct:"{terms[0]}"'
              f'+AND+receivedate:[{approval_date}+TO+20260101]')
    q = urllib.parse.quote(search, safe=':[]+"')

    while skip < CAP:
        url = f"{ENDPOINT}?search={q}&limit={PAGE}&skip={skip}&api_key={API_KEY}"
        try:
            r = creq.get(url, impersonate="chrome", timeout=45)
        except Exception as e:
            print(f"    retry: {type(e).__name__}")
            time.sleep(4)
            continue
        if r.status_code == 404:
            break
        r.raise_for_status()
        batch = r.json().get("results", [])
        if not batch:
            break
        reports.extend(batch)
        skip += len(batch)
        if len(batch) < PAGE:
            break
        time.sleep(0.2)

    return reports


def sole_primary_suspect(reports, terms):
    """Keep only cases where the target drug is the ONLY primary suspect."""
    kept = []
    for rep in reports:
        primary = [d for d in (rep.get("patient") or {}).get("drug", []) or []
                   if d.get("drugcharacterization") == "1"]
        if not primary:
            continue
        target = [d for d in primary if matches(d, terms)]
        others = [d for d in primary if not matches(d, terms)]
        if target and not others:          # target present, nothing else primary
            kept.append(rep)
    return kept


def label_counts(reports):
    s = sum(1 for r in reports if r.get("serious") == "1")
    n = sum(1 for r in reports if r.get("serious") == "2")
    return s, n


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"{'drug':<14}{'retrieved':>11}{'sole-suspect':>14}"
          f"{'serious':>9}{'non-ser':>9}{'base':>7}")
    print("-" * 64)

    for slug, terms, approval in PANEL:
        out = os.path.join(DATA_DIR, f"{slug}_suspect.json")
        if os.path.exists(out):
            with open(out, encoding="utf-8") as f:
                kept = json.load(f)
            print(f"{slug:<14}{'(cached)':>11}{len(kept):>14}")
            continue

        raw = fetch(slug, terms, approval)
        kept = sole_primary_suspect(raw, terms)
        s, n = label_counts(kept)
        base = s / (s + n) if (s + n) else 0

        with open(out, "w", encoding="utf-8") as f:
            json.dump(kept, f)
        print(f"{slug:<14}{len(raw):>11}{len(kept):>14}{s:>9}{n:>9}{base:>7.1%}")

    print("\nPanel extracted. Next: 03_build_features.py")


if __name__ == "__main__":
    main()
