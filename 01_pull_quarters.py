#!/usr/bin/env python3
"""
Step 1 — Retrieve the FAERS training corpus.

Pulls four quarters of 2025 from the openFDA drug-event endpoint, one file per
quarter, capped at 25,000 reports each. Resumable: completed quarters are skipped,
so a network failure mid-run costs nothing. Run this first.

Requires: secrets.txt (a single line containing your openFDA API key).
Output:   raw_quarter_<start>_<end>.json  x4  (100,000 reports total)
"""

import json
import os
import time
import urllib.parse

from curl_cffi import requests as creq

DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"   # <-- set to your data path
KEY_PATH = "secrets.txt"

ENDPOINT = "https://api.fda.gov/drug/event.json"
CAP_PER_QUARTER = 25000
PAGE = 500          # smaller pages are less likely to stall

QUARTERS = [("20250101", "20250331"), ("20250401", "20250630"),
            ("20250701", "20250930"), ("20251001", "20251231")]

with open(KEY_PATH) as f:
    API_KEY = f.read().strip()


def pull_quarter(d0, d1):
    """Page through one quarter, retrying transient failures."""
    reports, skip = [], 0
    search = f"receivedate:[{d0}+TO+{d1}]"
    q = urllib.parse.quote(search, safe=":[]+")

    while skip < CAP_PER_QUARTER:
        url = f"{ENDPOINT}?search={q}&limit={PAGE}&skip={skip}&api_key={API_KEY}"
        got = None
        for attempt in range(5):
            try:
                r = creq.get(url, impersonate="chrome", timeout=45)
            except Exception as e:
                print(f"    network retry (skip={skip}, attempt {attempt + 1}): "
                      f"{type(e).__name__}")
                time.sleep(5)
                continue
            if r.status_code == 404:        # openFDA's "no more results"
                return reports
            if r.status_code >= 500 and attempt < 4:
                print(f"    HTTP {r.status_code} retry (skip={skip})")
                time.sleep(4)
                continue
            got = r
            break

        if got is None:
            print(f"    giving up at skip={skip}; saving partial ({len(reports)})")
            return reports

        got.raise_for_status()
        batch = got.json().get("results", [])
        if not batch:
            break
        reports.extend(batch)
        skip += len(batch)
        print(f"    {len(reports)}...")
        if len(batch) < PAGE:
            break

    return reports


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    for d0, d1 in QUARTERS:
        path = os.path.join(DATA_DIR, f"raw_quarter_{d0}_{d1}.json")
        if os.path.exists(path):
            print(f"cached, skipping {d0}..{d1}")
            continue
        print(f"pulling {d0}..{d1} ...")
        reports = pull_quarter(d0, d1)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(reports, f)
        print(f"  saved {len(reports)} -> {path}")

    print("\nAll quarters cached. Next: 02_extract_panel.py")


if __name__ == "__main__":
    main()
