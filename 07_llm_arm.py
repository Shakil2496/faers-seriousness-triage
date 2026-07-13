
#!/usr/bin/env python3
"""
Phase C - Step 1 (FULL PANEL): LLM arm on ALL sole-suspect cases per drug.
 
Same leakage-safe prompt and model as before, but no 150-cap: scores every
case at the drug's natural base rate, so the LLM faces the same class imbalance
as the classical model. Checkpoints per drug (resumable). Prints running cost.
 
Writes to llm_arm_full/ (separate from the old balanced-150 llm_arm/).
 
Prereqs: pip install anthropic ; anthropic_key.txt with sk-ant-... key.
Cost: ~$5 for ~4,500 cases at Sonnet 5 intro pricing.
"""
 
import json
import os
import re
import time
 
from anthropic import Anthropic
 
DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"
KEY_PATH = "anthropic_key.txt"
OUT_DIR = os.path.join(DATA_DIR, "llm_arm_full")
MODEL = "claude-sonnet-5"
SLEEP = 0.25
 
PANEL = [("suzetrigine (Journavx)", "journavx"),
         ("elafibranor (Iqirvo)", "iqirvo"),
         ("xanomeline-trospium (Cobenfy)", "cobenfy"),
         ("resmetirom (Rezdiffra)", "rezdiffra"),
         ("seladelpar (Livdelzi)", "livdelzi"),
         ("sotatercept (Winrevair)", "sotatercept")]
 
with open(KEY_PATH) as f:
    client = Anthropic(api_key=f.read().strip())
 
SEX = {"1": "male", "2": "female"}
SYSTEM = (
    "You are a pharmacovigilance triage assistant. Given an adverse-event report "
    "(the drug and the reactions/patient information available at intake), estimate "
    "the probability that the case is SERIOUS by regulatory criteria (death, "
    "life-threatening, hospitalization, disability, congenital anomaly, or other "
    "medically important event). Use your medical knowledge of the drug and the "
    "reactions. Respond with ONE line only, exactly: PROB=<0.00-1.00> | <=12 word reason. "
    "Do not add anything else.")
 
 
def case_text(drug_label, rep):
    p = rep.get("patient") or {}
    reactions = [rx.get("reactionmeddrapt") or "" for rx in (p.get("reaction") or [])]
    reactions = [r for r in reactions if r]
    age = p.get("patientonsetage")
    sex = SEX.get(str(p.get("patientsex")), "unknown")
    drugs = p.get("drug") or []
    n_susp = sum(1 for d in drugs if d.get("drugcharacterization") == "1")
    n_conc = sum(1 for d in drugs if d.get("drugcharacterization") == "2")
    return (f"Suspect drug: {drug_label}\n"
            f"Reactions: {', '.join(reactions) if reactions else 'none reported'}\n"
            f"Patient age: {age if age else 'unknown'}; sex: {sex}\n"
            f"Suspect drugs on case: {n_susp}; concomitant drugs: {n_conc}")
 
 
def label_of(rep):
    s = rep.get("serious")
    return 1 if s == "1" else 0 if s == "2" else None
 
 
def ask_llm(drug_label, rep):
    msg = client.messages.create(
        model=MODEL, max_tokens=60, system=SYSTEM,
        messages=[{"role": "user", "content": case_text(drug_label, rep)}])
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    m = re.search(r"PROB\s*=\s*([01](?:\.\d+)?)", text)
    prob = float(m.group(1)) if m else None
    return prob, text.strip(), msg.usage.input_tokens, msg.usage.output_tokens
 
 
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tot_in = tot_out = 0
    for label, slug in PANEL:
        out_path = os.path.join(OUT_DIR, f"{slug}_llm.json")
        if os.path.exists(out_path):
            print(f"[skip] {label} already done")
            continue
        with open(os.path.join(DATA_DIR, f"{slug}_suspect.json"), encoding="utf-8") as f:
            reports = json.load(f)
        # only cases with a usable label (serious 1 or 2)
        cases = [r for r in reports if label_of(r) is not None]
        print(f"\n{label}: {len(cases)} cases")
        results = []
        for i, rep in enumerate(cases, 1):
            try:
                prob, raw, ti, to = ask_llm(label, rep)
            except Exception as e:
                print(f"  [{i}] error: {type(e).__name__} - retry")
                time.sleep(3)
                try:
                    prob, raw, ti, to = ask_llm(label, rep)
                except Exception as e2:
                    print(f"  [{i}] failed: {e2}")
                    continue
            tot_in += ti
            tot_out += to
            results.append({"safetyreportid": rep.get("safetyreportid"),
                            "label": label_of(rep), "llm_prob": prob, "raw": raw})
            if i % 100 == 0:
                cost = tot_in / 1e6 * 2 + tot_out / 1e6 * 10
                print(f"  {i}/{len(cases)}  (running cost ${cost:.2f})")
            time.sleep(SLEEP)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        got = sum(1 for r in results if r["llm_prob"] is not None)
        print(f"  saved {len(results)} ({got} parsed) -> {out_path}")
 
    cost = tot_in / 1e6 * 2 + tot_out / 1e6 * 10
    print(f"\ntokens: {tot_in} in / {tot_out} out  |  approx cost ${cost:.2f}")
 
 
if __name__ == "__main__":
    main()
 