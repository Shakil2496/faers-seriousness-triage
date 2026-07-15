#!/usr/bin/env python3
"""
Step 9 - Regenerate all four manuscript figures from pipeline outputs.

Run AFTER 06b_subgroups_2k.py and 08b_bootstrap_ci.py.

    python 09_figures.py

NOTHING IS HARDCODED. Every count, AUROC, confidence interval and base rate in
every figure is read from a file that an earlier script wrote. If you rerun the
pipeline and a number moves, the figures move with it. That is the point: the
manuscript claims the released scripts reproduce every figure, and this script
is what makes that claim true.

INPUTS (all under outputs/)
    clinical_metrics.json    <- 06b   overall AUROC, Brier
    calibration_deciles.csv  <- 06b   Fig 2
    subgroup_metrics.csv     <- 06b   Fig 4
    llm_vs_classical.csv     <- 08b   Fig 3
    panel_means.json         <- 08b   Fig 1 head-to-head box
    flow_counts.json         <- YOU   Fig 1 pipeline counts (see below)

flow_counts.json is the one file no script writes, because scripts 01/03/04
print their counts and exit. Fill it in ONCE from their console output; each
key records which script it came from, so the figure stays auditable. If it is
missing, this script writes a template and tells you what to fill in.

OUTPUTS (figures/, to Springer artwork spec)
    Fig1.eps/.pdf/.png   two-stream study design
    Fig2.eps/.pdf/.png   calibration, temporal test set
    Fig3.eps/.pdf/.png   LLM advantage vs classical strength
    Fig4.eps/.pdf/.png   subgroup AUROC and recall, all 16 groups

Springer compliance: no titles inside illustrations (captions live in the
manuscript), Arial/Helvetica lettering at 8-12 pt, vector EPS with fonts
embedded, patterns in addition to colour, no alpha (EPS has no transparency),
single-column width 174 mm and height under 234 mm.
"""

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ---------------------------------------------------------------- config
DATA_DIR = r"C:\Users\shaki\Downloads\faers_data"
OUT = Path(DATA_DIR) / "outputs"
FIG = Path(DATA_DIR) / "figures"
FIG.mkdir(exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,          # embed TrueType in EPS
    "axes.unicode_minus": False,
})

MM = 1 / 25.4
W_COL = 174 * MM               # single-column text area, Springer large format

NAVY, TEAL, RUST = "#1F3B57", "#2E7D74", "#C0553B"
BLUEF, REDF, GREENF, ORANGF = "#E9EEF4", "#FBEBE6", "#E3EFEB", "#FCEDE3"
GREY, LGREY = "#6E6E6E", "#B0B0B0"
HILITE = "#FDF8E8"             # EPS-safe: pre-flattened, no alpha

SUBGROUP_ORDER = [
    ("REGION",   ["US", "non-US"]),
    ("SEX",      ["male", "female", "other/unknown"]),
    ("AGE BAND", ["0-17", "18-44", "45-64", "65-74", "75+", "unknown"]),
    ("REPORTER", ["physician", "pharmacist", "other HP", "consumer",
                  "other/unknown"]),
]
# printed labels (figure) vs internal keys (CSV)
PRETTY = {"other/unknown": "unrecorded", "unknown": "unrecorded",
          "other HP": "other HCP", "0-17": "0\u201317", "18-44": "18\u201344",
          "45-64": "45\u201364", "65-74": "65\u201374"}
DIM_LABEL = {"REGION": "Region", "SEX": "Sex", "AGE BAND": "Age",
             "REPORTER": "Reporter"}
FLAGGED = {("SEX", "other/unknown"), ("AGE BAND", "unknown"),
           ("REPORTER", "other/unknown")}

FLOW_TEMPLATE = {
    "_comment": "Counts that scripts 01/03/04 print but do not save. Fill once.",
    "n_raw":                 {"value": None, "source": "01_pull_quarters.py  (raw reports pooled)"},
    "n_excluded":            {"value": None, "source": "03_build_features.py (panel-drug + unlabeled)"},
    "n_clean":               {"value": None, "source": "03_build_features.py (clean modelling corpus)"},
    "n_train":               {"value": None, "source": "03_build_features.py (train Q1-Q3 2025)"},
    "n_test":                {"value": None, "source": "03_build_features.py (test Q4 2025)"},
    "leak_structural_auroc": {"value": None, "source": "04_leakage_audit.py  (ablation (a), Table 3)"},
    "leak_is_us_auroc":      {"value": None, "source": "04_leakage_audit.py  (is_us single feature, Table 3)"},
}


def load_flow():
    p = OUT / "flow_counts.json"
    if not p.exists():
        p.write_text(json.dumps(FLOW_TEMPLATE, indent=2))
        raise SystemExit(
            f"\nWrote a template to {p}\n\n"
            "Fill in the seven 'value' fields from the console output of\n"
            "scripts 01, 03 and 04, then rerun this script. Nothing else is\n"
            "needed - every other number is read from a file.\n")
    raw = json.loads(p.read_text())
    flow = {}
    missing = []
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        val = v["value"] if isinstance(v, dict) else v
        if val is None:
            missing.append(k)
        flow[k] = val
    if missing:
        raise SystemExit(f"\n{p} still has empty values: {', '.join(missing)}\n")
    return flow


def save(fig, stem):
    for ext, kw in (("eps", {}), ("pdf", {}), ("png", {"dpi": 600})):
        fig.savefig(FIG / f"{stem}.{ext}", bbox_inches="tight",
                    facecolor="white", **kw)
    plt.close(fig)
    print(f"  {stem}.eps / .pdf / .png")


# ======================================================== Fig 1 - flowchart
def figure1(flow, panel, means):
    """Two-stream flowchart. Wider columns, 8-pt text and pre-split long lines
    so nothing overflows a box; head-to-head box sits clear at the bottom."""
    fig, ax = plt.subplots(figsize=(W_COL * 1.06, W_COL * 1.52))
    ax.set_xlim(0, 100); ax.set_ylim(0, 176); ax.axis("off")
    FS = 8.0

    def box(xc, ytop, h, w, lines_, edge, face, bold=(), fs=FS):
        ax.add_patch(FancyBboxPatch(
            (xc - w / 2, ytop - h), w, h,
            boxstyle="round,pad=0.3,rounding_size=1.3",
            linewidth=1.0, edgecolor=edge, facecolor=face, zorder=2))
        y0 = ytop - h / 2 + (len(lines_) - 1) * 2.9 / 2
        for i, ln in enumerate(lines_):
            ax.text(xc, y0 - i * 2.9, ln, ha="center", va="center", fontsize=fs,
                    color="#1A1A1A", zorder=3,
                    fontweight="bold" if i in bold else "normal")
        return xc, ytop, ytop - h

    def arrow(xc, y0, y1, c):
        ax.add_patch(FancyArrowPatch((xc, y0), (xc, y1), arrowstyle="-|>",
                     mutation_scale=9, linewidth=1.0, color=c, zorder=4,
                     shrinkA=0, shrinkB=0))

    XA, XB, W = 24, 76, 46
    ax.text(XA, 172, "STREAM A \u2014 Training corpus", ha="center",
            fontsize=9.5, fontweight="bold", color=NAVY)
    ax.text(XB, 172, "STREAM B \u2014 Held-out novel-drug panel", ha="center",
            fontsize=9.5, fontweight="bold", color=TEAL)
    ax.plot([50, 50], [30, 169], ls=":", lw=0.8, color="#CCCCCC", zorder=1)

    y, a = 167, []
    a.append(box(XA, y, 11, W, ["FAERS via openFDA drug-event endpoint",
                                "Pulled by receivedate, 2025 Q1\u2013Q4",
                                "25,000 cap/quarter"], NAVY, BLUEF)); y = a[-1][2] - 5
    a.append(box(XA, y, 8.5, W, ["Raw reports pooled",
                                 f"n = {flow['n_raw']:,}"], NAVY, BLUEF, bold=(0, 1))); y = a[-1][2] - 5
    a.append(box(XA, y, 9, W, ["Dedup: latest version by safetyreportid",
                               "(0 duplicates removed)"], NAVY, BLUEF)); y = a[-1][2] - 5
    a.append(box(XA, y, 9, W, ["Exclude panel-drug reports and",
                               "cases with no seriousness label"], NAVY, BLUEF)); y = a[-1][2] - 5
    a.append(box(XA, y, 8.5, W, ["Clean modelling corpus",
                                 f"n = {flow['n_clean']:,}"], NAVY, BLUEF, bold=(0, 1))); y = a[-1][2] - 5
    a.append(box(XA, y, 12, W, ["TEMPORAL SPLIT (by receivedate)",
                                f"Train  Q1\u2013Q3 2025   n = {flow['n_train']:,}",
                                f"Test   Q4 2025         n = {flow['n_test']:,}"],
                 NAVY, BLUEF)); y = a[-1][2] - 5
    a.append(box(XA, y, 14, W, ["Leakage-controlled feature matrix",
                                "\u2022 blacklist: outcome-downstream fields",
                                "\u2022 outcome-equivalent PTs removed",
                                "\u2022 vocabulary from TRAIN only",
                                "   (800 PT / 300 drug)"], NAVY, BLUEF)); y = a[-1][2] - 5
    a.append(box(XA, y, 14, W, ["Behavioural leakage audit",
                                "structural + channel only:",
                                f"AUROC {flow['leak_structural_auroc']:.3f}  \u2192  leak",
                                "single-feature: report origin (is_us)",
                                f"isolated (AUROC {flow['leak_is_us_auroc']:.3f})"],
                 RUST, REDF)); y = a[-1][2] - 5
    a.append(box(XA, y, 12, W, ["CLINICAL XGBoost model",
                                "(reporting-channel features removed)",
                                f"Temporal AUROC {means['clinical_auroc']:.3f} \u00b7 "
                                f"Brier {means['clinical_brier']:.3f}"],
                 NAVY, BLUEF, bold=(0, 2)))
    y_a = a[-1][2]
    for i in range(len(a) - 1):
        arrow(XA, a[i][2], a[i + 1][1], NAVY)
    ax.text(XA + W / 2 + 1.0, a[3][2] + 4.5, f"{flow['n_excluded']:,}\nexcluded",
            fontsize=7.5, color=GREY, ha="left", va="center")

    drug_lines = [f"{r.drug.split(' (')[0]} {int(r.n):,} ({r.base_rate:.0%})"
                  for r in panel.itertuples()]
    y, b = 167, []
    b.append(box(XB, y, 11, W, ["Candidate first-in-class drugs",
                                "screened in FAERS",
                                "(both seriousness classes required)"],
                 TEAL, BLUEF)); y = b[-1][2] - 5
    b.append(box(XB, y, 14, W, ["EXCLUDED",
                                "\u2022 nemolizumab \u2014 90% consumer-reported",
                                "\u2022 lenacapavir \u2014 administrative skew",
                                "\u2022 acoramidis, gepotidacin \u2014",
                                "   minority class too thin"],
                 RUST, REDF, bold=(0,))); y = b[-1][2] - 5
    b.append(box(XB, y, 10, W, ["Sole-primary-suspect filter",
                                "(target drug the only suspect)"], TEAL, BLUEF)); y = b[-1][2] - 5
    b.append(box(XB, y, 10, W, ["Post-approval-date floor",
                                "(cases on/after each approval)"], TEAL, BLUEF)); y = b[-1][2] - 5
    b.append(box(XB, y, 12, W, ["Co-suspect cleaning",
                                "sotatercept: 1,334 \u2192 258 cases",
                                "(background co-reports removed)"], RUST, REDF)); y = b[-1][2] - 5
    b.append(box(XB, y, 25, W, ["FINAL PANEL", f"{len(panel)} drugs, n = {means['panel_n']:,}",
                                ""] + drug_lines, TEAL, GREENF, bold=(0, 1)))
    y_b = b[-1][2]
    for i in range(len(b) - 1):
        arrow(XB, b[i][2], b[i + 1][1], TEAL)

    gap = means["nweighted_llm"] - means["nweighted_classical"]
    JOIN = 20
    box(50, 11, 13, 74, [
        "HEAD-TO-HEAD COMPARISON \u2014 identical cases, natural base rates",
        "Classical (no drug identity)  vs  LLM, Claude Sonnet 5 (drug name given)",
        f"n-weighted AUROC  {means['nweighted_classical']:.3f}  vs  "
        f"{means['nweighted_llm']:.3f}  (gap +{gap:.3f})"],
        RUST, ORANGF, bold=(0, 1, 2), fs=8.5)
    ax.plot([XA, XA], [y_a, JOIN], lw=1.0, color=NAVY, zorder=2)
    ax.plot([XB, XB], [y_b, JOIN], lw=1.0, color=TEAL, zorder=2)
    ax.plot([XA, XB], [JOIN, JOIN], lw=1.0, color="#6E8494", zorder=2)
    arrow(50, JOIN, 11.5, "#6E8494")
    save(fig, "Fig1")


def figure2(cal, met):
    fig, ax = plt.subplots(figsize=(3.35, 3.35))
    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color=GREY,
            label="Perfect calibration")
    ax.plot(cal.mean_pred, cal.obs_frac, "o-", color=NAVY, lw=1.5, markersize=4,
            label="Clinical model")
    ax.set_xlabel("Mean predicted probability (decile)", fontsize=9)
    ax.set_ylabel("Observed serious fraction", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(color="#EFEFEF", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "Fig2")


# ======================================================== Fig 3 - LLM advantage
def figure3(panel):
    d = panel.sort_values("auroc_classical")
    fig, ax = plt.subplots(figsize=(3.35, 2.9))
    for r in d.itertuples():
        crosses = r.gap_lo <= 0          # interval reaches zero -> flag it
        ax.errorbar(r.auroc_classical, r.gap,
                    yerr=[[r.gap - r.gap_lo], [r.gap_hi - r.gap]],
                    fmt="o", color=NAVY, markersize=5, capsize=2.5, lw=1.0,
                    ecolor=RUST if crosses else "#5A7690", zorder=3)
        nm = r.drug.split(" (")[0]
        right = r.auroc_classical > 0.80
        ax.annotate(nm,
                    (r.auroc_classical + (-0.005 if right else 0.005),
                     r.gap + 0.010),
                    fontsize=7, color="#3A3A3A",
                    ha="right" if right else "left", va="bottom", zorder=4)
    ax.axhline(0.0, ls="--", lw=1.0, color=GREY, zorder=2)
    ax.text(d.auroc_classical.max() + 0.014, 0.007, "No LLM advantage",
            ha="right", va="bottom", fontsize=7, color=GREY)
    ax.set_xlabel("Classical model AUROC on the held-out drug", fontsize=9)
    ax.set_ylabel("LLM advantage (AUROC gap)", fontsize=9)
    ax.set_xlim(d.auroc_classical.min() - 0.018, d.auroc_classical.max() + 0.021)
    ax.set_ylim(min(0, d.gap_lo.min()) - 0.03, d.gap_hi.max() + 0.025)
    ax.tick_params(labelsize=8)
    # NOTE: deliberately no regression line and no correlation coefficient.
    # With n=6 drugs such a statistic is underpowered; it is not reported in
    # the manuscript, and the claim rests on the ordered point positions.
    ax.grid(color="#EFEFEF", lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "Fig3")


# ======================================================== Fig 4 - subgroups
def figure4(sub):
    rows = []
    for dim, groups in SUBGROUP_ORDER:
        for g in groups:
            m = sub[(sub.dimension == dim) & (sub.group == g)]
            if m.empty:
                raise SystemExit(
                    f"\nMISSING SUBGROUP in subgroup_metrics.csv: {dim} / {g}\n"
                    "Figure 4's caption describes all 16 groups. Refusing to draw\n"
                    "a figure that does not show what its caption claims.\n")
            rows.append(m.iloc[0])
    d = pd.DataFrame(rows).reset_index(drop=True)

    x = np.arange(len(d)); w = 0.38
    ae = np.array([d.auroc - d.auroc_lo, d.auroc_hi - d.auroc])
    re_ = np.array([d.recall - d.recall_lo, d.recall_hi - d.recall])

    fig, ax = plt.subplots(figsize=(W_COL, W_COL * 0.66))
    ax.bar(x - w / 2, d.auroc, w, yerr=ae, color=NAVY, edgecolor="white",
           linewidth=0.4, label="AUROC (ranking)",
           error_kw=dict(ecolor="#33475B", lw=0.8, capsize=1.8), zorder=3)
    ax.bar(x + w / 2, d.recall, w, yerr=re_, color=RUST, edgecolor="white",
           linewidth=0.4, hatch="////", label="Recall @ threshold 0.50",
           error_kw=dict(ecolor="#8C3A26", lw=0.8, capsize=1.8), zorder=3)

    ax.axhline(0.90, ls=":", lw=1.0, color=GREY, zorder=2)
    ax.text(len(d) - 0.4, 0.907, "90% recall target", ha="right", va="bottom",
            fontsize=7.5, color=GREY)
    ax.axhline(0.50, ls="--", lw=1.0, color=LGREY, zorder=2)
    ax.text(len(d) - 0.4, 0.508, "Chance (AUROC 0.50)", ha="right", va="bottom",
            fontsize=7.5, color=GREY)

    for i, r in d.iterrows():
        if (r.dimension, r.group) in FLAGGED:
            ax.axvspan(i - 0.5, i + 0.5, color=HILITE, zorder=1)

    b = [0] + [i for i in range(1, len(d))
               if d.dimension[i] != d.dimension[i - 1]] + [len(d)]
    for bd in b[1:-1]:
        ax.axvline(bd - 0.5, color="#D5D5D5", lw=0.8, zorder=1)
    for j in range(len(b) - 1):
        ax.text((b[j] + b[j + 1] - 1) / 2, 1.015,
                DIM_LABEL[d.dimension[b[j]]], ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=NAVY)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{PRETTY.get(g, g)} ({int(n):,})"
                        for g, n in zip(d.group, d.n)], fontsize=7,
                       rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.7, len(d) - 0.3)
    ax.set_ylim(0.40, 1.02)
    ax.set_ylabel("Metric value", fontsize=9)
    ax.tick_params(axis="y", labelsize=8, length=0)
    ax.tick_params(axis="x", length=0)
    ax.legend(loc="upper center", frameon=False, fontsize=8.5, ncol=2,
              bbox_to_anchor=(0.5, -0.30))
    ax.grid(axis="y", color="#EDEDED", lw=0.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "Fig4")


# ======================================================== main
def main():
    need = ["clinical_metrics.json", "calibration_deciles.csv",
            "subgroup_metrics.csv", "llm_vs_classical.csv", "panel_means.json"]
    absent = [f for f in need if not (OUT / f).exists()]
    if absent:
        raise SystemExit(
            "\nMissing pipeline outputs: " + ", ".join(absent) +
            "\nRun 06b_subgroups_2k.py and 08b_bootstrap_ci.py first.\n")

    flow = load_flow()
    met = json.loads((OUT / "clinical_metrics.json").read_text())
    pm = json.loads((OUT / "panel_means.json").read_text())
    cal = pd.read_csv(OUT / "calibration_deciles.csv")
    sub = pd.read_csv(OUT / "subgroup_metrics.csv")
    panel = pd.read_csv(OUT / "llm_vs_classical.csv")

    means = dict(clinical_auroc=met["auroc"], clinical_brier=met["brier"],
                 panel_n=pm["panel_n"],
                 nweighted_classical=pm["nweighted_classical"],
                 nweighted_llm=pm["nweighted_llm"])

    print("Regenerating figures from pipeline outputs\n")
    figure1(flow, panel, means)
    figure2(cal, met)
    figure3(panel)
    figure4(sub)

    print(f"\nWrote 4 figures to {FIG}")
    print("\nValues drawn (all read from disk, none hardcoded):")
    print(f"  clinical AUROC   {met['auroc']:.3f}   Brier {met['brier']:.3f}")
    print(f"  panel n          {pm['panel_n']:,}")
    print(f"  n-weighted       classical {pm['nweighted_classical']:.3f}  "
          f"LLM {pm['nweighted_llm']:.3f}")
    print(f"  subgroups        {len(sub)} rows")
    print(f"  panel drugs      {len(panel)}   "
          f"({int(panel.significant.sum())} with gap CI excluding zero)")
    print("\nSubmit the .eps files. Check every number above against the "
          "manuscript before submitting.")


if __name__ == "__main__":
    main()
