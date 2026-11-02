#!/usr/bin/env python3
"""
Student dropout analysis (Portfolio project #05)

Question: which factors are most associated with a student dropping out of a
higher-education programme, and how early can an at-risk student be flagged?

Run:  python analysis.py
Outputs: figures/*.png, outputs/*.csv, outputs/results.json

Definition used throughout
--------------------------
dropout = 1 if Target == "Dropout", else 0 (Graduate and Enrolled -> 0).
"Enrolled" means still studying at the end of the normal course length, so the
final outcome is unknown. A sensitivity check that drops them is run in the
modelling step.
"""
import json
import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "dataset.csv"
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
SEED = 42
results = {}

# --------------------------------------------------------------------------
# 1. Load + clean
# --------------------------------------------------------------------------
RENAME = {
    "Marital status": "marital_status",
    "Application mode": "application_mode",
    "Application order": "application_order",
    "Course": "course",
    "Daytime/evening attendance": "daytime",
    "Previous qualification": "prev_qualification",
    "Nacionality": "nationality",
    "Mother's qualification": "mother_qual",
    "Father's qualification": "father_qual",
    "Mother's occupation": "mother_occ",
    "Father's occupation": "father_occ",
    "Displaced": "displaced",
    "Educational special needs": "special_needs",
    "Debtor": "debtor",
    "Tuition fees up to date": "tuition_ok",
    "Gender": "male",  # 1 = male, 0 = female (original UCI coding)
    "Scholarship holder": "scholarship",
    "Age at enrollment": "age",
    "International": "international",
    "Unemployment rate": "unemployment",
    "Inflation rate": "inflation",
    "GDP": "gdp",
    "Target": "target",
}
for s, label in (("1st", "s1"), ("2nd", "s2")):
    for raw, short in (
        ("credited", "credited"),
        ("enrolled", "enrolled"),
        ("evaluations", "evaluations"),
        ("approved", "approved"),
        ("grade", "grade"),
        ("without evaluations", "without_eval"),
    ):
        RENAME[f"Curricular units {s} sem ({raw})"] = f"{label}_{short}"

df = pd.read_csv(DATA, encoding="utf-8-sig")
df.columns = df.columns.str.strip()
df = df.rename(columns=RENAME)
assert df.shape[1] == 35, df.shape

df["dropout"] = (df["target"] == "Dropout").astype(int)
for s in ("s1", "s2"):
    enrolled = df[f"{s}_enrolled"]
    df[f"{s}_rate"] = np.where(enrolled > 0, df[f"{s}_approved"] / enrolled.replace(0, np.nan), np.nan)

# Course codes are re-numbered 1-17 in this copy of the dataset. Names below are
# an INFERENCE (sorted order of the original UCI programme codes). Check: the
# only programmes with daytime == 0 are codes 3 and 17, which are the two
# evening programmes in the original data.
COURSE_NAMES = {
    1: "Biofuel Production Technologies",
    2: "Animation & Multimedia Design",
    3: "Social Service (evening)",
    4: "Agronomy",
    5: "Communication Design",
    6: "Veterinary Nursing",
    7: "Informatics Engineering",
    8: "Equinculture",
    9: "Management",
    10: "Social Service",
    11: "Tourism",
    12: "Nursing",
    13: "Oral Hygiene",
    14: "Advertising & Marketing Mgmt",
    15: "Journalism & Communication",
    16: "Basic Education",
    17: "Management (evening)",
}
evening_codes = sorted(df.loc[df["daytime"] == 0, "course"].unique().tolist())

# --------------------------------------------------------------------------
# 2. Data quality
# --------------------------------------------------------------------------
n = len(df)
results["data_quality"] = {
    "rows": int(n),
    "columns": int(df.shape[1] - 3),  # minus dropout + 2 derived rates
    "missing_cells": int(df.drop(columns=["s1_rate", "s2_rate"]).isna().sum().sum()),
    "duplicate_rows": int(df.drop(columns=["dropout", "s1_rate", "s2_rate"]).duplicated().sum()),
    "target_counts": df["target"].value_counts().to_dict(),
    "dropout_rate_pct": round(df["dropout"].mean() * 100, 1),
    "evening_course_codes": [int(c) for c in evening_codes],
    "s1_no_units_enrolled": int((df["s1_enrolled"] == 0).sum()),
    "s2_no_units_enrolled": int((df["s2_enrolled"] == 0).sum()),
    "no_units_same_students_both_sems": bool(((df["s1_enrolled"] == 0) == (df["s2_enrolled"] == 0)).all()),
    "age_min_max": [int(df["age"].min()), int(df["age"].max())],
    "approved_gt_enrolled_s1": int((df["s1_approved"] > df["s1_enrolled"]).sum()),
    "approved_gt_enrolled_s2": int((df["s2_approved"] > df["s2_enrolled"]).sum()),
}
overall = df["dropout"].mean() * 100
print("== Data quality ==")
print(json.dumps(results["data_quality"], indent=2))


# --------------------------------------------------------------------------
# 3. Dropout rate by factor
# --------------------------------------------------------------------------
def rate_rows(factor, series, order=None, labels=None):
    g = df.groupby(series, observed=True)["dropout"].agg(["sum", "count"])
    if order is not None:
        g = g.reindex(order).dropna()
    rows = []
    for level, r in g.iterrows():
        rows.append(
            {
                "factor": factor,
                "level": (labels or {}).get(level, level),
                "n": int(r["count"]),
                "dropouts": int(r["sum"]),
                "dropout_rate_pct": round(r["sum"] / r["count"] * 100, 1),
            }
        )
    return rows


yn = {0: "No", 1: "Yes"}
age_bins = pd.cut(
    df["age"], [0, 19, 22, 25, 30, 40, 100], labels=["≤19", "20–22", "23–25", "26–30", "31–40", "41+"]
)


def approval_bin(rate, enrolled):
    return np.select(
        [enrolled == 0, rate == 0, rate < 0.5, rate < 0.75, rate < 1],
        ["No units enrolled", "0%", "1–49%", "50–74%", "75–99%"],
        default="100%",
    )


APPROVAL_ORDER = ["No units enrolled", "0%", "1–49%", "50–74%", "75–99%", "100%"]
df["s1_rate_bin"] = approval_bin(df["s1_rate"], df["s1_enrolled"])
df["s2_rate_bin"] = approval_bin(df["s2_rate"], df["s2_enrolled"])

rows = []
rows += rate_rows("Tuition fees up to date", "tuition_ok", [1, 0], yn)
rows += rate_rows("Debtor", "debtor", [0, 1], yn)
rows += rate_rows("Scholarship holder", "scholarship", [1, 0], yn)
rows += rate_rows("Gender", "male", [0, 1], {0: "Female", 1: "Male"})
rows += rate_rows("Age at enrollment", age_bins, ["≤19", "20–22", "23–25", "26–30", "31–40", "41+"])
rows += rate_rows("Displaced", "displaced", [0, 1], yn)
rows += rate_rows("Daytime attendance", "daytime", [1, 0], {1: "Daytime", 0: "Evening"})
rows += rate_rows("International", "international", [0, 1], yn)
rows += rate_rows("Special needs", "special_needs", [0, 1], yn)
rows += rate_rows("Sem 1 approval rate", "s1_rate_bin", APPROVAL_ORDER)
rows += rate_rows("Sem 2 approval rate", "s2_rate_bin", APPROVAL_ORDER)
rows += rate_rows("Course", "course", sorted(df["course"].unique()), COURSE_NAMES)
rates = pd.DataFrame(rows)
rates.to_csv(OUT / "dropout_rate_by_factor.csv", index=False)
print("\n== Dropout rate by factor (% of all students) ==")
print(rates[rates.factor != "Course"].to_string(index=False))

# Significance for the binary factors (chi-square on all students, dropout vs not)
sig = []
for col, name in [
    ("tuition_ok", "Tuition fees up to date"),
    ("debtor", "Debtor"),
    ("scholarship", "Scholarship holder"),
    ("male", "Male"),
    ("displaced", "Displaced"),
    ("daytime", "Daytime attendance"),
    ("international", "International"),
    ("special_needs", "Special needs"),
]:
    ct = pd.crosstab(df[col], df["dropout"])
    chi2, p, _, _ = stats.chi2_contingency(ct, correction=False)
    r1 = df.loc[df[col] == 1, "dropout"].mean() * 100
    r0 = df.loc[df[col] == 0, "dropout"].mean() * 100
    sig.append(
        {
            "factor": name,
            "dropout_rate_if_1_pct": round(r1, 1),
            "dropout_rate_if_0_pct": round(r0, 1),
            "diff_pp": round(r1 - r0, 1),
            "chi2_p_value": float(p),
        }
    )
sig = pd.DataFrame(sig)
sig.to_csv(OUT / "binary_factor_tests.csv", index=False)
print("\n== Binary factor tests ==")
print(sig.to_string(index=False))

# Macro-economic indicators are cohort-level (a handful of distinct values)
macro = df.groupby(["unemployment", "inflation", "gdp"])["dropout"].agg(["mean", "count"]).reset_index()
results["macro"] = {
    "distinct_combinations": int(len(macro)),
    "corr_with_dropout": {
        c: round(float(df[c].corr(df["dropout"])), 3) for c in ("unemployment", "inflation", "gdp")
    },
}
print("\n== Macro ==")
print(results["macro"])

# --------------------------------------------------------------------------
# 4. How much does each variable tell on its own? (single-feature CV AUC)
# --------------------------------------------------------------------------
NOMINAL = [
    "marital_status", "application_mode", "course", "prev_qualification",
    "nationality", "mother_qual", "father_qual", "mother_occ", "father_occ",
]
BACKGROUND = [
    "marital_status", "application_mode", "application_order", "course", "daytime",
    "prev_qualification", "nationality", "mother_qual", "father_qual", "mother_occ",
    "father_occ", "displaced", "special_needs", "male", "age", "international",
]
FINANCE = ["scholarship", "debtor", "tuition_ok"]
SEM1 = ["s1_credited", "s1_enrolled", "s1_evaluations", "s1_approved", "s1_grade", "s1_without_eval", "s1_rate"]
SEM2 = ["s2_credited", "s2_enrolled", "s2_evaluations", "s2_approved", "s2_grade", "s2_without_eval", "s2_rate"]
MACRO = ["unemployment", "inflation", "gdp"]

GROUP = {}
for c in BACKGROUND:
    GROUP[c] = "Background"
for c in FINANCE:
    GROUP[c] = "Financial"
for c in SEM1:
    GROUP[c] = "Semester 1"
for c in SEM2:
    GROUP[c] = "Semester 2"
for c in MACRO:
    GROUP[c] = "Economy"

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def make_model():
    return HistGradientBoostingClassifier(
        max_depth=3, max_iter=150, learning_rate=0.08,
        categorical_features="from_dtype", random_state=SEED,
    )


def prep(frame, cols):
    X = frame[cols].copy()
    for c in cols:
        if c in NOMINAL:
            X[c] = X[c].astype("category")
    return X


def cv_auc(frame, cols, y):
    scores = cross_val_score(make_model(), prep(frame, cols), y, cv=cv, scoring="roc_auc")
    return float(scores.mean()), float(scores.std())


y_all = df["dropout"]
single = []
for c in BACKGROUND + FINANCE + SEM1 + SEM2 + MACRO:
    m, s = cv_auc(df, [c], y_all)
    single.append({"feature": c, "group": GROUP[c], "cv_auc": round(m, 3), "cv_auc_sd": round(s, 3)})
single = pd.DataFrame(single).sort_values("cv_auc", ascending=False)
single.to_csv(OUT / "single_feature_auc.csv", index=False)
print("\n== Single-feature CV AUC (top 15) ==")
print(single.head(15).to_string(index=False))

# --------------------------------------------------------------------------
# 5. How early can we flag risk? (stage-wise models)
# --------------------------------------------------------------------------
STAGES = [
    ("Background only (at enrollment)", BACKGROUND),
    ("+ Financial status", BACKGROUND + FINANCE),
    ("+ Semester 1 results", BACKGROUND + FINANCE + SEM1),
    ("+ Semester 2 results", BACKGROUND + FINANCE + SEM1 + SEM2),
]
stage_rows = []
grad_mask = df["target"] != "Enrolled"
for name, cols in STAGES:
    m, s = cv_auc(df, cols, y_all)
    m2, s2 = cv_auc(df[grad_mask], cols, df.loc[grad_mask, "dropout"])
    stage_rows.append(
        {
            "stage": name,
            "n_features": len(cols),
            "cv_auc": round(m, 3),
            "cv_auc_sd": round(s, 3),
            "cv_auc_dropout_vs_graduate": round(m2, 3),
        }
    )
stages = pd.DataFrame(stage_rows)
stages.to_csv(OUT / "stage_models.csv", index=False)
print("\n== Stage-wise models (5-fold CV AUC) ==")
print(stages.to_string(index=False))

# --------------------------------------------------------------------------
# 6. Adjusted odds ratios (logistic regression, bootstrap 95% CI)
# --------------------------------------------------------------------------
Xl = pd.DataFrame(index=df.index)
Xl["Tuition fees not up to date"] = 1 - df["tuition_ok"]
Xl["Debtor"] = df["debtor"]
Xl["Scholarship holder"] = df["scholarship"]
Xl["Male"] = df["male"]
Xl["Aged 23+ at enrollment"] = (df["age"] >= 23).astype(int)
Xl["Displaced"] = df["displaced"]
Xl["Sem 1 approval rate (per +10 pp)"] = df["s1_rate"].fillna(0) * 10
Xl["Sem 2 approval rate (per +10 pp)"] = df["s2_rate"].fillna(0) * 10
report_cols = list(Xl.columns)
# Control only (not reported): the 180 students with no units enrolled have an
# undefined approval rate (filled with 0). Evening programme is not added
# because it is fully determined by the course dummies.
Xl["_no_units_enrolled"] = (df["s1_enrolled"] == 0).astype(int)
course_dummies = pd.get_dummies(df["course"], prefix="course", drop_first=True).astype(int)
Xfull = pd.concat([Xl, course_dummies], axis=1).values.astype(float)
yv = y_all.values


def fit_or(X, y):
    lr = LogisticRegression(C=1e6, max_iter=5000)
    lr.fit(X, y)
    return lr.coef_[0][: len(report_cols)]


coef = fit_or(Xfull, yv)
rng = np.random.default_rng(SEED)
boots = []
for _ in range(300):
    idx = rng.integers(0, n, n)
    boots.append(fit_or(Xfull[idx], yv[idx]))
boots = np.array(boots)
or_tbl = pd.DataFrame(
    {
        "predictor": report_cols,
        "odds_ratio": np.exp(coef).round(2),
        "ci_low": np.exp(np.percentile(boots, 2.5, axis=0)).round(2),
        "ci_high": np.exp(np.percentile(boots, 97.5, axis=0)).round(2),
    }
)
or_tbl.to_csv(OUT / "adjusted_odds_ratios.csv", index=False)
print("\n== Adjusted odds ratios (course-adjusted) ==")
print(or_tbl.to_string(index=False))

# --------------------------------------------------------------------------
# 7. A simple, explainable early-warning rule (descriptive, in-sample)
# --------------------------------------------------------------------------
def rule_stats(flag, label):
    flagged = int(flag.sum())
    tp = int((flag & (df["dropout"] == 1)).sum())
    return {
        "rule": label,
        "flagged_students": flagged,
        "flagged_share_pct": round(flagged / n * 100, 1),
        "precision_pct": round(tp / flagged * 100, 1) if flagged else None,
        "recall_pct": round(tp / df["dropout"].sum() * 100, 1),
    }


rules = [
    rule_stats(df["tuition_ok"] == 0, "Tuition fees not up to date"),
    rule_stats(df["s1_approved"] <= 2, "Sem 1: ≤2 units approved"),
    rule_stats((df["tuition_ok"] == 0) | (df["s1_approved"] <= 2), "Either of the above"),
    rule_stats(
        (df["tuition_ok"] == 0) | (df["s1_approved"] <= 2) | (df["s2_approved"] <= 2),
        "Either, or Sem 2: ≤2 units approved",
    ),
]
rules = pd.DataFrame(rules)
rules.to_csv(OUT / "early_warning_rules.csv", index=False)
print("\n== Early-warning rules ==")
print(rules.to_string(index=False))

# --------------------------------------------------------------------------
# 8. Figures
# --------------------------------------------------------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BLUE, DIM = "#e1e0d9", "#c3c2b7", "#2a78d6", "#c3c2b7"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.linewidth": 1.0,
    }
)


def style_hbar(ax, xmax, ticks, fmt):
    ax.set_xlim(0, xmax)
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt.format(t) for t in ticks])
    ax.xaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0)
    ax.tick_params(axis="y", labelsize=9.5)


def hbars(ax, labels, values, colors=None, label_fmt="{:.0f}%", xmax=105, ticks=(0, 25, 50, 75, 100),
          tick_fmt="{:.0f}%", height=0.5):
    y = np.arange(len(labels))
    ax.barh(y, values, height=height, color=colors or BLUE, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_ylim(len(labels) - 0.5, -0.5)
    style_hbar(ax, xmax, list(ticks), tick_fmt)
    for yi, v in zip(y, values):
        ax.text(v + xmax * 0.012, yi, label_fmt.format(v), va="center", ha="left", color=INK2, fontsize=9.5, zorder=4,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))


def ref_line(ax, value, text=None):
    ax.axvline(value, color=MUTED, linewidth=1.0, zorder=2)
    if text:
        ax.text(value + 1, -0.62, text, color=MUTED, fontsize=8.5, va="bottom", ha="left")


def get_items(factor):
    sub = rates[rates.factor == factor]
    labels = [f"{r.level}  (n={r.n:,})" for r in sub.itertuples()]
    return labels, sub["dropout_rate_pct"].tolist()


# Figure 1: dropout rate by headline factors ---------------------------------
panels = [
    ("Tuition fees up to date", "Tuition fees up to date"),
    ("Debtor", "Debtor"),
    ("Scholarship holder", "Scholarship holder"),
    ("Gender", "Gender"),
    ("Age at enrollment", "Age at enrollment"),
    ("Displaced", "Displaced"),
]
fig, axes = plt.subplots(2, 3, figsize=(13, 6.8), gridspec_kw={"hspace": 0.55, "wspace": 0.75})
fig.subplots_adjust(top=0.84)
for ax, (factor, title) in zip(axes.ravel(), panels):
    labels, vals = get_items(factor)
    hbars(ax, labels, vals)
    ref_line(ax, overall)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
fig.suptitle("Dropout rate by student profile", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.975)
fig.text(0.01, 0.915,
         f"Share of students in each group who dropped out. Grey line = all students ({overall:.1f}%, n={n:,}).",
         color=INK2, fontsize=10, ha="left")
fig.savefig(FIG / "01_dropout_by_profile.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# Figure 2: academic performance ---------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, (factor, title) in zip(axes, [("Sem 1 approval rate", "Semester 1"), ("Sem 2 approval rate", "Semester 2")]):
    sub = rates[rates.factor == factor].set_index("level").reindex(APPROVAL_ORDER)
    x = np.arange(len(sub))
    ax.bar(x, sub["dropout_rate_pct"], width=0.55, color=BLUE, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{lvl.replace('No units enrolled', 'No units' + chr(10) + 'enrolled')}\nn={int(r.n):,}" for lvl, r in sub.iterrows()],
        fontsize=9, color=INK2,
    )
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f"{t}%" for t in (0, 25, 50, 75, 100)])
    ax.yaxis.grid(True, color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.axhline(overall, color=MUTED, linewidth=1.0, zorder=2)
    for xi, v in zip(x, sub["dropout_rate_pct"]):
        ax.text(xi, v + 1.8, f"{v:.0f}%", ha="center", va="bottom", color=INK2, fontsize=9.5, zorder=4)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
    ax.set_xlabel("Share of enrolled units approved", color=INK2, fontsize=9.5, labelpad=8)
axes[0].text(len(APPROVAL_ORDER) - 0.5, overall + 1.5, f"all students {overall:.0f}%", color=MUTED, fontsize=8.5, ha="right", va="bottom")
fig.suptitle("Dropout rate by share of units approved", x=0.09, ha="left", fontsize=14, fontweight="bold", color=INK, y=1.03)
fig.savefig(FIG / "02_dropout_by_approval_rate.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# Figure 3: dropout rate by course -------------------------------------------
sub = rates[rates.factor == "Course"].sort_values("dropout_rate_pct", ascending=False)
labels = [f"{r.level}  (n={r.n:,})" for r in sub.itertuples()]
colors = [BLUE if r.n >= 50 else DIM for r in sub.itertuples()]
fig, ax = plt.subplots(figsize=(9.5, 6.4))
hbars(ax, labels, sub["dropout_rate_pct"].tolist(), colors=colors, height=0.55)
ref_line(ax, overall, f"all students {overall:.0f}%")
fig.suptitle("Dropout rate by programme", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.98)
fig.text(0.01, 0.925, "Grey bar = fewer than 50 students (too small to read much into). Programme names inferred, see README.",
         color=INK2, fontsize=9.5, ha="left")
fig.subplots_adjust(top=0.88)
fig.savefig(FIG / "03_dropout_by_course.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# Figure 4: single-feature AUC -------------------------------------------------
NICE = {
    "s2_approved": "Sem 2 units approved", "s2_grade": "Sem 2 average grade", "s2_rate": "Sem 2 approval rate",
    "s1_approved": "Sem 1 units approved", "s1_grade": "Sem 1 average grade", "s1_rate": "Sem 1 approval rate",
    "s2_evaluations": "Sem 2 evaluations", "s1_evaluations": "Sem 1 evaluations",
    "tuition_ok": "Tuition fees up to date", "debtor": "Debtor", "scholarship": "Scholarship holder",
    "age": "Age at enrollment", "male": "Gender", "course": "Programme", "application_mode": "Application mode",
    "s2_enrolled": "Sem 2 units enrolled", "s1_enrolled": "Sem 1 units enrolled", "displaced": "Displaced",
    "s2_credited": "Sem 2 units credited", "s1_credited": "Sem 1 units credited",
    "daytime": "Daytime / evening", "prev_qualification": "Previous qualification",
    "marital_status": "Marital status", "application_order": "Application order",
    "mother_qual": "Mother's qualification", "father_qual": "Father's qualification",
    "mother_occ": "Mother's occupation", "father_occ": "Father's occupation",
    "s1_without_eval": "Sem 1 units w/o evaluation", "s2_without_eval": "Sem 2 units w/o evaluation",
    "nationality": "Nationality", "international": "International", "special_needs": "Special needs",
    "unemployment": "Unemployment rate", "inflation": "Inflation rate", "gdp": "GDP",
}
top = single.head(12)
fig, ax = plt.subplots(figsize=(9, 5.4))
labels = [f"{NICE[r.feature]}  ·  {r.group}" for r in top.itertuples()]
y = np.arange(len(top))
ax.barh(y, top["cv_auc"] - 0.5, left=0.5, height=0.5, color=BLUE, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_ylim(len(top) - 0.5, -0.5)
ax.set_xlim(0.5, 0.92)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.xaxis.grid(True, color=GRID, linewidth=1.0)
ax.set_axisbelow(True)
ax.tick_params(length=0)
for yi, v in zip(y, top["cv_auc"]):
    ax.text(v + 0.006, yi, f"{v:.2f}", va="center", color=INK2, fontsize=9.5, zorder=4)
ax.set_xlabel("5-fold cross-validated AUC using this one variable alone (0.50 = no better than chance)", color=INK2, fontsize=9.5, labelpad=8)
fig.suptitle("Which single variables predict dropout best?", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.99)
fig.savefig(FIG / "04_single_feature_auc.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# Figure 5: stage-wise AUC ------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.6))
y = np.arange(len(stages))
ax.barh(y, stages["cv_auc"] - 0.5, left=0.5, height=0.45, color=BLUE, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(stages["stage"])
ax.set_ylim(len(stages) - 0.5, -0.5)
ax.set_xlim(0.5, 1.0)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ax.xaxis.grid(True, color=GRID, linewidth=1.0)
ax.set_axisbelow(True)
ax.tick_params(length=0)
for yi, r in zip(y, stages.itertuples()):
    ax.text(r.cv_auc + 0.008, yi, f"{r.cv_auc:.2f}", va="center", color=INK2, fontsize=9.5, zorder=4)
ax.set_xlabel("5-fold cross-validated AUC (0.50 = no better than chance)", color=INK2, fontsize=9.5, labelpad=8)
fig.suptitle("How early can an at-risk student be flagged?", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK, y=1.0)
fig.savefig(FIG / "05_stage_model_auc.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# Figure 6: adjusted odds ratios -----------------------------------------------
tbl = or_tbl.sort_values("odds_ratio", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9.5, 5.6))
y = np.arange(len(tbl))
for yi, r in zip(y, tbl.itertuples()):
    ax.plot([r.ci_low, r.ci_high], [yi, yi], color=BLUE, linewidth=2, solid_capstyle="round", zorder=3)
ax.plot(tbl["odds_ratio"], y, "o", markersize=8.5, color=BLUE, markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
ax.axvline(1, color=MUTED, linewidth=1.0, zorder=2)
ax.set_xscale("log")
ticks = [0.1, 0.25, 0.5, 1, 2, 4, 8, 16]
lo = min(0.08, tbl["ci_low"].min() * 0.8)
hi = tbl["ci_high"].max() * 1.25
ax.set_xlim(lo, hi)
ax.set_xticks([t for t in ticks if lo <= t <= hi])
ax.set_xticklabels([str(t) for t in ticks if lo <= t <= hi])
ax.minorticks_off()
ax.set_yticks(y)
ax.set_yticklabels(tbl["predictor"])
ax.set_ylim(len(tbl) - 0.5, -0.5)
ax.xaxis.grid(True, color=GRID, linewidth=1.0)
ax.set_axisbelow(True)
ax.tick_params(length=0)
for yi, r in zip(y, tbl.itertuples()):
    ax.text(1.02, yi, f"OR {r.odds_ratio:.2f}   ({r.ci_low:.2f}–{r.ci_high:.2f})", va="center", color=INK2,
            fontsize=9.5, transform=ax.get_yaxis_transform(), clip_on=False)
ax.set_xlabel("Adjusted odds ratio of dropping out (log scale; 1 = no effect; left of 1 = protective)", color=INK2, fontsize=9.5, labelpad=8)
fig.suptitle("Effect of each factor with the others held fixed", x=0.01, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.99)
fig.text(0.01, 0.935, "Logistic regression, also adjusted for programme; dots = estimate, lines = bootstrap 95% interval.", color=INK2, fontsize=9.5, ha="left")
fig.subplots_adjust(top=0.88)
fig.savefig(FIG / "06_adjusted_odds_ratios.png", dpi=170, bbox_inches="tight")
plt.close(fig)

# --------------------------------------------------------------------------
# 9. Save machine-readable results
# --------------------------------------------------------------------------
results["stage_models"] = stages.to_dict(orient="records")
results["top_single_features"] = single.head(10).to_dict(orient="records")
results["adjusted_odds_ratios"] = or_tbl.to_dict(orient="records")
results["early_warning_rules"] = rules.to_dict(orient="records")
with open(OUT / "results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False, default=str)
print("\nDone. Figures ->", FIG, "| tables ->", OUT)
