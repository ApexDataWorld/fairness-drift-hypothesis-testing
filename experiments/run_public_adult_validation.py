"""Optional public-data validation using the Adult dataset from OpenML.

This script requires internet access on first run. If the dataset cannot be
fetched, the script exits gracefully and writes an explanatory note. It does not
make production-data claims.
"""
from __future__ import annotations

import os
from pathlib import Path
from math import sqrt

_MPLCONFIGDIR = Path(os.environ.get("MPLCONFIGDIR", "results/.matplotlib")).resolve()
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
_XDG_CACHE_HOME = Path(os.environ.get("XDG_CACHE_HOME", "results/.cache")).resolve()
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.datasets import fetch_openml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fairness_drift.alert_rules import benjamini_hochberg

OUT_TABLES = Path("results/tables")
OUT_FIGS = Path("results/figures")
DATA_HOME = Path("data/public_cache")
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)
DATA_HOME.mkdir(parents=True, exist_ok=True)


def two_prop_pvalue(x1, n1, x2, n2):
    p_pool = (x1 + x2) / (n1 + n2)
    se = sqrt(max(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2), 1e-12))
    z = ((x2 / n2) - (x1 / n1)) / se
    return 2 * norm.sf(abs(z))


def main():
    try:
        data = fetch_openml("adult", version=2, as_frame=True, data_home=str(DATA_HOME))
    except Exception as exc:
        note = OUT_TABLES / "public_adult_validation_NOT_RUN.txt"
        note.write_text(
            "Adult public-data validation was not run because the dataset could not be fetched.\n"
            f"Reason: {exc}\nRun this script in an environment with internet access.\n"
        )
        print(note.read_text())
        return
    df = data.frame.copy()
    # Adult uses target class labels like <=50K and >50K.
    target = data.target.name if hasattr(data, "target") and data.target is not None else "class"
    if target not in df.columns:
        # common fallback
        target = df.columns[-1]
    df["favorable"] = df[target].astype(str).str.contains(">50K", regex=False).astype(int)
    group_col = "sex" if "sex" in df.columns else "race"
    ref_group = df[group_col].mode().iloc[0]
    alt_group = [g for g in df[group_col].dropna().unique() if g != ref_group][0]
    df = df[df[group_col].isin([ref_group, alt_group])].sample(frac=1.0, random_state=42).reset_index(drop=True)
    df["window"] = pd.qcut(np.arange(len(df)), 8, labels=False)
    baseline_windows = [0, 1]
    rows = []
    for w, sub in df.groupby("window"):
        a = sub[sub[group_col].eq(ref_group)]
        b = sub[sub[group_col].eq(alt_group)]
        if len(a) == 0 or len(b) == 0:
            continue
        rate_a, rate_b = a["favorable"].mean(), b["favorable"].mean()
        rows.append({
            "window": int(w),
            "reference_group": ref_group,
            "monitored_group": alt_group,
            "n_reference": int(len(a)),
            "n_monitored": int(len(b)),
            "selection_rate_reference": rate_a,
            "selection_rate_monitored": rate_b,
            "gap": rate_b - rate_a,
            "p_value": two_prop_pvalue(a["favorable"].sum(), len(a), b["favorable"].sum(), len(b)),
        })
    out = pd.DataFrame(rows)
    baseline_gap = out[out["window"].isin(baseline_windows)]["gap"].mean()
    out["baseline_gap"] = baseline_gap
    out["drift"] = out["gap"] - baseline_gap
    out["p_adj_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    out["alert"] = (out["p_adj_bh"] < 0.05) & (out["drift"].abs() > 0.03)
    out.to_csv(OUT_TABLES / "public_adult_validation.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(out["window"], out["gap"], marker="o", label="Selection parity gap")
    plt.axhline(baseline_gap, linestyle="--", label="Baseline gap")
    plt.xlabel("Simulated monitoring window")
    plt.ylabel("Monitored minus reference selection rate")
    plt.title("Public Adult dataset fairness gap by window")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "public_adult_fairness_gap_by_window.png", dpi=300)
    plt.close()
    print("Wrote public Adult validation outputs.")


if __name__ == "__main__":
    main()
