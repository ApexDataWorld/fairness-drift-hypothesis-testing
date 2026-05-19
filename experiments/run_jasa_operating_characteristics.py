"""Generate JASA-oriented operating-characteristic tables and figures.

Run from repository root:
    python experiments/run_jasa_operating_characteristics.py
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fairness_drift.alert_rules import SUPPORTED_RULES, apply_alert_rule
from fairness_drift.power import make_power_table

OUT_TABLES = Path("results/tables")
OUT_FIGS = Path("results/figures")
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)


def two_prop_pvalue(x1, n1, x2, n2):
    p_pool = (x1 + x2) / (n1 + n2)
    se = sqrt(max(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2), 1e-12))
    z = ((x2 / n2) - (x1 / n1)) / se
    return 2 * norm.sf(abs(z))


def run_power_grid(seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    n = 400
    p_a = 0.40
    windows = 8
    reps = 400
    drift_grid = [0.00, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]
    for drift in drift_grid:
        for rep in range(reps):
            pvals, drifts = [], []
            for w in range(windows):
                # Drift starts after two baseline windows and grows toward target drift.
                ramp = max(w - 1, 0) / max(windows - 2, 1)
                p_b = np.clip(p_a - drift * ramp, 0.001, 0.999)
                xa = rng.binomial(n, p_a)
                xb = rng.binomial(n, p_b)
                pval = two_prop_pvalue(xa, n, xb, n)
                observed_gap = xb / n - xa / n
                pvals.append(pval)
                drifts.append(observed_gap)
            replicate_results = pd.DataFrame(
                {
                    "window": np.arange(windows),
                    "p_value": pvals,
                    "drift": drifts,
                }
            )
            for rule in SUPPORTED_RULES:
                alerts = apply_alert_rule(
                    replicate_results,
                    rule=rule,
                    p_col="p_value",
                    effect_col="drift",
                    alpha=0.05,
                    delta=0.03,
                    persistence_windows=2,
                )["alert"].to_numpy()
                rows.append({
                    "true_drift": drift,
                    "replicate": rep,
                    "rule": rule,
                    "any_alert": bool(alerts.any()),
                    "first_alert_window": int(np.argmax(alerts)) if alerts.any() else np.nan,
                    "num_alert_windows": int(alerts.sum()),
                })
    df = pd.DataFrame(rows)
    summary = df.groupby(["true_drift", "rule"], as_index=False).agg(
        detection_rate=("any_alert", "mean"),
        mean_alert_windows=("num_alert_windows", "mean"),
        median_first_alert_window=("first_alert_window", "median"),
    )
    summary.to_csv(OUT_TABLES / "jasa_operating_characteristics.csv", index=False)
    return summary


def make_figures(summary):
    power = summary[summary["rule"].eq("adjusted_p_plus_materiality")]
    plt.figure(figsize=(7, 4.5))
    plt.plot(power["true_drift"], power["detection_rate"], marker="o")
    plt.xlabel("True fairness drift magnitude")
    plt.ylabel("Empirical detection probability")
    plt.title("Power curve for BH + materiality alert rule")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "power_by_drift_magnitude.png", dpi=300)
    plt.close()

    no_drift = summary[summary["true_drift"].eq(0.0)]
    plt.figure(figsize=(8, 4.5))
    plt.bar(no_drift["rule"], no_drift["detection_rate"])
    plt.xlabel("Alert rule")
    plt.ylabel("False-alert probability under no drift")
    plt.title("False-alert behavior by alert rule")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "false_alert_rate_by_rule.png", dpi=300)
    plt.close()

    power_table = make_power_table()
    power_table.to_csv(OUT_TABLES / "jasa_power_table.csv", index=False)
    plt.figure(figsize=(7, 4.5))
    plt.plot(power_table["n_per_group"], power_table["mde_80pct_power"], marker="o")
    plt.xscale("log")
    plt.xlabel("Subgroup n per monitoring window")
    plt.ylabel("Minimum detectable gap for 80% power")
    plt.title("Minimum detectable fairness gap by subgroup size")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "mde_by_subgroup_size.png", dpi=300)
    plt.close()


def main():
    summary = run_power_grid()
    make_figures(summary)
    print("Wrote JASA operating-characteristic outputs to results/tables and results/figures")


if __name__ == "__main__":
    main()
