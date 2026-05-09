from __future__ import annotations

import numpy as np
import pandas as pd


DEFAULT_SENSITIVITY_THRESHOLDS = (0.01, 0.02, 0.03, 0.05)


def baseline_adjusted_fairness_drift(
    fairness_metrics: pd.DataFrame,
    metric_col: str = "selection_parity_gap",
    baseline_windows: tuple[int, ...] = (0, 1),
) -> pd.DataFrame:
    """Compare each current fairness gap with the scenario baseline gap."""
    rows = []
    for scenario, part in fairness_metrics.groupby("scenario"):
        baseline_gap = part.loc[part["time_window"].isin(baseline_windows), metric_col].mean()
        for _, row in part.iterrows():
            current_gap = row[metric_col]
            rows.append(
                {
                    "scenario": scenario,
                    "time_window": int(row["time_window"]),
                    "metric": metric_col,
                    "baseline_gap": baseline_gap,
                    "current_gap": current_gap,
                    "baseline_adjusted_drift": current_gap - baseline_gap,
                    "absolute_drift": abs(current_gap - baseline_gap),
                }
            )
    return pd.DataFrame(rows)


def difference_in_differences_drift(
    fairness_metrics: pd.DataFrame,
    metric_col: str = "selection_parity_gap",
    baseline_windows: tuple[int, ...] = (0, 1),
) -> pd.DataFrame:
    """Backward-compatible alias for the baseline-adjusted drift statistic."""
    return baseline_adjusted_fairness_drift(fairness_metrics, metric_col, baseline_windows)


def summarize_drift(
    fairness_metrics: pd.DataFrame,
    multiple_testing_results: pd.DataFrame,
    baseline_windows: tuple[int, ...] = (0, 1),
    drift_threshold: float = 0.03,
) -> pd.DataFrame:
    drift = baseline_adjusted_fairness_drift(fairness_metrics, baseline_windows=baseline_windows)
    bh = multiple_testing_results[
        (multiple_testing_results["correction_method"] == "benjamini_hochberg")
        & (multiple_testing_results["test_name"] == "two_proportion_z_selection")
    ][["scenario", "time_window", "p_value_adjusted", "reject_null"]]
    summary = drift.merge(bh, on=["scenario", "time_window"], how="left")
    summary["drift_threshold"] = drift_threshold
    summary["drift_flag"] = summary["reject_null"].fillna(False) & (summary["absolute_drift"] > drift_threshold)
    summary["p_value_adjusted"] = summary["p_value_adjusted"].replace({np.nan: None})
    return summary


def sensitivity_threshold_table(
    drift_summary: pd.DataFrame,
    thresholds: tuple[float, ...] = DEFAULT_SENSITIVITY_THRESHOLDS,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        flags = drift_summary["reject_null"].fillna(False) & (drift_summary["absolute_drift"] > threshold)
        for scenario, scenario_flags in flags.groupby(drift_summary["scenario"]):
            rows.append(
                {
                    "scenario": scenario,
                    "drift_threshold": threshold,
                    "flagged_windows": int(scenario_flags.sum()),
                    "total_windows": int(scenario_flags.size),
                    "flag_rate": float(scenario_flags.mean()),
                }
            )
    return pd.DataFrame(rows)
