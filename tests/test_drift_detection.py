import pandas as pd

from fairness_drift.drift_detection import (
    baseline_adjusted_fairness_drift,
    sensitivity_threshold_table,
    summarize_drift,
)


def test_baseline_adjusted_fairness_drift_uses_baseline_windows():
    metrics = pd.DataFrame(
        {
            "scenario": ["s", "s", "s"],
            "time_window": [0, 1, 2],
            "selection_parity_gap": [0.01, 0.03, 0.08],
        }
    )
    drift = baseline_adjusted_fairness_drift(metrics, baseline_windows=(0, 1))
    current = drift.loc[drift["time_window"] == 2].iloc[0]
    assert current["baseline_gap"] == 0.02
    assert round(current["baseline_adjusted_drift"], 6) == 0.06


def test_sensitivity_threshold_table_counts_flags_by_threshold():
    metrics = pd.DataFrame(
        {
            "scenario": ["s", "s"],
            "time_window": [0, 1],
            "selection_parity_gap": [0.0, 0.04],
        }
    )
    tests = pd.DataFrame(
        {
            "scenario": ["s", "s"],
            "time_window": [0, 1],
            "correction_method": ["benjamini_hochberg", "benjamini_hochberg"],
            "test_name": ["two_proportion_z_selection", "two_proportion_z_selection"],
            "p_value_adjusted": [0.9, 0.01],
            "reject_null": [False, True],
        }
    )
    summary = summarize_drift(metrics, tests, baseline_windows=(0,))
    sensitivity = sensitivity_threshold_table(summary, thresholds=(0.03, 0.05))
    assert sensitivity.loc[sensitivity["drift_threshold"] == 0.03, "flagged_windows"].iloc[0] == 1
    assert sensitivity.loc[sensitivity["drift_threshold"] == 0.05, "flagged_windows"].iloc[0] == 0
