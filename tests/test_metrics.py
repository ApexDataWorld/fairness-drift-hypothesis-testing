import pandas as pd

from fairness_drift.metrics import (
    compute_fairness_metrics,
    disparate_impact_ratio,
    selection_parity_gap,
    selection_rate_by_group,
)


def test_selection_metrics_are_grouped_correctly():
    df = pd.DataFrame(
        {
            "scenario": ["s"] * 4,
            "time_window": [0] * 4,
            "group": ["A", "A", "B", "B"],
            "decision": [1, 0, 1, 1],
            "observed_label": [1, 0, 1, 0],
            "model_score": [0.8, 0.3, 0.9, 0.7],
        }
    )
    rates = selection_rate_by_group(df)
    assert rates["A"] == 0.5
    assert rates["B"] == 1.0
    assert selection_parity_gap(df) == 0.5
    assert disparate_impact_ratio(df) == 0.5


def test_compute_fairness_metrics_includes_requested_gaps():
    df = pd.DataFrame(
        {
            "scenario": ["s"] * 4,
            "time_window": [0] * 4,
            "group": ["A", "A", "B", "B"],
            "decision": [1, 0, 1, 0],
            "observed_label": [1, 0, 1, 0],
            "model_score": [0.8, 0.2, 0.7, 0.4],
        }
    )
    metrics = compute_fairness_metrics(df)
    assert {"equal_opportunity_gap", "false_positive_rate_gap", "false_negative_rate_gap"}.issubset(metrics.columns)
    assert metrics.loc[0, "selection_parity_gap"] == 0.0
