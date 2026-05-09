import pandas as pd

from fairness_drift.bootstrap import bootstrap_gap_ci


def test_bootstrap_gap_ci_brackets_estimate_for_simple_case():
    df = pd.DataFrame(
        {
            "group": ["A"] * 20 + ["B"] * 20,
            "decision": [1] * 10 + [0] * 10 + [1] * 15 + [0] * 5,
        }
    )
    ci = bootstrap_gap_ci(df, n_bootstrap=100, random_seed=123)
    assert ci["ci_lower"] <= ci["estimate"] <= ci["ci_upper"]
    assert ci["confidence_level"] == 0.95
