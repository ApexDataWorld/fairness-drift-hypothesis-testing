import pandas as pd

from fairness_drift.hypothesis_tests import (
    chi_square_independence_test,
    fisher_exact_test,
    ks_score_shift_test,
    mann_whitney_score_test,
    two_proportion_z_test,
)


def test_two_proportion_z_test_returns_p_value():
    result = two_proportion_z_test(50, 100, 30, 100)
    assert 0 <= result["p_value"] <= 1
    assert result["statistic"] != 0


def test_categorical_tests_return_p_values():
    df = pd.DataFrame({"group": ["A"] * 10 + ["B"] * 10, "decision": [1] * 8 + [0] * 2 + [1] * 3 + [0] * 7})
    assert 0 <= chi_square_independence_test(df)["p_value"] <= 1
    assert 0 <= fisher_exact_test(df)["p_value"] <= 1


def test_score_distribution_tests_return_p_values():
    df = pd.DataFrame(
        {
            "group": ["A"] * 5 + ["B"] * 5,
            "model_score": [0.1, 0.2, 0.2, 0.3, 0.3, 0.7, 0.8, 0.8, 0.9, 0.9],
        }
    )
    baseline = df.iloc[:5].copy()
    current = df.iloc[5:].copy()
    assert 0 <= mann_whitney_score_test(df)["p_value"] <= 1
    assert 0 <= ks_score_shift_test(current, baseline)["p_value"] <= 1
