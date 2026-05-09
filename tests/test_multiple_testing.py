import pandas as pd

from fairness_drift.multiple_testing import apply_multiple_testing_correction


def test_multiple_testing_outputs_all_methods():
    results = pd.DataFrame(
        {
            "scenario": ["s", "s", "s"],
            "time_window": [0, 1, 2],
            "test_name": ["a", "a", "a"],
            "statistic": [1.0, 2.0, 3.0],
            "p_value": [0.001, 0.03, 0.9],
        }
    )
    corrected = apply_multiple_testing_correction(results)
    assert set(corrected["correction_method"]) == {"bonferroni", "holm", "benjamini_hochberg"}
    assert "p_value_adjusted" in corrected.columns
    assert "reject_null" in corrected.columns
