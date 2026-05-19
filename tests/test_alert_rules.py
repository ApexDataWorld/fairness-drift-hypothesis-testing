import pandas as pd

from fairness_drift.alert_rules import SUPPORTED_RULES, apply_alert_rule, benjamini_hochberg


def test_benjamini_hochberg_preserves_original_order():
    adjusted = benjamini_hochberg([0.20, 0.01, 0.04])
    assert adjusted[1] <= adjusted[2] <= adjusted[0]


def test_all_named_alert_rules_run():
    df = pd.DataFrame(
        {
            "window": [0, 1, 2],
            "p_value": [0.20, 0.01, 0.01],
            "drift": [0.01, 0.04, 0.05],
        }
    )
    for rule in SUPPORTED_RULES:
        out = apply_alert_rule(df, rule=rule)
        assert "alert" in out.columns
        assert len(out) == len(df)


def test_persistence_requires_consecutive_candidates():
    df = pd.DataFrame(
        {
            "window": [0, 1, 2],
            "p_value": [0.01, 0.20, 0.01],
            "drift": [0.04, 0.01, 0.04],
        }
    )
    out = apply_alert_rule(df, rule="adjusted_p_plus_materiality_plus_persistence")
    assert not out["alert"].any()
