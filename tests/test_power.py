from fairness_drift.power import two_proportion_power, minimum_detectable_effect, make_power_table
from fairness_drift.alert_rules import benjamini_hochberg


def test_power_increases_with_effect_size():
    p_small = two_proportion_power(200, 0.40, 0.05)
    p_large = two_proportion_power(200, 0.40, 0.15)
    assert p_large > p_small


def test_mde_positive():
    assert minimum_detectable_effect(200, 0.40, 0.80) > 0


def test_power_table_columns():
    table = make_power_table(sample_sizes=[50], effect_sizes=[0.10])
    assert "power_gap_0.10" in table.columns


def test_bh_adjustment_monotone_range():
    adjusted = benjamini_hochberg([0.01, 0.04, 0.20])
    assert all(0 <= x <= 1 for x in adjusted)
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
