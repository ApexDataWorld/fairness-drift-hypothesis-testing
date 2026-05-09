from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest


def two_proportion_z_test(success_a: int, n_a: int, success_b: int, n_b: int) -> dict:
    if n_a == 0 or n_b == 0:
        return {"statistic": np.nan, "p_value": np.nan}
    statistic, p_value = proportions_ztest([success_a, success_b], [n_a, n_b])
    return {"statistic": float(statistic), "p_value": float(p_value)}


def chi_square_independence_test(df: pd.DataFrame, group_col: str = "group") -> dict:
    table = pd.crosstab(df[group_col], df["decision"])
    statistic, p_value, dof, expected = stats.chi2_contingency(table)
    return {"statistic": float(statistic), "p_value": float(p_value), "dof": int(dof), "expected_min": float(expected.min())}


def fisher_exact_test(df: pd.DataFrame, group_col: str = "group") -> dict:
    table = pd.crosstab(df[group_col], df["decision"]).reindex(index=["A", "B"], columns=[0, 1], fill_value=0)
    odds_ratio, p_value = stats.fisher_exact(table.to_numpy())
    return {"statistic": float(odds_ratio), "p_value": float(p_value)}


def permutation_test_score_difference(
    df: pd.DataFrame,
    group_col: str = "group",
    n_permutations: int = 1000,
    random_seed: int = 42,
) -> dict:
    rng = np.random.default_rng(random_seed)
    observed = _mean_score_gap(df, group_col)
    groups = df[group_col].to_numpy().copy()
    scores = df["model_score"].to_numpy()
    extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(groups)
        permuted = pd.DataFrame({group_col: groups, "model_score": scores})
        if abs(_mean_score_gap(permuted, group_col)) >= abs(observed):
            extreme += 1
    p_value = (extreme + 1) / (n_permutations + 1)
    return {"statistic": float(observed), "p_value": float(p_value)}


def ks_score_shift_test(current: pd.DataFrame, baseline: pd.DataFrame) -> dict:
    statistic, p_value = stats.ks_2samp(current["model_score"], baseline["model_score"])
    return {"statistic": float(statistic), "p_value": float(p_value)}


def mann_whitney_score_test(df: pd.DataFrame, group_col: str = "group") -> dict:
    group_a = df.loc[df[group_col] == "A", "model_score"]
    group_b = df.loc[df[group_col] == "B", "model_score"]
    statistic, p_value = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")
    return {"statistic": float(statistic), "p_value": float(p_value)}


def run_window_hypothesis_tests(
    df: pd.DataFrame,
    baseline_windows: tuple[int, ...] = (0, 1),
    n_permutations: int = 500,
    random_seed: int = 42,
) -> pd.DataFrame:
    rows = []
    baseline = df[df["time_window"].isin(baseline_windows)]
    for (scenario, time_window), part in df.groupby(["scenario", "time_window"]):
        baseline_part = baseline[baseline["scenario"] == scenario]
        counts = part.groupby("group")["decision"].agg(["sum", "count"])
        if {"A", "B"}.issubset(counts.index):
            z = two_proportion_z_test(
                int(counts.loc["A", "sum"]),
                int(counts.loc["A", "count"]),
                int(counts.loc["B", "sum"]),
                int(counts.loc["B", "count"]),
            )
            rows.append(_test_row(scenario, time_window, "two_proportion_z_selection", z))

        chi = chi_square_independence_test(part)
        rows.append(_test_row(scenario, time_window, "chi_square_group_decision", chi))

        if chi["expected_min"] < 5 or scenario == "small_subgroup_drift":
            rows.append(_test_row(scenario, time_window, "fisher_exact_group_decision", fisher_exact_test(part)))

        rows.append(
            _test_row(
                scenario,
                time_window,
                "permutation_score_gap",
                permutation_test_score_difference(
                    part,
                    n_permutations=n_permutations,
                    random_seed=random_seed + int(time_window),
                ),
            )
        )
        rows.append(_test_row(scenario, time_window, "mann_whitney_score_gap", mann_whitney_score_test(part)))
        if not baseline_part.empty:
            rows.append(_test_row(scenario, time_window, "ks_score_shift_vs_baseline", ks_score_shift_test(part, baseline_part)))
    return pd.DataFrame(rows)


def _test_row(scenario: str, time_window: int, test_name: str, result: dict) -> dict:
    return {
        "scenario": scenario,
        "time_window": time_window,
        "test_name": test_name,
        "statistic": result.get("statistic", np.nan),
        "p_value": result.get("p_value", np.nan),
    }


def _mean_score_gap(df: pd.DataFrame, group_col: str) -> float:
    means = df.groupby(group_col)["model_score"].mean()
    return float(means.get("B", np.nan) - means.get("A", np.nan))
