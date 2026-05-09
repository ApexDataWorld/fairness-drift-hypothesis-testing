from __future__ import annotations

import numpy as np
import pandas as pd


def selection_rate_by_group(df: pd.DataFrame, group_col: str = "group") -> pd.Series:
    return df.groupby(group_col)["decision"].mean()


def selection_parity_gap(df: pd.DataFrame, group_col: str = "group") -> float:
    rates = selection_rate_by_group(df, group_col)
    return _gap_from_series(rates)


def disparate_impact_ratio(df: pd.DataFrame, group_col: str = "group") -> float:
    rates = selection_rate_by_group(df, group_col)
    if len(rates) < 2:
        return np.nan
    denominator = rates.max()
    return np.nan if denominator == 0 else rates.min() / denominator


def confusion_rates_by_group(
    df: pd.DataFrame,
    group_col: str = "group",
    label_col: str = "observed_label",
) -> pd.DataFrame:
    labeled = df.dropna(subset=[label_col]).copy()
    rows = []
    for group, part in labeled.groupby(group_col):
        y = part[label_col].astype(int)
        pred = part["decision"].astype(int)
        positives = y == 1
        negatives = y == 0
        rows.append(
            {
                group_col: group,
                "tpr": _safe_mean(pred[positives] == 1),
                "fpr": _safe_mean(pred[negatives] == 1),
                "fnr": _safe_mean(pred[positives] == 0),
                "support": len(part),
                "positives": int(positives.sum()),
                "negatives": int(negatives.sum()),
            }
        )
    return pd.DataFrame(rows).set_index(group_col)


def calibration_by_bin(
    df: pd.DataFrame,
    n_bins: int = 5,
    group_col: str = "group",
    label_col: str = "observed_label",
) -> pd.DataFrame:
    labeled = df.dropna(subset=[label_col]).copy()
    labeled["score_bin"] = pd.cut(labeled["model_score"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    return (
        labeled.groupby([group_col, "score_bin"], observed=False)
        .agg(
            mean_score=("model_score", "mean"),
            observed_positive_rate=(label_col, "mean"),
            n=("decision", "size"),
        )
        .reset_index()
    )


def score_distribution_summary(df: pd.DataFrame, group_col: str = "group") -> pd.DataFrame:
    return (
        df.groupby(group_col)["model_score"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )


def compute_fairness_metrics(df: pd.DataFrame, group_col: str = "group") -> pd.DataFrame:
    rows = []
    keys = ["scenario", "time_window"]
    for (scenario, time_window), part in df.groupby(keys):
        rates = selection_rate_by_group(part, group_col)
        confusion = confusion_rates_by_group(part, group_col)
        score_summary = score_distribution_summary(part, group_col).set_index(group_col)
        row = {
            "scenario": scenario,
            "time_window": time_window,
            "selection_rate_A": rates.get("A", np.nan),
            "selection_rate_B": rates.get("B", np.nan),
            "selection_parity_gap": rates.get("B", np.nan) - rates.get("A", np.nan),
            "disparate_impact_ratio": disparate_impact_ratio(part, group_col),
            "tpr_A": confusion["tpr"].get("A", np.nan) if not confusion.empty else np.nan,
            "tpr_B": confusion["tpr"].get("B", np.nan) if not confusion.empty else np.nan,
            "fpr_A": confusion["fpr"].get("A", np.nan) if not confusion.empty else np.nan,
            "fpr_B": confusion["fpr"].get("B", np.nan) if not confusion.empty else np.nan,
            "fnr_A": confusion["fnr"].get("A", np.nan) if not confusion.empty else np.nan,
            "fnr_B": confusion["fnr"].get("B", np.nan) if not confusion.empty else np.nan,
            "score_mean_A": score_summary["mean"].get("A", np.nan),
            "score_mean_B": score_summary["mean"].get("B", np.nan),
        }
        row["equal_opportunity_gap"] = row["tpr_B"] - row["tpr_A"]
        row["false_positive_rate_gap"] = row["fpr_B"] - row["fpr_A"]
        row["false_negative_rate_gap"] = row["fnr_B"] - row["fnr_A"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def _gap_from_series(values: pd.Series) -> float:
    if len(values) < 2:
        return np.nan
    return values.max() - values.min()


def _safe_mean(values: pd.Series | np.ndarray) -> float:
    return np.nan if len(values) == 0 else float(np.mean(values))
