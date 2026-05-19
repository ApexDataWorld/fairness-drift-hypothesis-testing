"""Alert-rule helpers for repeated fairness inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

SUPPORTED_RULES = (
    "raw_p_only",
    "adjusted_p_only",
    "effect_size_only",
    "adjusted_p_plus_materiality",
    "adjusted_p_plus_materiality_plus_persistence",
)


def benjamini_hochberg(p_values):
    """Return Benjamini-Hochberg adjusted p-values in original order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be one-dimensional")
    m = len(p)
    if m == 0:
        return np.array([])
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def apply_alert_rule(
    df: pd.DataFrame,
    rule: str = "adjusted_p_plus_materiality",
    p_col: str = "p_value",
    effect_col: str = "drift",
    alpha: float = 0.05,
    delta: float = 0.03,
    family_cols: list[str] | None = None,
    time_col: str = "window",
    persistence_windows: int = 2,
) -> pd.DataFrame:
    """Apply one named fairness alert rule.

    Supported rules are raw_p_only, adjusted_p_only, effect_size_only,
    adjusted_p_plus_materiality, and
    adjusted_p_plus_materiality_plus_persistence.
    """
    if rule not in SUPPORTED_RULES:
        raise ValueError(f"Unsupported alert rule {rule!r}; expected one of {SUPPORTED_RULES}")

    out = df.copy()
    families = family_cols or []
    if families:
        out["p_adj_bh"] = out.groupby(families, group_keys=False)[p_col].transform(benjamini_hochberg)
    else:
        out["p_adj_bh"] = benjamini_hochberg(out[p_col].to_numpy())

    raw = out[p_col] < alpha
    adjusted = out["p_adj_bh"] < alpha
    material = out[effect_col].abs() > delta

    if rule == "raw_p_only":
        out["alert"] = raw
    elif rule == "adjusted_p_only":
        out["alert"] = adjusted
    elif rule == "effect_size_only":
        out["alert"] = material
    elif rule == "adjusted_p_plus_materiality":
        out["alert"] = adjusted & material
    else:
        out["alert_candidate"] = adjusted & material
        out["alert"] = _apply_persistence(out, families, time_col, persistence_windows)

    out["rule"] = rule
    out["alpha"] = alpha
    out["delta"] = delta
    return out


def _apply_persistence(
    df: pd.DataFrame,
    family_cols: list[str],
    time_col: str,
    persistence_windows: int,
) -> pd.Series:
    if persistence_windows <= 1:
        return df["alert_candidate"]
    ordered = df.sort_values(family_cols + [time_col] if family_cols else [time_col])
    if family_cols:
        persisted = ordered.groupby(family_cols)["alert_candidate"].transform(
            lambda s: s.rolling(persistence_windows, min_periods=persistence_windows).sum().ge(persistence_windows)
        )
    else:
        persisted = ordered["alert_candidate"].rolling(
            persistence_windows,
            min_periods=persistence_windows,
        ).sum().ge(persistence_windows)
    return persisted.reindex(df.index).fillna(False).astype(bool)
