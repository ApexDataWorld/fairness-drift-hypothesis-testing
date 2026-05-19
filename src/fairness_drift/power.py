"""Power and minimum-detectable-effect utilities for fairness drift monitoring.

These functions are intentionally dependency-light and use normal approximations
for two-proportion tests. They are designed for planning and reproducibility in
JASA-facing operating-characteristic experiments, not for legal determinations.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class PowerSetting:
    n_per_group: int
    baseline_rate: float = 0.40
    alpha: float = 0.05
    two_sided: bool = True


def _validate_probability(x: float, name: str) -> None:
    if not 0 < x < 1:
        raise ValueError(f"{name} must be in (0, 1); got {x!r}")


def two_proportion_power(
    n_per_group: int,
    baseline_rate: float,
    effect_size: float,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Approximate power for detecting p2 - p1 = effect_size.

    Uses the Wald normal approximation for two independent proportions with
    balanced group sizes. For unbalanced or very small samples, confirm with
    simulation or exact methods.
    """
    if n_per_group <= 0:
        raise ValueError("n_per_group must be positive")
    _validate_probability(baseline_rate, "baseline_rate")
    _validate_probability(alpha, "alpha")
    p1 = baseline_rate
    p2 = min(max(baseline_rate + effect_size, 1e-9), 1 - 1e-9)
    se = sqrt(p1 * (1 - p1) / n_per_group + p2 * (1 - p2) / n_per_group)
    if se == 0:
        return 1.0 if effect_size else 0.0
    z_alt = abs(effect_size) / se
    crit = norm.ppf(1 - alpha / 2) if two_sided else norm.ppf(1 - alpha)
    # P(|Z| > crit) under mean shift z_alt.
    power = norm.sf(crit - z_alt) + norm.cdf(-crit - z_alt) if two_sided else norm.sf(crit - z_alt)
    return float(min(max(power, 0.0), 1.0))


def minimum_detectable_effect(
    n_per_group: int,
    baseline_rate: float = 0.40,
    target_power: float = 0.80,
    alpha: float = 0.05,
    grid: Iterable[float] | None = None,
) -> float:
    """Return the smallest absolute proportion gap reaching target power."""
    _validate_probability(target_power, "target_power")
    if grid is None:
        grid = np.linspace(0.001, 0.40, 400)
    for gap in grid:
        if two_proportion_power(n_per_group, baseline_rate, float(gap), alpha) >= target_power:
            return float(gap)
    return float("nan")


def make_power_table(
    sample_sizes: Iterable[int] = (50, 100, 200, 500, 1000, 2000),
    effect_sizes: Iterable[float] = (0.05, 0.10, 0.15, 0.20),
    baseline_rate: float = 0.40,
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for n in sample_sizes:
        row = {"n_per_group": int(n), "baseline_rate": baseline_rate, "alpha": alpha}
        for gap in effect_sizes:
            row[f"power_gap_{gap:.2f}"] = two_proportion_power(int(n), baseline_rate, float(gap), alpha)
        row["mde_80pct_power"] = minimum_detectable_effect(int(n), baseline_rate, 0.80, alpha)
        rows.append(row)
    return pd.DataFrame(rows)
