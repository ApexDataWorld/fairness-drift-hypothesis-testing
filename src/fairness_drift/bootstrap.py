from __future__ import annotations

import numpy as np
import pandas as pd

from fairness_drift.metrics import selection_parity_gap


def bootstrap_gap_ci(
    df: pd.DataFrame,
    metric_func=selection_parity_gap,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> dict:
    rng = np.random.default_rng(random_seed)
    estimates = []
    for _ in range(n_bootstrap):
        sample = df.sample(frac=1.0, replace=True, random_state=int(rng.integers(0, 2**32 - 1)))
        estimates.append(metric_func(sample))
    alpha = 1.0 - confidence_level
    lower, upper = np.nanquantile(estimates, [alpha / 2, 1 - alpha / 2])
    return {
        "estimate": float(metric_func(df)),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "confidence_level": confidence_level,
    }


def bootstrap_gap_table(
    df: pd.DataFrame,
    n_bootstrap: int = 500,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> pd.DataFrame:
    rows = []
    for idx, ((scenario, time_window), part) in enumerate(df.groupby(["scenario", "time_window"])):
        result = bootstrap_gap_ci(
            part,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            random_seed=random_seed + idx,
        )
        result.update({"scenario": scenario, "time_window": time_window, "metric": "selection_parity_gap"})
        rows.append(result)
    return pd.DataFrame(rows)
