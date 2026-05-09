from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SimulationConfig:
    random_seed: int = 42
    n_per_window: int = 1200
    n_windows: int = 8
    decision_threshold: float = 0.55
    label_delay_rate: float = 0.2


SCENARIOS = (
    "no_drift_baseline",
    "selection_parity_drift",
    "equal_opportunity_drift",
    "false_positive_rate_drift",
    "small_subgroup_drift",
    "multiple_testing_false_alarm_control",
)


def generate_production_logs(
    scenario: str,
    config: SimulationConfig | None = None,
) -> pd.DataFrame:
    """Generate synthetic production ML logs for one drift scenario."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    cfg = config or SimulationConfig()
    rng = np.random.default_rng(cfg.random_seed + SCENARIOS.index(scenario) * 1000)
    frames = []

    for time_window in range(cfg.n_windows):
        drift_strength = max(0, time_window - 2) / max(1, cfg.n_windows - 3)
        subgroup_prob_b = 0.06 if scenario == "small_subgroup_drift" else 0.5
        group = rng.choice(["A", "B"], size=cfg.n_per_window, p=[1 - subgroup_prob_b, subgroup_prob_b])
        group_b = (group == "B").astype(float)

        latent = rng.normal(0.0, 1.0, cfg.n_per_window) - 0.15 * group_b
        label_prob = _sigmoid(latent)
        true_label = rng.binomial(1, label_prob)

        score_logit = 1.4 * latent + rng.normal(0.0, 0.7, cfg.n_per_window)
        score = _sigmoid(score_logit)

        threshold = np.full(cfg.n_per_window, cfg.decision_threshold)
        threshold += _scenario_threshold_shift(scenario, group, true_label, drift_strength)
        threshold = np.clip(threshold, 0.05, 0.95)
        decision = (score >= threshold).astype(int)

        label_delay = rng.binomial(1, cfg.label_delay_rate, cfg.n_per_window).astype(bool)
        observed_label = true_label.astype(float)
        observed_label[label_delay & (time_window >= cfg.n_windows - 2)] = np.nan

        frames.append(
            pd.DataFrame(
                {
                    "timestamp": pd.Timestamp("2025-01-01") + pd.to_timedelta(time_window, unit="W"),
                    "time_window": time_window,
                    "person_id": [
                        f"{scenario[:3]}-{time_window:02d}-{i:06d}" for i in range(cfg.n_per_window)
                    ],
                    "group": group,
                    "model_score": score,
                    "decision": decision,
                    "true_label": true_label,
                    "observed_label": observed_label,
                    "label_delayed": label_delay,
                    "scenario": scenario,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def generate_all_scenarios(config: SimulationConfig | None = None) -> pd.DataFrame:
    """Generate synthetic logs for every configured scenario."""
    return pd.concat(
        [generate_production_logs(scenario, config=config) for scenario in SCENARIOS],
        ignore_index=True,
    )


def _scenario_threshold_shift(
    scenario: str,
    group: np.ndarray,
    true_label: np.ndarray,
    drift_strength: float,
) -> np.ndarray:
    group_b = group == "B"
    shift = np.zeros(len(group))
    active = drift_strength > 0

    if not active or scenario in {"no_drift_baseline", "multiple_testing_false_alarm_control"}:
        return shift

    if scenario == "selection_parity_drift":
        shift[group_b] = 0.22 * drift_strength
    elif scenario == "equal_opportunity_drift":
        shift[group_b & (true_label == 1)] = 0.28 * drift_strength
    elif scenario == "false_positive_rate_drift":
        shift[group_b & (true_label == 0)] = -0.22 * drift_strength
    elif scenario == "small_subgroup_drift":
        shift[group_b] = 0.32 * drift_strength

    return shift


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
