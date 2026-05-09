from __future__ import annotations

from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fairness_drift.bootstrap import bootstrap_gap_table
from fairness_drift.data_generator import SimulationConfig, generate_all_scenarios
from fairness_drift.drift_detection import sensitivity_threshold_table, summarize_drift
from fairness_drift.hypothesis_tests import run_window_hypothesis_tests
from fairness_drift.metrics import compute_fairness_metrics
from fairness_drift.multiple_testing import apply_multiple_testing_correction
from fairness_drift.reporting import ensure_output_dirs, write_tables
from fairness_drift.visualization import (
    plot_bootstrap_ci,
    plot_fairness_gap_by_time,
    plot_p_values_by_time,
    plot_selection_rate_by_time,
)


def main() -> None:
    config = _load_config()
    baseline_windows = tuple(config.get("baseline_windows", [0, 1]))
    sim_config = SimulationConfig(
        random_seed=config["random_seed"],
        n_per_window=config["n_per_window"],
        n_windows=config["n_windows"],
        decision_threshold=config["decision_threshold"],
    )

    logs = generate_all_scenarios(sim_config)
    Path("data/synthetic").mkdir(parents=True, exist_ok=True)
    logs.to_csv("data/synthetic/production_logs.csv", index=False)

    fairness_metrics = compute_fairness_metrics(logs)
    hypothesis_tests = run_window_hypothesis_tests(
        logs,
        baseline_windows=baseline_windows,
        n_permutations=config["permutation"]["n_permutations"],
        random_seed=config["random_seed"],
    )
    multiple_testing = apply_multiple_testing_correction(
        hypothesis_tests,
        alpha=config["multiple_testing_alpha"],
    )
    drift_summary = summarize_drift(fairness_metrics, multiple_testing, baseline_windows=baseline_windows)
    sensitivity_table = sensitivity_threshold_table(drift_summary)
    bootstrap_table = bootstrap_gap_table(
        logs,
        n_bootstrap=config["bootstrap"]["n_bootstrap"],
        confidence_level=config["bootstrap"]["confidence_level"],
        random_seed=config["random_seed"],
    )

    tables_dir, figures_dir = ensure_output_dirs("results")
    write_tables(fairness_metrics, hypothesis_tests, multiple_testing, drift_summary, "results")
    sensitivity_table.to_csv(tables_dir / "threshold_sensitivity_results.csv", index=False)
    bootstrap_table.to_csv(tables_dir / "bootstrap_ci_fairness_gap.csv", index=False)
    plot_selection_rate_by_time(fairness_metrics, figures_dir / "selection_rate_by_time.png")
    plot_fairness_gap_by_time(fairness_metrics, figures_dir / "fairness_gap_by_time.png")
    plot_p_values_by_time(hypothesis_tests, figures_dir / "p_values_by_time.png")
    plot_bootstrap_ci(bootstrap_table, figures_dir / "bootstrap_ci_fairness_gap.png")

    print("Generated fairness drift experiment outputs under results/ and data/synthetic/.")


def _load_config() -> dict:
    with Path("configs/experiment_config.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


if __name__ == "__main__":
    main()
