from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_output_dirs(results_dir: str | Path = "results") -> tuple[Path, Path]:
    root = Path(results_dir)
    tables = root / "tables"
    figures = root / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    return tables, figures


def write_tables(
    fairness_metrics: pd.DataFrame,
    hypothesis_tests: pd.DataFrame,
    multiple_testing: pd.DataFrame,
    drift_summary: pd.DataFrame,
    results_dir: str | Path = "results",
) -> None:
    tables, _ = ensure_output_dirs(results_dir)
    fairness_metrics.to_csv(tables / "fairness_metrics.csv", index=False)
    hypothesis_tests.to_csv(tables / "hypothesis_test_results.csv", index=False)
    multiple_testing.to_csv(tables / "multiple_testing_results.csv", index=False)
    drift_summary.to_csv(tables / "drift_detection_summary.csv", index=False)
