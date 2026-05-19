from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED = [
    "results/tables/fairness_metrics.csv",
    "results/tables/hypothesis_test_results.csv",
    "results/tables/multiple_testing_results.csv",
    "results/tables/drift_detection_summary.csv",
    "results/tables/threshold_sensitivity_results.csv",
    "results/tables/bootstrap_ci_fairness_gap.csv",
    "results/figures/selection_rate_by_time.png",
    "results/figures/fairness_gap_by_time.png",
    "results/figures/p_values_by_time.png",
    "results/figures/bootstrap_ci_fairness_gap.png",
    "results/tables/jasa_operating_characteristics.csv",
    "results/tables/jasa_power_table.csv",
    "results/figures/power_by_drift_magnitude.png",
    "results/figures/false_alert_rate_by_rule.png",
    "results/figures/mde_by_subgroup_size.png",
]

OPTIONAL_PUBLIC = [
    "results/tables/public_adult_validation.csv",
    "results/figures/public_adult_fairness_gap_by_window.png",
]


def main():
    parser = argparse.ArgumentParser(description="Verify required JASA reproducibility outputs.")
    parser.add_argument("--include-public", action="store_true", help="Also require optional Adult public-data outputs.")
    args = parser.parse_args()

    required = REQUIRED + (OPTIONAL_PUBLIC if args.include_public else [])
    bad = [p for p in required if not _exists_and_nonempty(Path(p))]
    if bad:
        print("Missing or empty required JASA outputs:")
        for p in bad:
            print(" -", p)
        raise SystemExit(1)
    print("All required JASA outputs exist and are non-empty.")


def _exists_and_nonempty(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


if __name__ == "__main__":
    main()
