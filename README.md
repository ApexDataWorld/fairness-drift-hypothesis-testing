# fairness-drift-hypothesis-testing

Runnable experiment code for **Technical Paper 4: Statistical Hypothesis Testing for Detecting Bias and Fairness Drift in Production Machine Learning Systems**.

This repository simulates production-style ML decision logs over time, computes subgroup fairness metrics, runs statistical hypothesis tests, applies multiple-testing correction, and exports manuscript-ready tables and figures. It does not rewrite or replace the paper, and it does not claim production results.

## What This Repo Does

- Generates synthetic production-style logs with time windows, person IDs, subgroup attributes, scores, binary decisions, labels, delayed labels, and scenario labels.
- Computes fairness metrics including selection rate, parity gaps, disparate impact, TPR/FPR/FNR gaps, calibration-by-bin, and score summaries.
- Runs hypothesis tests for subgroup differences and drift, including z-tests, chi-square, Fisher exact, bootstrap intervals, permutation tests, KS tests, Mann-Whitney U tests, and a baseline-adjusted fairness drift statistic.
- Applies Bonferroni, Holm, and Benjamini-Hochberg FDR correction.
- Produces CSV tables and PNG figures under `results/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Run Experiments

Generate synthetic logs only:

```bash
python experiments/run_simulation.py
```

Run the full fairness drift experiment and regenerate all tables and figures:

```bash
python experiments/run_fairness_drift_experiment.py
```

The full experiment writes:

- `results/tables/fairness_metrics.csv`
- `results/tables/hypothesis_test_results.csv`
- `results/tables/multiple_testing_results.csv`
- `results/tables/drift_detection_summary.csv`
- `results/tables/threshold_sensitivity_results.csv`
- `results/figures/selection_rate_by_time.png`
- `results/figures/fairness_gap_by_time.png`
- `results/figures/p_values_by_time.png`
- `results/figures/bootstrap_ci_fairness_gap.png`

## Configuration

Default settings live in `configs/experiment_config.yaml`. The scripts use fixed random seeds so outputs are reproducible.

## JASA Reproducibility

The JASA upgrade adds power analysis, alert-rule operating characteristics, an output verifier, a CI workflow, and optional public Adult dataset validation. See `README_JASA_UPGRADE.md` for the exact reproduction workflow.

Regenerate the JASA outputs with:

```bash
python experiments/run_fairness_drift_experiment.py
python experiments/run_jasa_operating_characteristics.py
python scripts/verify_jasa_outputs.py
```

Additional JASA outputs include:

- `results/tables/jasa_operating_characteristics.csv`
- `results/tables/jasa_power_table.csv`
- `results/figures/power_by_drift_magnitude.png`
- `results/figures/false_alert_rate_by_rule.png`
- `results/figures/mde_by_subgroup_size.png`

The Adult public-data validation script is optional and is not required for CI or the default synthetic-only validation.

## Tests

```bash
pytest
```

## Reviewer-Safe Data Note

All generated results are synthetic-only unless replaced by approved public data or properly de-identified data. The repository makes no production-performance, deployment, regulatory, or business-impact claims; any tables and figures from the default workflow come from runnable simulation code.
