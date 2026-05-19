# JASA Upgrade for Fairness Drift Hypothesis Testing

This overlay upgrades the repository from a synthetic fairness-monitoring demonstration into a JASA-oriented reproducible statistical monitoring package.

## What is added

- Power and minimum-detectable-effect utilities: `src/fairness_drift/power.py`
- False-discovery-controlled alert-rule utilities: `src/fairness_drift/alert_rules.py`
- Operating-characteristic simulation: `experiments/run_jasa_operating_characteristics.py`
- Optional public-data validation using Adult/OpenML: `experiments/run_public_adult_validation.py`
- CI workflow: `.github/workflows/reproduce-jasa.yml`
- Output verifier: `scripts/verify_jasa_outputs.py`
- Unit tests for the new methods: `tests/test_power.py`

## Reproduce from a clean checkout

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
python experiments/run_fairness_drift_experiment.py
python experiments/run_jasa_operating_characteristics.py
python scripts/verify_jasa_outputs.py
```

Optional public-data validation:

```bash
python experiments/run_public_adult_validation.py
```

The public-data script needs internet access on first run. It exits gracefully if OpenML cannot be reached.

## JASA outputs added

- `results/tables/jasa_operating_characteristics.csv`
- `results/tables/jasa_power_table.csv`
- `results/figures/power_by_drift_magnitude.png`
- `results/figures/false_alert_rate_by_rule.png`
- `results/figures/mde_by_subgroup_size.png`
- optional: `results/tables/public_adult_validation.csv`
- optional: `results/figures/public_adult_fairness_gap_by_window.png`

## Reviewer-safe claim

Default results are synthetic unless the optional public-data validation is executed. No proprietary production data, protected health information, or regulated enterprise data are included.

## Makefile shortcuts

```bash
make install
make test
make reproduce
make jasa
make verify
make clean
```

`make reproduce` and `make jasa` run the core synthetic experiment, run the JASA operating-characteristics experiment, and verify required outputs. They do not require optional public-data validation.

## Manuscript note

The final QA manuscript reference is stored under `paper/manuscript/`. Before submission, regenerate tables and figures from code and confirm that manuscript-facing numbers match the committed outputs.
