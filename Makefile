.PHONY: install test reproduce jasa reproduce-jasa public-data verify clean

install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

test:
	pytest -q

reproduce:
	python experiments/run_fairness_drift_experiment.py
	python experiments/run_jasa_operating_characteristics.py
	python scripts/verify_jasa_outputs.py

jasa: reproduce

reproduce-jasa: reproduce

public-data:
	python experiments/run_public_adult_validation.py

verify:
	python scripts/verify_jasa_outputs.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache results/.cache results/.matplotlib data/public_cache
	rm -f results/tables/public_adult_validation_NOT_RUN.txt
