.PHONY: setup test lint format triage sweep mechanistic triangulate clean

setup:
	pip install -r requirements.txt

test:
	PYTHONPATH=. pytest tests/ -v

lint:
	black --check probes/ scripts/ tests/ && ruff check probes/ scripts/ tests/

format:
	black probes/ scripts/ tests/ && ruff check --fix probes/ scripts/ tests/

triage:
	PYTHONPATH=. python3 scripts/run_contamination_triage.py

sweep:
	PYTHONPATH=. python3 scripts/run_behavioral_sweep.py

mechanistic:
	PYTHONPATH=. python3 scripts/run_mechanistic_sweep.py

triangulate:
	PYTHONPATH=. python3 scripts/run_triangulation.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage && find . -name "*.pyc" -delete
