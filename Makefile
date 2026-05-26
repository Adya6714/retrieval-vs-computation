.PHONY: bootstrap clone-sources setup test lint format triage sweep triangulate clean help

help:
	@echo ""
	@echo "Common targets:"
	@echo "  make bootstrap     Fresh checkout: install deps + clone external sources"
	@echo "  make clone-sources Re-clone planbench + gsm-symbolic + fast-downward"
	@echo "  make setup         pip install -r requirements.txt"
	@echo "  make test          Run unit tests"
	@echo "  make sweep         Run BW Probe-1 behavioral sweep"
	@echo "  make triage        Run BW Probe-3 contamination triage"
	@echo "  make triangulate   Run BW Probe-3 triangulation"
	@echo "  make clean         Remove __pycache__ + .pytest_cache + *.pyc"
	@echo ""

# Fresh checkout: dependencies + upstream sources (PlanBench, GSM-Symbolic, Fast Downward)
bootstrap: setup clone-sources
	@echo ""
	@echo "Bootstrap complete. See ANALYSIS.md for the consolidated research summary."
	@echo "See README.md > 'Reproducing results' for sweep commands."

# External upstream sources (gitignored; reclone per environment)
clone-sources:
	@mkdir -p data/sources tools
	@if [ ! -d data/sources/planbench ]; then \
		echo "Cloning karthikv792/LLMs-Planning -> data/sources/planbench"; \
		git clone --depth 1 https://github.com/karthikv792/LLMs-Planning.git data/sources/planbench; \
	else echo "data/sources/planbench already present"; fi
	@if [ ! -d data/sources/gsm_symbolic ]; then \
		echo "Cloning apple/ml-gsm-symbolic -> data/sources/gsm_symbolic"; \
		git clone --depth 1 https://github.com/apple/ml-gsm-symbolic.git data/sources/gsm_symbolic; \
	else echo "data/sources/gsm_symbolic already present"; fi
	@if [ ! -d tools/fast-downward ]; then \
		echo "Cloning Fast Downward -> tools/fast-downward"; \
		git clone --depth 1 https://github.com/aibasel/downward.git tools/fast-downward; \
	else echo "tools/fast-downward already present"; fi

setup:
	pip install -r requirements.txt

test:
	PYTHONPATH=. pytest tests/ -v

lint:
	black --check probes/ scripts/ tests/ && ruff check probes/ scripts/ tests/

format:
	black probes/ scripts/ tests/ && ruff check --fix probes/ scripts/ tests/

triage:
	PYTHONPATH=. python3 scripts/BW_P3_SCR_run_contamination_triage.py

sweep:
	PYTHONPATH=. python3 scripts/BW_P1_SCR_run_behavioral_sweep.py

triangulate:
	PYTHONPATH=. python3 scripts/BW_P3_SCR_run_triangulation.py

clean:
	rm -rf __pycache__ .pytest_cache .coverage && find . -name "*.pyc" -delete
