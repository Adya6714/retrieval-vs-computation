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

# ─── v1.1 corrected rerun (see V11_RUNBOOK.md) ──────────────────────────────
.PHONY: v11-setup v11-bank v11-qa v11-test v11-dry v11-models v11-pilot v11-p1 v11-e1 v11-e2 v11-p2
RUN ?= main
MODELS ?= all
WORKERS ?= 8

v11-setup:
	pip install -r requirements.txt
	@test -d data/sources/planbench || git clone --depth 1 https://github.com/karthikv792/LLMs-Planning.git data/sources/planbench
	@test -f .env || cp env.example .env

v11-bank:        ## rebuild data/v11 from the legacy banks
	python -m rvc11 build-bank

v11-qa:          ## every QA gate, offline, no key
	python -m rvc11 qa

v11-test:
	PYTHONPATH=. pytest tests_v11 -q

v11-dry:         ## whole pipeline with scripted models, no key
	python -m rvc11 run p1_surface --run _dry --models mock_drop,mock_noisy --limit 4
	python -m rvc11 run e1_instruction --run _dry --models mock_drop --limit 4
	python -m rvc11 run e2_noise --run _dry --models mock_noisy --limit 5
	python -m rvc11 run p2_plan --run _dry --models mock_drop --limit 10
	python -m rvc11 p2-continue --run _dry
	@echo "open results/v11/_dry/analysis/REPORT.md"

v11-models:      ## check model ids against the live catalogue (needs the key)
	python -m rvc11 models-check

v11-pilot:       ## 2 problems per subtype on every model: cost and format check
	python -m rvc11 run p1_surface --run pilot --models $(MODELS) --limit 2 --workers $(WORKERS)

v11-p1:
	python -m rvc11 run p1_surface --run $(RUN) --models $(MODELS) --workers $(WORKERS)

v11-e1:
	python -m rvc11 run e1_instruction --run $(RUN) --models $(MODELS) --workers $(WORKERS)

v11-e2:
	python -m rvc11 run e2_noise --run $(RUN) --models $(MODELS) --workers $(WORKERS)

v11-p2:
	python -m rvc11 run p2_plan --run $(RUN) --models $(MODELS) --workers $(WORKERS)
	python -m rvc11 p2-continue --run $(RUN) --workers $(WORKERS)
