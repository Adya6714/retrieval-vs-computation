# Beyond Accuracy — Retrieval vs Computation in LLM Reasoning

**Authors:** Adya
**Affiliation:** BITS Pilani
**Target venues:** BlackboxNLP 2026, GenBench 2026 (primary); EMNLP Findings (stretch)
**Research design:** `CHARTER.md`
**Status:** Multi-family evaluation code is active for **Blocksworld (BW)**, **GSM**, and **Algorithmic (ALGO)**. Outputs in `results/` include historical + refreshed runs; always use the family-specific sections below to pick canonical files for analysis.

---

## Start Here (for analysis)

If you only read one part of this README, read this section.

### 1) Core theory in one paragraph

The project asks whether LLM performance is driven by:
- **retrieval** (pattern recall from training-like inputs), or
- **computation** (actual stepwise reasoning on structure).

This is tested with **three probes** on aligned instances:
- **Probe 1 (behavioral invariance):** how answers change under controlled variants (`W1..W6`).
- **Probe 2 (process coupling):** whether declared strategy/plans match stepwise execution and reaction to injection.
- **Probe 3 (contamination):** whether behavior correlates with training-data proximity (Infini-gram scores).

### 2) Family map (what “CW/BW, GSM, ALGO” each means)

- **BW** (`blocksworld` / planning suite): planning and action validity stress-test.
- **GSM** (`arithmetic_reasoning`): arithmetic reasoning robustness.
- **ALGO** (`coin_change`, `shortest_path`, `wis`): algorithmic reasoning and greedy-vs-optimal structure.

### 3) Canonical outputs to open first

- **BW**
  - Probe 1 sweep: `results/BW_P1_RES_behavioral_sweep.csv`
  - Probe 2: `results/BW_P2_RES_cci.csv`, `results/BW_P2_RES_tep.csv`
  - Probe 3 triage: `results/BW_P3_RES_contamination_triage.csv`
  - Triangulation: `results/BW_P3_RES_triangulation_per_instance_*.csv`
- **GSM**
  - Probe 1 sweep: `results/GSM_P1_RES_behavioral_sweep_*.csv`
  - Probe 1 metrics: `results/GSM_P1_RES_*.csv`
  - Probe 2: `results/GSM_P2_RES_cci.csv`, `results/GSM_P2_RES_metrics_summary.csv`
  - Probe 3: `results/GSM_P3_RES_contamination_triage.csv`, `results/GSM_P3_RES_contamination_regression_*.txt`
- **ALGO**
  - Question bank: `data/problems/question_bank_algo.csv`
  - Probe 1 sweep: `results/ALGO_P1_RES_behavioral_sweep_*.csv`
  - Probe 1 metrics: `results/ALGO_P1_RES_metrics.csv`
  - Probe 2 phase outputs: `results/ALGO_P2_RES_phase1_*.csv`, `results/ALGO_P2_RES_phase2_normal.csv`, `results/ALGO_P2_RES_phase2_injected.csv`
  - Probe 2 metrics: `results/ALGO_P2_RES_metrics.csv`
  - Probe 3 triage: `results/ALGO_P3_RES_contamination.csv`
  - Triangulation: `results/ALGO_P3_RES_triangulation.csv`, `results/ALGO_P3_RES_regression.txt`

### 4) Where logs and run traces live

- `PROJECT_LOG.md`: high-level timeline, decisions, bugs, checkpoints.
- `results/`: probe outputs and metric tables.
- `logs/`: command/run logs where present.
- `figures/` or `results/figures/`: generated publication figures.

---

## What this project does

We test whether LLMs reason through problems or recognize them from training data. Three independent probes — surface invariance, plan-execution coupling, and contamination indexing — run on the same 45 problem instances. The core contribution is per-instance triangulation: when all three probes agree on the same diagnosis for an instance, that convergence is the evidence.

Full methodology is in `CHARTER.md`. This README covers only the codebase.

---

## Evaluation pipelines by family (clear run map)

### Blocksworld (BW) pipeline

1. **Probe 1**
   - Run: `scripts/BW_P1_SCR_run_behavioral_sweep.py`
   - Output: `results/BW_P1_RES_behavioral_sweep.csv`
2. **Probe 2**
   - Plan/execution: `scripts/BW_P2_SCR_run_cci.py`, `scripts/BW_P2_SCR_run_tep.py`
   - Outputs: `results/BW_P2_RES_cci.csv`, `results/BW_P2_RES_tep.csv`
3. **Probe 3 + Triangulation**
   - Triage: `scripts/BW_P3_SCR_run_contamination_triage.py`
   - Merge/regression: `scripts/BW_P3_SCR_run_triangulation.py`
   - Outputs: `results/BW_P3_RES_contamination_triage.csv`, `results/BW_P3_RES_triangulation_per_instance_*.csv`

### GSM pipeline

1. **Bank prep**
   - `scripts/GSM_PX_SCR_fix_question_bank.py`
   - `scripts/GSM_PX_SCR_generate_w6.py`
2. **Probe 1**
   - Sweep: `scripts/BW_P1_SCR_run_behavioral_sweep.py --family arithmetic_reasoning ...`
   - Metrics: `scripts/GSM_P1_SCR_compute_metrics.py`
3. **Probe 2**
   - Run: `scripts/GSM_P2_SCR_run_probe2.py`
   - Metrics: `scripts/GSM_P2_SCR_compute_metrics.py`
4. **Probe 3**
   - Triage: `scripts/BW_P3_SCR_run_contamination_triage.py --family arithmetic_reasoning ...`
   - Triangulation/regression: `scripts/BW_P3_SCR_run_triangulation.py --family arithmetic_reasoning ...`
5. **Figures**
   - `scripts/GSM_P1_FIG_generate.py`, `scripts/GSM_P2_FIG_generate.py`, `scripts/GSM_P3_FIG_generate.py`

### Algorithmic (ALGO) pipeline

1. **Bank prep and auditing**
   - Fixes: `scripts/ALGO_PX_SCR_fix_question_bank.py`
   - Backfills: `scripts/ALGO_PX_SCR_backfill_greedy_metadata.py`, `scripts/ALGO_PX_SCR_add_critical_step.py`
   - Audit: `scripts/ALGO_PX_SCR_audit_bank.py`
2. **Probe 1**
   - Sweep: `scripts/ALGO_P1_SCR_run_behavioral_sweep.py`
   - Metrics: `scripts/ALGO_P1_SCR_compute_metrics.py`
3. **Probe 2**
   - Phase 1: `scripts/ALGO_P2_SCR_run_phase1.py`
   - Phase 2: `scripts/ALGO_P2_SCR_run_phase2.py`
   - Metrics: `scripts/ALGO_P2_SCR_compute_metrics.py`
4. **Probe 3**
   - Triage/decomposition: `scripts/BW_P3_SCR_run_contamination_triage.py --family algorithmic --decompose-contamination ...`
   - Triangulation: `scripts/ALGO_P3_SCR_triangulation.py`
5. **Figures**
   - `scripts/ALGO_P1_FIG_generate.py`, `scripts/ALGO_P2_FIG_generate.py`, `scripts/ALGO_P3_FIG_generate.py`

---

## Exactly Which Result Files To Analyze

This section is the canonical analysis checklist. Use it as your "what to open next" guide.

### BW (Blocksworld / planning suite)

- **Probe 1 (behavioral)**
  - Primary: `results/BW_P1_RES_behavioral_sweep.csv`
  - Legacy compatibility artifact: `results/BW_RES_P1_behavioral_sweep.csv`
- **Probe 2 (process)**
  - `results/BW_P2_RES_phase1_plans.csv`
  - `results/BW_P2_RES_cci.csv`
  - `results/BW_P2_RES_tep.csv`
  - `results/BW_P2_RES_validity_comparison.csv`
  - `results/BW_P2_RES_validity_summary.txt`
  - Debug trace: `results/BW_P2_LOG_injection_trace.txt`
- **Probe 3 (contamination + triangulation)**
  - `results/BW_P3_RES_contamination_triage.csv`
  - `results/BW_P3_RES_triangulation_per_instance_claude37.csv`
  - `results/BW_P3_RES_triangulation_per_instance_gpt4o.csv`
  - `results/BW_P3_RES_triangulation_per_instance_llama8b.csv`
  - `results/BW_P3_RES_contamination_regression_claude37.txt`
  - `results/BW_P3_RES_contamination_regression_gpt4o.txt`
  - `results/BW_P3_RES_contamination_regression_llama8b.txt`

### GSM (arithmetic_reasoning)

- **Probe 1 (behavioral + metrics)**
  - `results/GSM_P1_RES_behavioral_sweep_claude.csv`
  - `results/GSM_P1_RES_behavioral_sweep_gpt4o.csv`
  - `results/GSM_P1_RES_behavioral_sweep_llama.csv`
  - `results/GSM_P1_RES_css.csv`
  - `results/GSM_P1_RES_var.csv`
  - `results/GSM_P1_RES_vri.csv`
  - `results/GSM_P1_RES_rcs.csv`
  - `results/GSM_P1_RES_rcs_by_difficulty.csv`
  - `results/GSM_P1_RES_step_count_sensitivity.csv`
  - `results/GSM_P1_RES_w4_gap.csv`
- **Probe 2**
  - `results/GSM_P2_RES_cci.csv`
  - `results/GSM_P2_RES_metrics_summary.csv`
  - `results/GSM_P2_LOG_human_approval_queue.csv`
- **Probe 3**
  - `results/GSM_P3_RES_contamination_triage.csv`
  - `results/GSM_P3_RES_triangulation_per_instance_claude.csv`
  - `results/GSM_P3_RES_triangulation_per_instance_gpt4o.csv`
  - `results/GSM_P3_RES_contamination_regression_claude.txt`
  - `results/GSM_P3_RES_contamination_regression_gpt4o.txt`

### ALGO (coin_change / shortest_path / wis)

- **Probe 1 (behavioral + metrics)**
  - `results/ALGO_P1_RES_behavioral_sweep_claude.csv`
  - `results/ALGO_P1_RES_behavioral_sweep_gpt4o.csv`
  - `results/ALGO_P1_RES_behavioral_sweep_llama.csv`
  - `results/ALGO_P1_RES_behavioral_sweep_mock.csv`
  - `results/ALGO_P1_RES_human_review_queue.csv`
  - `results/ALGO_P1_RES_metrics.csv`
- **Probe 2**
  - `results/ALGO_P2_RES_phase1_claude.csv`
  - `results/ALGO_P2_RES_phase1_gpt4o.csv`
  - `results/ALGO_P2_RES_phase1_llama.csv`
  - `results/ALGO_P2_RES_phase2_normal.csv`
  - `results/ALGO_P2_RES_phase2_injected.csv`
  - `results/ALGO_P2_RES_metrics.csv`
- **Probe 3**
  - `results/ALGO_P3_RES_contamination.csv`
  - `results/ALGO_P3_RES_triangulation.csv`
  - `results/ALGO_P3_RES_regression.txt`
  - `results/ALGO_PX_RES_bank_audit.csv` (bank quality gate; use before interpreting ALGO outputs)

---

## Script-to-Output Responsibility Map (complete)

### ALGO scripts

- `scripts/ALGO_PX_SCR_fix_question_bank.py`: fixes/normalizes ALGO bank rows and metadata.
- `scripts/ALGO_PX_SCR_backfill_greedy_metadata.py`: fills `greedy_succeeds` + `instance_type` in ALGO bank.
- `scripts/ALGO_PX_SCR_add_critical_step.py`: fills `critical_step_index` in ALGO bank.
- `scripts/ALGO_PX_SCR_audit_bank.py`: strict structural/content audit -> `results/ALGO_PX_RES_bank_audit.csv`.
- `scripts/ALGO_PX_SCR_generate_w6.py`: procedural W6 generation for ALGO rows.
- `scripts/ALGO_PX_SCR_verify_sp005.py`, `scripts/ALGO_PX_SCR_verify_wis005.py`: one-off solver verification utilities.
- `scripts/ALGO_P1_SCR_run_behavioral_sweep.py`: Probe 1 ALGO sweeps.
- `scripts/ALGO_P1_SCR_compute_metrics.py`: Probe 1 ALGO metric aggregation.
- `scripts/ALGO_P2_SCR_run_phase1.py`: Probe 2 Phase 1 strategy elicitation.
- `scripts/ALGO_P2_SCR_run_phase2.py`: Probe 2 Phase 2 stepwise execution + injection.
- `scripts/ALGO_P2_SCR_compute_metrics.py`: Probe 2 metric table generation.
- `scripts/BW_P3_SCR_run_contamination_triage.py`: shared triage runner also used for ALGO contamination decomposition.
- `scripts/ALGO_P3_SCR_triangulation.py`: ALGO-only cross-probe merge + labeling + regression summary.
- `scripts/ALGO_P1_FIG_generate.py`, `scripts/ALGO_P2_FIG_generate.py`, `scripts/ALGO_P3_FIG_generate.py`: ALGO figure generation.

### GSM scripts

- `scripts/GSM_PX_SCR_fix_question_bank.py`, `scripts/GSM_PX_SCR_generate_w6.py`: GSM bank prep.
- `scripts/GSM_P1_SCR_compute_metrics.py`: Probe 1 GSM metrics.
- `scripts/GSM_P2_SCR_run_probe2.py`, `scripts/GSM_P2_SCR_compute_metrics.py`: Probe 2 GSM run + summary.
- `scripts/GSM_P1_FIG_generate.py`, `scripts/GSM_P2_FIG_generate.py`, `scripts/GSM_P3_FIG_generate.py`: GSM figures.

### BW scripts

- `scripts/BW_P1_SCR_run_behavioral_sweep.py`: Probe 1 BW sweeps.
- `scripts/BW_P2_SCR_extract_phase1_plans.py`: Probe 2 plan extraction.
- `scripts/BW_P2_SCR_run_cci.py`, `scripts/BW_P2_SCR_run_tep.py`: Probe 2 BW metrics.
- `scripts/BW_P2_SCR_generate_figures.py`: BW Probe 2 figures.
- `scripts/BW_P3_SCR_run_contamination_triage.py`: Probe 3 triage.
- `scripts/BW_P3_SCR_run_triangulation.py`: Probe 3 triangulation/regression.

### Shared / support scripts

- `scripts/run_mechanistic_sweep.py`: mechanistic runs.
- `scripts/check_transformerlens_support.py`: mechanistic stack compatibility check.
- `scripts/test_api_keys.py`: endpoint readiness check.
- `scripts/maintenance/`: helper/admin utilities.

## How to read findings quickly

- **Retrieval-like evidence:** collapse on `W3`, positive contamination slope, high local-greedy behavior on adversarial rows.
- **Computation-like evidence:** robust structural variants (`W2/W4`), stable stepwise behavior, low contamination dependence.
- **Triangulation decision point:** prefer conclusions where Probe 1 + Probe 2 + Probe 3 agree for the same instance/model.

---

## Current repo state (Apr 2026)

- Active dataset is `data/problems/question_bank.csv` with unified schema:
  `problem_id, variant_type, problem_text, correct_answer, problem_family, problem_subtype, difficulty, contamination_pole, source, verifier_function, difficulty_params, notes`.
- Current bank has full Planning Suite coverage (`planning_suite` only):
  - 20 canonical instances total (15 Blocksworld + 5 Mystery Blocksworld)
  - Variants present: `W1=20`, `W2=20`, `W3=20`, `W4=20`, `W5=15`, `W6=25`
  - Total rows: `140`
- GSM + Algorithmic family support exists in verifier/metrics pipeline, but those families are not yet loaded into the active question bank.
- `results/` currently contains mixed historical outputs; treat these as working artifacts, not final paper-grade outputs, until one clean rerun is done from a frozen bank.
- Probe 2 metrics are implemented and in active use: `CCI` (`probes/behavioral/cci.py`) and `TEP` (`probes/behavioral/tep.py`).

## Artifact naming convention (BW-first)

To keep probe ownership and purpose obvious, use this pattern for scripts, results, figures, and logs:

- `BW_<PROBE>_<TYPE>_<PURPOSE>.<ext>`
- `<PROBE>`: `P1`, `P2`, `P3`, `PX` (cross-probe/shared)
- `<TYPE>`: `SCR` (script), `RES` (result), `FIG` (figure), `LOG` (log), `UTIL` (one-off utility)

Examples:

- `BW_P1_SCR_run_behavioral_sweep.py`
- `BW_P2_RES_cci.csv`
- `BW_P3_FIG_contamination_scatter.pdf`
- `BW_P3_LOG_contamination_triage_20260422.log`

This convention is the target for ongoing cleanup; some legacy files still use older names.

## Probe 3 Colab mechanistic artifacts

These two root-level artifacts are Probe 3 mechanistic outputs produced in Google Colab runs:

- `RvC Mechanistic Sweep.csv`
- `RvC Mechanistic Sweep.ipynb`

Keep them linked to Probe 3 in documentation and archive decisions.

---

## Repository structure

```
retrieval-vs-computation/
│
├── CHARTER.md                            # Full research design (read this first)
├── README.md                             # This file
├── PROJECT_LOG.md                        # Bugs, decisions, blockers, work log
├── requirements.txt                      # Project and dev dependencies (black, ruff, pytest)
├── .env.example                          # Secret template — copy to .env
├── .gitignore
├── .cursorrules                          # Cursor IDE conventions
├── Makefile                              # Standard commands
│
├── configs/
│   ├── models.yaml                       # Model names, endpoints, GPU requirements
│   ├── probes.yaml                       # Variant types and metrics per probe
│   └── paths.yaml                        # Data and results directory paths
│
├── data/problems/
│   ├── probe1_instances.csv              # 45 canonical problems — PENDING (Adya)
│   ├── probe1_variants.csv               # W1-W6 variants — PENDING (team)
│   └── probe2_instances.csv              # 8 Blocksworld instances — PENDING
│
├── probes/
│   ├── common/
│   │   ├── io.py                         # CSV load/write helpers
│   │   ├── parsers.py                    # Extract answers from raw model output
│   │   └── stats.py                      # Bootstrap CI, Wilcoxon, effect size
│   │
│   ├── contamination/
│   │   ├── infinigram_client.py          # Infini-gram API wrapper with disk cache
│   │   ├── score.py                      # Contamination scoring (n-gram fingerprint)
│   │   ├── triage.py                     # Core triage runner
│   │   └── verify.py                     # Answer verification per family
│   │
│   ├── behavioral/
│   │   ├── mock_client.py                # Fake client for local testing (Layer 1)
│   │   ├── anthropic_client.py           # Anthropic API client (Layer 2, dormant)
│   │   ├── openai_client.py              # OpenRouter client (Layer 2, dormant)
│   │   ├── css.py                        # CSS: variant answer consistency
│   │   ├── rcs.py                        # RCS: W5 reversal correctness
│   │   ├── cas.py                        # CAS: consistent failure on hard-tier
│   │   ├── cci.py                        # CCI: plan-execution coupling (Probe 2)
│   │   └── tep.py                        # TEP: state corruption response (Probe 2)
│   │
│   ├── mechanistic/
│   │   ├── load_model.py                 # Qwen2.5-7B via TransformerLens
│   │   ├── activations.py                # Residual stream extraction
│   │   ├── similarity.py                 # Layer-wise cosine similarity
│   │   ├── logit_lens.py                 # Per-layer token predictions
│   │   ├── patching.py                   # Activation patching + random control
│   │   └── sanity_check.py               # Gate 2: IOI-style pipeline verification
│   │
│   └── triangulation/
│       └── per_instance.py               # Cross-probe alignment per instance
│
├── scripts/
│   ├── test_api_keys.py                  # Verify keys before sweeps
│   ├── check_gpu.py                      # Verify GPU before mechanistic runs
│   ├── check_transformerlens_support.py  # Confirm Qwen2.5-7B in TL registry
│   ├── generate_w6_variants.py           # Procedural W6 generation (seeded)
│   ├── BW_P3_SCR_run_contamination_triage.py  # Probe 3 contamination entrypoint
│   ├── BW_P1_SCR_run_behavioral_sweep.py      # Probe 1 behavioral entrypoint
│   ├── run_mechanistic_sweep.py          # Phase 3-4 mechanistic entrypoint
│   └── BW_P3_SCR_run_triangulation.py         # Final cross-probe analysis
│
├── notebooks/
│   └── probe1_triage_plot.py             # Phase 1 go/no-go scatter plot
│
├── results/                              # Output CSVs (gitignored)
├── figures/                              # Generated plots (gitignored)
│
├── team/
│   ├── shasshy.md                        # Shaswat's task list
│   ├── nandini.md                        # Nandini's task list
│   ├── CONTRIBUTING_VARIANTS.md          # Variant writing rules and examples
│   └── REVIEW_PROTOCOL.md                # Cross-review process
│
└── tests/
    ├── test_verifiers.py
    ├── test_parsers.py
    ├── test_stats.py
    └── test_mock_sweep.py
```

### What each top-level area does

- `probes/common/`: shared loaders/parsers/stats used by all probes.
- `probes/contamination/`: Infini-gram retrieval counts + contamination scoring + answer verification.
- `probes/behavioral/`: Probe 1 metrics (`css`, `rcs`, `cas`) and Probe 2 metrics (`cci`, `tep`) plus model clients.
- `probes/mechanistic/`: Probe 3 internals (activation extraction, similarity, logit lens, patching).
- `probes/triangulation/`: per-instance convergence diagnosis across probes.
- `scripts/`: runnable entrypoints and dataset utilities (generation, migration, scoring, extraction).
- `data/problems/`: canonical and variant bank files; `question_bank.csv` is the single source of truth.
- `results/`: generated probe outputs and regressions used by downstream analysis.
- `team/`: writing/review process docs for dataset and experiment governance.

---

## How to run things

### First-time setup
```bash
cp .env.example .env                    # fill in API keys
make setup                              # install all dependencies
python scripts/test_api_keys.py        # verify Infini-gram reachable (free, no key)
```

### Tests — no API keys or GPU needed
```bash
make test
```

### Phase 1 — Contamination triage
```bash
make triage                             # full run

# Partial/safe runs — already-scored rows skipped automatically
python scripts/BW_P3_SCR_run_contamination_triage.py --limit 5
python scripts/BW_P3_SCR_run_contamination_triage.py --family blocksworld
python scripts/BW_P3_SCR_run_contamination_triage.py --no-resume   # rescore everything

# Gate 1 plot
python notebooks/probe1_triage_plot.py
```

### Practical run order for current Blocksworld bank
```bash
# 0) Validate schema / row counts manually first
python - <<'PY'
import csv, collections
rows=list(csv.DictReader(open('data/problems/question_bank.csv', newline='', encoding='utf-8')))
print('rows', len(rows))
print(collections.Counter(r['variant_type'] for r in rows))
PY

# 1) Probe 1 behavioral (resume-safe, model by model)
python scripts/BW_P1_SCR_run_behavioral_sweep.py --model anthropic/claude-3.7-sonnet --resume
python scripts/BW_P1_SCR_run_behavioral_sweep.py --model openai/gpt-4o --resume
python scripts/BW_P1_SCR_run_behavioral_sweep.py --model meta-llama/llama-3.1-8b-instruct --resume

# 2) Probe 2 prep + runs
python scripts/BW_P2_SCR_extract_phase1_plans.py
python scripts/BW_P2_SCR_run_cci.py --models anthropic/claude-3.7-sonnet openai/gpt-4o meta-llama/llama-3.1-8b-instruct --resume
python scripts/BW_P2_SCR_run_tep.py --models anthropic/claude-3.7-sonnet openai/gpt-4o meta-llama/llama-3.1-8b-instruct --resume

# 3) Contamination + triangulation
python scripts/BW_P3_SCR_run_contamination_triage.py --resume
python scripts/BW_P3_SCR_run_triangulation.py --behavioral-model anthropic/claude-3.7-sonnet
python scripts/BW_P3_SCR_run_triangulation.py --behavioral-model openai/gpt-4o --output results/BW_P3_RES_triangulation_per_instance_gpt4o.csv --regression-output results/BW_P3_RES_contamination_regression_gpt4o.txt
python scripts/BW_P3_SCR_run_triangulation.py --behavioral-model meta-llama/llama-3.1-8b-instruct --output results/BW_P3_RES_triangulation_per_instance_llama8b.csv --regression-output results/BW_P3_RES_contamination_regression_llama8b.txt
```

### Phase 3-4 — Behavioral sweep
```bash
# Dry run: zero API credits, uses mock_client
python scripts/BW_P1_SCR_run_behavioral_sweep.py --dry-run --limit 5

# Real run (needs API key in .env)
python scripts/BW_P1_SCR_run_behavioral_sweep.py --model claude-sonnet-3-7 --family blocksworld
python scripts/BW_P1_SCR_run_behavioral_sweep.py --resume    # safe to interrupt and resume
```

### Phase 3-4 — Mechanistic sweep
```bash
# Local dry run: GPT-2 on CPU, catches most pipeline bugs without GPU
python scripts/run_mechanistic_sweep.py --dry-run

# Real run (needs GPU)
python scripts/run_mechanistic_sweep.py --family blocksworld --limit 8
python scripts/run_mechanistic_sweep.py --resume
```

### Final triangulation
```bash
make triangulate
```

### Makefile targets
| Command | What it does |
|---------|-------------|
| `make setup` | Install all dependencies |
| `make test` | Run full test suite |
| `make lint` | Check formatting (black + ruff) |
| `make format` | Auto-format |
| `make triage` | Phase 1 contamination triage |
| `make sweep` | Behavioral sweep |
| `make mechanistic` | Mechanistic sweep |
| `make triangulate` | Final triangulation analysis |
| `make clean` | Remove cache and compiled files |

---

## Layer 1 vs Layer 2

**Layer 1 — runs locally now, no credits or GPU:**
- `probes/common/` — io, parsers, stats
- `probes/contamination/` — Infini-gram is a free public API
- `probes/behavioral/` metric modules — pure computation, no API calls
- `mock_client.py` — end-to-end pipeline testing
- All scripts with `--dry-run` or `--limit`
- Full test suite

**Layer 2 — code written, execution deferred:**
- `anthropic_client.py`, `openai_client.py` — need API keys in `.env`
- All of `probes/mechanistic/` real runs — need 24GB VRAM GPU
- `run_behavioral_sweep.py` on real models
- `run_mechanistic_sweep.py` on Qwen

---

## Cleanup plan (recommended before final paper runs)

- Freeze `question_bank.csv` and avoid manual edits between runs; only scripted changes.
- Normalize naming conventions in `results/` (single canonical output per probe/model, archive old files to `results/archive/`).
- Keep one authoritative model naming convention (`meta-llama/llama-3.1-8b-instruct` vs legacy aliases).
- Add a `scripts/audit_question_bank.py` and `scripts/audit_results_coverage.py` so missing rows are detected before spending API credits.
- After Planning Suite finalization, ingest GSM + Algorithmic families into `question_bank.csv` using the same 7-row (`canonical` + `W1..W6`) structure.

---

## Maintenance map (what to keep, what to clean)

### Core files to keep

- `data/problems/question_bank.csv`: source of truth for all canonical + variant prompts and answers.
- `results/BW_P1_RES_behavioral_sweep.csv`: Probe 1 run log (all model responses and correctness flags).
- `results/BW_P2_RES_cci.csv`: Probe 2A execution-coupling outputs.
- `results/BW_P2_RES_tep.csv`: Probe 2B perturbation outputs.
- `results/BW_P3_RES_contamination_triage.csv`: contamination scores for canonical set.
- `results/BW_P3_RES_triangulation_per_instance_*.csv`: merged per-instance diagnosis.
- `scripts/BW_P1_SCR_run_behavioral_sweep.py`: primary Probe 1 runner.
- `scripts/BW_P2_SCR_run_cci.py`, `scripts/BW_P2_SCR_run_tep.py`: Probe 2 runners.
- `scripts/BW_P3_SCR_run_contamination_triage.py`: contamination runner.
- `scripts/BW_P3_SCR_run_triangulation.py`: final merge + regression analysis.
- `probes/contamination/verify.py`: verifier logic used across scoring and gates.

### Utility/one-off scripts (safe cleanup candidates after freeze)

- `fix_csv.py`: one-time question bank repair utility.
- `scripts/run_probe1_repair_and_batches.py`: one-off orchestration script for targeted reruns and skip-guarded fills.
- `scripts/generate_mbw_w5_variants.py`: MBW W5 generation utility (keep only if future regeneration needed).
- `scripts/generate_w6_variants.py`: W6 generation utility (keep if regenerating procedural rows again).
- `scripts/temp_flag_records.py`, `scripts/propagate_answers.py`, `scripts/translate_w3_answers.py`: migration/repair helpers; archive if no longer used.

### Figures and analysis scripts

- `scripts/BW_P2_SCR_generate_figures.py`: Probe 2 figure bundle using `results/BW_P2_RES_*.csv`.
- `analysis/figures/*.py`: modular figure scripts for Section 8 graphs.
- `analysis/figures/output/`: generated publication figures (PNG/PDF).

### Path/naming note

- The repo now supports both legacy and prefixed output names in several scripts (fallback logic added).
- Current active outputs use probe-prefixed names (`results/BW_P1_RES_*`, `results/BW_P2_RES_*`, `results/BW_P3_RES_*`).
- Before final archival, choose one naming convention and migrate consistently.

### Suggested cleanup workflow

1. Freeze `question_bank.csv`.
2. Archive old logs/experimental outputs to `results/archive/` (keep final outputs at root `results/`).
3. Keep only one canonical result file per probe/model.
4. Keep scripts that are runners or shared utilities; move one-off repair scripts to `scripts/archive/`.
5. Rerun triangulation once after final contamination + sweep freeze, then tag that output set as final.

## Problem families and variant types

Three families, 15 problems each, 45 total. Full justification in `CHARTER.md` Section 10.

| Family | Sub-types | Probe 1 | Probe 2 | Probe 3 |
|--------|-----------|---------|---------|---------|
| Planning Suite | BW (8), Logistics (4), Mystery BW (3) | All | BW only | All |
| Arithmetic | GSM-Symbolic (8), GSM-P1/P2 (4), GSM-NoOp (3) | All | None | All |
| Algorithmic | SP (4), WIS (4), Coin Change (4), Knapsack (3) | All | None | All |

Variant types (W1-W6):

| Code | Name | Applies to | Answer changes? |
|------|------|------------|----------------|
| W1 | Lexical paraphrase | All families | No |
| W2 | Structural reformat | All families | No |
| W3 | Entity rename | All families | No |
| W4 | Formal notation | Planning (BW+Logistics) + Algorithmic | No |
| W6 | Procedural regeneration | All families | No (new instance, same structure) |
| W5 | Reversal | BW, Shortest Path, Coin Change only | **Yes** |

W5 is never pooled into CSS. It uses its own metric RCS (`probes/behavioral/rcs.py`).

---

## Metrics

| Metric | Module | Probe | What it measures |
|--------|--------|-------|-----------------|
| CSS | `behavioral/css.py` | 1 | Fraction of W1-W4, W6 variants with matching answer |
| RCS | `behavioral/rcs.py` | 1 | W5 reversal answer correctness |
| CAS | `behavioral/cas.py` | 1 | Failure consistency on hard-tier instances |
| CCI | `behavioral/cci.py` | 2 | Plan-execution step match rate |
| TEP | `behavioral/tep.py` | 2 | Adaptation after mid-execution state corruption |

All aggregate metrics use bootstrap 95% CI (10,000 resamples) from `probes/common/stats.py`.

---

## Statistical requirements (non-negotiable per CHARTER.md)

- Bootstrap 95% CI, 10,000 resamples on every aggregate number
- Wilcoxon signed-rank test for all paired comparisons
- Problem-family fixed effects in contamination OLS regression
- Random-position control for every activation patching experiment
- Effect sizes reported alongside p-values

---

## Compute budget

| Task | Local | GPU | Notes |
|------|-------|-----|-------|
| Contamination triage | Yes | No | Infini-gram free |
| Behavioral sweep (mock) | Yes | No | Dry-run only |
| Behavioral sweep (real) | No | No | API keys needed |
| Mechanistic dry-run | Yes | No | GPT-2 via `--dry-run` |
| Mechanistic real run | No | Yes | 24GB VRAM min (Qwen2.5-7B fp16) |

GPU options if BITS cluster unavailable: RunPod A100 (~$1.50/hr), Lambda Labs, Vast.ai.
Estimated GPU hours: ~50 including debugging.
Estimated API cost: ~$100-150 for full behavioral sweep on 2 closed models.
