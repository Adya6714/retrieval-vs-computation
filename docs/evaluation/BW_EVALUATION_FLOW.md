# Blocksworld (BW) Evaluation Flow

This document is the exact execution and interpretation map for BW evaluation in this repo.

---

## 1) Question Bank (what file, why)

- Primary bank: `data/problems/question_bank_bw.csv`
- Rows used for BW runs:
  - `problem_family` / `problem_subtype` under planning-suite categories
  - `variant_type` in `{canonical, W1, W2, W3, W4, W5, W6}`
- Why this file:
  - It is the unified schema used by BW runners and verifiers.
  - It includes canonical prompts and perturbation variants used in Probe 1 and downstream probes.

Related schema guardrails:
- `probes/common/io.py` (loading + validation helpers)
- `probes/contamination/verify.py` (family-specific answer verification)

---

## 2) First Steps (pre-run checks)

1. Validate bank exists and has required columns.
2. Run API/environment checks:
   - `scripts/test_api_keys.py`
3. Optional smoke run for contamination:
   - `scripts/BW_P3_SCR_run_contamination_triage.py --limit 5`
4. Confirm output paths in `results/` are clean (or use `--resume` intentionally).

Why:
- Prevent expensive reruns due to schema or endpoint failure.

---

## 3) Probe 1 (behavioral invariance)

### Runner
- Script: `scripts/BW_P1_SCR_run_behavioral_sweep.py`

### Output
- `results/raw/BW_P1_behavioral.csv`

### What it does
- Queries model per `(problem_id, variant_type)`.
- Verifies answers via family verifier.
- Stores raw answer + correctness flags.

### Key internals
- clients: `probes/behavioral/openai_client.py`, `probes/behavioral/anthropic_client.py`
- verifier: `probes/contamination/verify.py`
- metric modules used later:
  - `probes/behavioral/css.py`
  - `probes/behavioral/rcs.py`
  - `probes/behavioral/cas.py`

---

## 4) Probe 2 (plan/execution + perturbation)

### Step A: Extract phase-1 plans
- Script: `scripts/BW_P2_SCR_extract_phase1_plans.py`
- Output: `results/raw/BW_P2_plans.csv`

### Step B: CCI run
- Script: `scripts/BW_P2_SCR_run_cci.py`
- Output: `results/raw/BW_P2_cci.csv`

### Step C: TEP run
- Script: `scripts/BW_P2_SCR_run_tep.py`
- Output: `results/raw/BW_P2_tep.csv`
- Optional debug trace:
  - `results/BW_P2_LOG_injection_trace.txt`

### Why this decomposition
- CCI captures declared-plan vs execution consistency.
- TEP captures trajectory sensitivity after state corruption.

### Key BW planning/state functions
- `probes/behavioral/bw_cci_pipeline.py`
  - `parse_pddl()`
  - `execute_action()`
  - `make_turn1_prompt()`
  - `make_followup_prompt()`
  - `goal_reached()`
  - `state_to_narrative()`
  - `seeded_inject_error()`

### Probe 2 metrics helper modules
- `probes/behavioral/cci.py` (`compute_cci`)
- `probes/behavioral/tep.py` (`compute_tep`)

---

## 5) Probe 3 (contamination)

### Runner
- Script: `scripts/BW_P3_SCR_run_contamination_triage.py`

### Output
- `results/raw/BW_P3_contamination.csv`

### What it does
- Scores contamination via Infini-gram n-gram matching.
- Uses resume-safe row skipping by `problem_id`.

### Key internals
- `probes/contamination/infinigram_client.py`
- `probes/contamination/score.py` (`score_problem`)

---

## 6) Triangulation (merge all probes)

### Runner
- Script: `scripts/BW_P3_SCR_run_triangulation.py`

### Outputs
- `results/derived/BW_P3_triangulation_claude.csv`
- `results/derived/BW_P3_triangulation_gpt4o.csv`
- `results/derived/BW_P3_triangulation_llama.csv`
- `results/BW_P3_RES_contamination_regression_*.txt`

### What it does
- Merges Probe 1 + Probe 2 + Probe 3 signals by instance.
- Produces per-instance diagnosis and contamination regression summaries.

Key merge/diagnosis helpers:
- `probes/triangulation/per_instance.py`

---

## 7) Conclusion / defense quick answers

- **Q: Which file is BW source-of-truth?**  
  `data/problems/question_bank_bw.csv`.
- **Q: Where is Probe 2 state logic defined?**  
  `probes/behavioral/bw_cci_pipeline.py`.
- **Q: Which files prove Probe 2 happened?**  
  `results/raw/BW_P2_plans.csv`, `results/raw/BW_P2_cci.csv`, `results/raw/BW_P2_tep.csv`.
- **Q: Which files are final BW evidence for paper tables?**  
  `results/BW_P3_RES_triangulation_per_instance_*.csv` + regression txts.
