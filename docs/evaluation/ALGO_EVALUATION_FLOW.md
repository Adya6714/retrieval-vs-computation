# Algorithmic (ALGO) Evaluation Flow

This is the exact step-by-step map for ALGO (`coin_change`, `shortest_path`, `wis`) so you can defend process details in review/professor discussions.

---

## 1) Question Bank (what file, why)

- Primary ALGO bank: `data/problems/question_bank_algo.csv`
- Required fields used by pipeline:
  - `problem_id`
  - `problem_subtype` (`coin_change`, `shortest_path`, `wis`)
  - `variant_type` (`canonical`, `W1..W6`)
  - `difficulty_params` JSON (critical metadata carrier)

Why this file:
- ALGO scripts are designed around structured `difficulty_params` for verifier correctness, phase-2 state prompts, and injection step control.

Bank prep/quality scripts:
- `scripts/ALGO_PX_SCR_fix_question_bank.py`
- `scripts/ALGO_PX_SCR_backfill_greedy_metadata.py`
- `scripts/ALGO_PX_SCR_add_critical_step.py`
- `scripts/ALGO_PX_SCR_audit_bank.py`
- `scripts/ALGO_PX_SCR_generate_w6.py`

Audit output:
- `results/ALGO_PX_RES_bank_audit.csv`

---

## 2) First Steps (must-do before probe runs)

1. Run fix/backfill scripts as needed.
2. Run strict audit:
   - `scripts/ALGO_PX_SCR_audit_bank.py`
3. Confirm audit says pass before expensive probe runs.
4. Ensure `critical_step_index` exists in bank for adversarial rows.

Why:
- Probe 2 injected condition depends directly on `difficulty_params.critical_step_index`.
- Missing/malformed params will invalidate phase-2 and derived metrics.

---

## 3) Probe 1 (behavioral robustness)

### Runner
- `scripts/ALGO_P1_SCR_run_behavioral_sweep.py`

### Outputs
- `results/raw/ALGO_P1_behavioral_claude.csv`
- `results/raw/ALGO_P1_behavioral_gpt4o.csv`
- `results/raw/ALGO_P1_behavioral_llama.csv`
- `results/ALGO_P1_RES_behavioral_sweep_mock.csv`
- `results/raw/ALGO_P1_review_queue.csv`

### Metrics
- Script: `scripts/ALGO_P1_SCR_compute_metrics.py`
- Output: `results/derived/ALGO_P1_metrics.csv`

### Key verifier internals
- `probes/contamination/verify_algo.py`
  - `verify_coinchange()`
  - `verify_coinchange_scoops()`
  - `verify_sp()`
  - `verify_wis()`
  - dispatcher: `verify_algo()`

---

## 4) Probe 2 (process: Phase 1 + Phase 2)

### 4.1 Phase 1 (strategy declaration)

- Script: `scripts/ALGO_P2_SCR_run_phase1.py`
- Outputs:
  - `results/raw/ALGO_P2_phase1_claude.csv`
  - `results/raw/ALGO_P2_phase1_gpt4o.csv`
  - `results/raw/ALGO_P2_phase1_llama.csv`

What it captures:
- `stated_algorithm`
- `greedy_assessment_correct`
- `predicted_first_decision`
- `critical_point_identified`
- `phase1_parseable`

Key parsing functions:
- `_parse_phase1_fields()`
- `_split_answers()`
- `_parse_yes_no()`
- `_extract_algorithm()`
- `_critical_match()`

### 4.2 Phase 2 (step-by-step execution)

- Script: `scripts/ALGO_P2_SCR_run_phase2.py`
- Outputs:
  - `results/raw/ALGO_P2_phase2_normal.csv`
  - `results/raw/ALGO_P2_phase2_injected.csv`

What it does:
- Generates step prompts from current state (subtype-specific).
- Enforces/labels response structure (`Decision` + `Reason`).
- Classifies response type before parsing:
  - `compliant`, `full_solution_dump`, `partial_compliance`, `refusal`, `format_ignored`
- Uses `critical_step_index` from bank for adversarial injection point.
- Emits reasoning categories per step.

Key functions for defense questions:
- `classify_reasoning_type()`
- `parse_decision_reason()`
- `_cc_prompt()`, `_sp_prompt()`, `_wis_prompt()`
- `run_one()` (per-instance step runner)
- `_done_pairs()` (resume behavior)

### 4.3 Why Phase 1 vs Phase 2 is “difficult” and how we measured it

If asked “Why is plan measurement hard?”:
- Phase 1 is a **declarative strategy statement** (coarse, text-heavy).
- Phase 2 is **stateful local decisions** at each step.
- They are different granularity levels; consistency must be inferred via structured fields.

How this repo handles that:
- Phase 1 extracts normalized fields (`stated_algorithm`, `predicted_first_decision`, `critical_point_identified`).
- Phase 2 logs each step’s parsed decision + reasoning type + compliance status.
- `scripts/ALGO_P2_SCR_compute_metrics.py` computes:
  - `CCI_algorithm`
  - `CCI_first_decision`
  - `CCI_critical`
  - `CCI_composite`
  - `TEP_refined`
  - `FDI`, `SC`, `RDI`, `RTDA`

Metrics output:
- `results/derived/ALGO_P2_metrics.csv`

---

## 5) Probe 3 (contamination decomposition)

### Runner
- `scripts/BW_P3_SCR_run_contamination_triage.py` (shared triage script)

Run with ALGO settings:
- `--family algorithmic`
- `--bank-path data/problems/question_bank_algo.csv`
- `--decompose-contamination`

Output:
- `results/raw/ALGO_P3_contamination.csv`

Important fields:
- `template_contamination_score`
- `instance_contamination_score`
- `difficulty_numeric`

Key internals:
- `probes/contamination/score.py` (`score_problem`)
- `probes/contamination/infinigram_client.py` (query + retry + cache)

---

## 6) Triangulation (final ALGO diagnosis)

### Runner
- `scripts/ALGO_P3_SCR_triangulation.py`

### Outputs
- `results/derived/ALGO_P3_triangulation.csv` (per-instance merged diagnosis)
- `results/ALGO_P3_RES_regression.txt` (OLS + bootstrap summary + table section)

What triangulation merges:
- Probe 1 behavioral outputs
- Probe 2 metrics
- Probe 3 contamination
- bank metadata

Core labels generated:
- `retrieval_signal`
- `computation_signal`
- `mixed`
- `ambiguous`

---

## 7) Conclusion / defense quick answers

- **Q: Which bank file is ALGO source-of-truth?**  
  `data/problems/question_bank_algo.csv`.
- **Q: Where is plan declaration measured?**  
  `results/ALGO_P2_RES_phase1_*.csv` from `ALGO_P2_SCR_run_phase1.py`.
- **Q: Where is stepwise execution measured?**  
  `results/raw/ALGO_P2_phase2_normal.csv` and `..._injected.csv` from `ALGO_P2_SCR_run_phase2.py`.
- **Q: How do we connect Phase 1 and Phase 2?**  
  `scripts/ALGO_P2_SCR_compute_metrics.py` via CCI/TEP/FDI/SC/RDI metrics.
- **Q: Where is final per-instance ALGO diagnosis?**  
  `results/derived/ALGO_P3_triangulation.csv`.
