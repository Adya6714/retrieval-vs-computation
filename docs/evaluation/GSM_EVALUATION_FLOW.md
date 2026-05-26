# GSM Evaluation Flow

This document is the exact execution and interpretation map for GSM evaluation in this repo.

---

## 1) Question Bank (what file, why)

- Primary GSM bank: `data/problems/question_bank_gsm.csv`
- Legacy compatibility path may exist, but canonical path for current runs is:
  - `data/problems/question_bank_gsm.csv`

Why this file:
- Contains GSM canonical/variant rows in the unified schema used by shared runners.

Bank preparation scripts:
- `scripts/GSM_PX_SCR_fix_question_bank.py`
- `scripts/GSM_PX_SCR_generate_w6.py`

---

## 2) First Steps (pre-run checks)

1. Validate GSM bank schema.
2. Run environment/API checks:
   - `scripts/test_api_keys.py`
3. If needed, normalize/fix bank:
   - `scripts/GSM_PX_SCR_fix_question_bank.py`
4. Optional W6 generation:
   - `scripts/GSM_PX_SCR_generate_w6.py`

---

## 3) Probe 1 (behavior + GSM-specific metrics)

### Behavioral sweep
- Script: `scripts/BW_P1_SCR_run_behavioral_sweep.py` with GSM family/bank flags
- Outputs:
  - `results/raw/GSM_P1_behavioral_claude.csv`
  - `results/raw/GSM_P1_behavioral_gpt4o.csv`
  - `results/raw/GSM_P1_behavioral_llama.csv`

### Probe 1 metrics
- Script: `scripts/GSM_P1_SCR_compute_metrics.py`
- Outputs:
  - `results/derived/GSM_P1_metrics.csv`
  - `results/GSM_P1_RES_var.csv`
  - `results/GSM_P1_RES_vri.csv`
  - `results/GSM_P1_RES_rcs.csv`
  - `results/GSM_P1_RES_rcs_by_difficulty.csv`
  - `results/GSM_P1_RES_step_count_sensitivity.csv`
  - `results/GSM_P1_RES_w4_gap.csv`

Why:
- GSM has dedicated reporting tables beyond baseline sweep correctness.

---

## 4) Probe 2 (GSM-specific)

### Runner
- Script: `scripts/GSM_P2_SCR_run_probe2.py`

### Outputs
- `results/raw/GSM_P2_cci.csv`
- `results/raw/GSM_P2_review_queue.csv`

### Metrics summary
- Script: `scripts/GSM_P2_SCR_compute_metrics.py`
- Output: `results/derived/GSM_P2_metrics.csv`

### Key internals
- `probes/behavioral/gsm_metrics.py` (metric helpers)
- `probes/behavioral/cci.py` / `tep.py` (shared metric modules)

---

## 5) Probe 3 (contamination + triangulation)

### Contamination triage
- Script: `scripts/BW_P3_SCR_run_contamination_triage.py` (shared runner)
- Output: `results/raw/GSM_P3_contamination.csv`

### Triangulation/regression
- Script: `scripts/BW_P3_SCR_run_triangulation.py` with GSM inputs/family
- Outputs:
  - `results/derived/GSM_P3_triangulation_claude.csv`
  - `results/derived/GSM_P3_triangulation_gpt4o.csv`
  - `results/GSM_P3_RES_contamination_regression_claude.txt`
  - `results/GSM_P3_RES_contamination_regression_gpt4o.txt`

---

## 6) Triangulation interpretation

What to inspect first:
1. `GSM_P1_RES_var.csv` and `GSM_P1_RES_vri.csv` (surface/structure robustness)
2. `GSM_P2_RES_cci.csv` and summary (process coupling)
3. `GSM_P3_RES_contamination_triage.csv` (data proximity)
4. `GSM_P3_RES_triangulation_per_instance_*.csv` (converged diagnosis)

---

## 7) Conclusion / defense quick answers

- **Q: What is the GSM source-of-truth bank?**  
  `data/problems/question_bank_gsm.csv`.
- **Q: Where are raw model outputs for GSM Probe 1?**  
  `results/GSM_P1_RES_behavioral_sweep_*.csv`.
- **Q: Where is GSM Probe 2 evidence?**  
  `results/raw/GSM_P2_cci.csv` + `results/derived/GSM_P2_metrics.csv`.
- **Q: Which files are final GSM convergence evidence?**  
  `results/GSM_P3_RES_triangulation_per_instance_*.csv`.
