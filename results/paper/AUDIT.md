# Final Consolidation Audit

## Step 1 — ALGO bank fixes

Input: `data/problems/question_bank_algo.csv`  
Output: `data/problems/question_bank_algo_fixed.csv`

- Added/updated `original_correct_answer_buggy`, `instance_type`, and `greedy_succeeds` top-level columns.
- Recomputed DP-optimal `correct_answer` for `WIS_003` and `WIS_004` across all variants.
- Renamed verifier typo `veryify_WIS` -> `verify_WIS`.
- Normalized `difficulty` casing to lowercase.

Diff summary from run:

- `rows_with_wis_answer_fix`: 10
- `verifier_function_renamed`: 25
- `difficulty_lowercased`: 75
- `instance_type_extracted`: 100
- `greedy_succeeds_extracted`: 100

Before/after canonical examples:

- `WIS_003`: `Selected: {1, 2, 3, 4, 10, 11, 13}, Total: 55` -> `Selected: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}, Total: 119`
- `WIS_004`: `Selected: {0, 1, 3, 6, 9, 11, 12}, Total: 37` -> `Selected: {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, Total: 77`

Verifier typo status:

- old typo count (`veryify_WIS`) in original bank: 25
- old typo count (`veryify_WIS`) in fixed bank: 0

## Step 2 — GSM bank fix

Input: `data/problems/gsm_question_bank.csv`  
Output: `data/problems/gsm_question_bank_fixed.csv`

- Set `contamination_pole=high` for all `problem_subtype=gsm_symbolic` rows (all variants).
- Normalized `variant_type` to lowercase.

Diff summary from run:

- `contamination_pole_rows_changed`: 42
- `variant_type_rows_changed`: 0
- `gsm_symbolic` rows in fixed bank with `high`: 105 / 105

## Step 3 — Probe 1 metric recompute

Commands executed:

- `python scripts/BW_P1_SCR_compute_metrics.py --output results/final/`
- `python scripts/GSM_P1_SCR_compute_metrics.py --bank data/problems/gsm_question_bank_fixed.csv --output results/final/`
- `python scripts/ALGO_P1_SCR_compute_metrics.py --bank data/problems/question_bank_algo_fixed.csv --output results/final/`

Outputs written:

- `results/final/BW_P1_RES_metrics.csv` (30 rows)
- `results/final/GSM_P1_RES_var.csv` (21 rows)
- `results/final/GSM_P1_RES_css.csv` (60 rows)
- `results/final/GSM_P1_RES_rcs.csv` (60 rows)
- `results/final/GSM_P1_RES_w4_gap.csv` (60 rows)
- `results/final/GSM_P1_RES_vri.csv` (3 rows)
- `results/final/GSM_P1_RES_rcs_by_difficulty.csv` (9 rows)
- `results/final/GSM_P1_RES_step_count_sensitivity.csv` (6 rows)
- `results/final/ALGO_P1_RES_metrics.csv` (158 rows)

## Step 4 — Cross-family CSS regressions

Script created and executed:

- `scripts/consolidate/run_css_regressions.py`

Outputs:

- `results/final/cross_family_regression.csv`
- `results/final/cross_family_regression.txt`

Includes per-family/per-model:

- OLS slope for `CSS ~ contamination_score`
- 10,000-resample bootstrap CI
- OLS p-value + bootstrap p-value
- Cook's D high leverage flags (`Cook's D > 4/N`)

## Step 5 — Cross-family summary table

Script created and executed:

- `scripts/consolidate/make_table1.py`

Output:

- `results/final/TABLE1_cross_family.csv` (9 rows)

## Step 6 — Figure regeneration + figure logic fixes

Updated figure logic:

- `scripts/ALGO_P1_FIG_generate.py`
  - VAR heatmap now includes bootstrap CI annotation per cell.
  - Cross-family heuristic figure now uses per-model, per-subtype GSS (CC/SP/WIS distinct).
  - Uses fixed ALGO bank + final metrics when available.
- `scripts/GSM_P1_FIG_generate.py`
  - VAR heatmap now includes bootstrap CI annotation per cell.

Regeneration commands executed:

- `python scripts/ALGO_P1_FIG_generate.py`
- `python scripts/GSM_P1_FIG_generate.py`
- `python scripts/GSM_P3_FIG_generate.py`
- `python scripts/ALGO_P3_FIG_generate.py`
- `python scripts/BW_P2_SCR_generate_figures.py`
- `python scripts/BW_P3_FIG_probe1_triage_plot.py`

Logical figures regenerated: 19

- ALGO P1: 5
- GSM P1: 4
- GSM P3: 3
- ALGO P3: 1
- BW P2: 5
- BW P3: 1

## Consolidation totals

- Rows fixed in ALGO bank: 10 answer updates; 25 verifier renames; 75 difficulty normalizations.
- Rows fixed in GSM bank: 42 contamination-pole updates.
- Metrics recomputed (rows across final metric CSV outputs): 407.
- Figures regenerated (logical count): 19.

