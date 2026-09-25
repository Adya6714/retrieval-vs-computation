# C3 corrections changelog

Date: 2026-09-03  
Scope: remove/correct five known false or unrecovered figures everywhere they appear in derived tables, scripts, and coverage notes.

---

## 1. N3 Llama mechanistic–behavioral Spearman — REMOVED (degenerate)

**Problem.** `N3_Llama` was reported as ρ=0.147, CI [0.086, 0.243], clustered p=0.0003 with **1/60** W3-correct. A cluster bootstrap over a near-constant binary outcome is undefined: the CI is narrow because almost every resample reproduces the same degenerate configuration.

**Actions.**
- Cleared ρ / CI / p for Llama in `results/derived/N3_algo_mech_behavior_link.csv` (note: `outcome variable degenerate (1/60 positive); correlation not estimable`).
- Cleared the same estimate in `results/derived/ALL_INFERENCE_AUDIT.csv` (`N3_Llama-3.1-8B`); kept `previously_reported_p=0.0003` as withdrawal audit trail.
- Guard in `scripts/consolidate/algo_mech_behavior_link.py` and `scripts/consolidate/inference_audit.py`: require `min(n_pos, n_neg) ≥ 5` before estimating Spearman / cluster bootstrap.
- Added O13 / MEASUREMENT_FAILURES rows (exact O13 reason text below).

**O13 row (verbatim reason):**  
`ALGO mechanistic-behavioral link, Llama-3.1-8B: outcome variable degenerate (1/60 positive), correlation not estimable.`

Qwen2.5-1.5B remains undefined (0/61 positives) — unchanged.

---

## 2. BW Probe 2 “canonical execution 0.02–0.22” — REPLACED

**Problem.** Handoff prose “0.02–0.22 canonical” is not recoverable from raw CCI.

**Actual values** (`BW_P2_floor_documentation.csv` / `results/raw/BW_P2_cci.csv`):
| Model | goal_reached_rate | partial_goal_achievement_mean |
|-------|-------------------|-------------------------------|
| Claude | 0.0 | 0.0251 |
| GPT-4o | 0.0 | 0.0083 |
| Llama | 0.0 | 0.1355 |

**Actions.**
- Updated notes in `BW_P2_floor_documentation.csv` to forbid citing 0.02–0.22.
- Set `canonical_execution_accuracy=0.0` for BW Claude/GPT-4o/Llama rows in `COVERAGE_PROBE2.csv` and documented the dead-twice note (item 5).
- O13 BW P2 floor row already used the correct partial_goal range; left as-is.

No committed vault prose still cited 0.02–0.22 (search clean).

---

## 3. Qwen2.5-7B ALGO median gold-token rank — 80,369 (not 80,440)

**Canonical source:** `Mech Frequency Controlled Summary.csv` slice  
`Qwen/Qwen2.5-7B-Instruct | ALGO | all` → `median_rank_canonical = 80369.0`.

**Actions.**
- Confirmed `O13_measurement_failures.csv` already lists **80369** (no change needed).
- No derived CSV or vault note still reporting 80440 as the median; incidental raw-layer ranks containing the integer 80440 were left untouched (not the summary statistic).
- N3 task prose historically said 80,440 — treat that as superseded by the Summary CSV.

---

## 4. BW W6 goal towers 1.74 vs 1.0 — O1 run; marked UNRESOLVED

**Problem.** N1 cited W6 mean `n_goal_towers = 1.74` vs canonical 1.00. Current parser (`count_goal_towers` in `probes/contamination/bw_instance_metrics.py`) yields **1.0 for all K3 valid pairs** (n=47 one-tower subset ≡ all valid), so the “one-tower control” is vacuous. Full W6 bank mean under the same parser ≈ **0.85** (values in {0,1}) — neither matches 1.74.

**O1 hand-validation** (`O1_tower_parser_handcheck_sample.csv`): 10 W6 instances; manual connected-component count **agrees 10/10** with the parser (all = 1). Parser is internally consistent with its definition; the **1.74 figure remains unrecovered**, so the tower control cannot ground an L1 claim.

**Actions.**
- Completed / refreshed `results/derived/O1_tower_parser_handcheck_sample.csv`.
- Annotated `N1_bw_w6_stratified_accuracy.csv` `w6_one_goal_tower` rows: vacuous/UNRESOLVED; do not interpret as L1 surviving tower control.
- Comment in `scripts/consolidate/bw_canonical_w6_matched_compare.py` at the one-tower subset.
- O13 + MEASUREMENT_FAILURES status **UNRESOLVED**; instruction: do not claim “L1 survives the one-tower control.”

Naming-matched controls (n=13) are unaffected by this withdrawal.

---

## 5. COVERAGE_PROBE2 — BW Probe 2 dead twice over

**Action.** Every BW row in `COVERAGE_PROBE2.csv` now carries (in `reason_if_missing`):

> BW Probe 2 dead twice over: (1) floor effects on Claude/GPT-4o/Llama (goal_reached=0; partial_goal_mean GPT-4o=0.008 to Llama=0.136); (2) no data at all for Gemini, o4-mini, DeepSeek. Do not use 0.02–0.22 canonical-execution figure (unrecovered).

Models with data also have `canonical_execution_accuracy=0.0` and per-model `partial_goal_mean`.

---

## Files touched

| Path | Change |
|------|--------|
| `scripts/consolidate/algo_mech_behavior_link.py` | Degeneracy guard (`min_cell < 5`) |
| `scripts/consolidate/inference_audit.py` | Same guard; clears N3 Llama estimates |
| `scripts/consolidate/bw_canonical_w6_matched_compare.py` | Comment: vacuous one-tower control |
| `scripts/consolidate/o13_oracle_and_failures.py` | Persist C3 N3-Llama + tower-UNRESOLVED rows in TABLE B |
| `results/derived/N3_algo_mech_behavior_link.csv` | Llama ρ/CI/p cleared |
| `results/derived/ALL_INFERENCE_AUDIT.csv` | N3_Llama estimate cleared; withdrawal note |
| `results/derived/O13_measurement_failures.csv` | +N3 Llama row; +BW tower UNRESOLVED |
| `results/derived/MEASUREMENT_FAILURES.csv` | Matching two rows |
| `results/derived/O1_tower_parser_handcheck_sample.csv` | O1 10-instance hand-check |
| `results/derived/N1_bw_w6_stratified_accuracy.csv` | `note` on one-tower rows |
| `results/derived/COVERAGE_PROBE2.csv` | BW dead-twice + goal_reached=0 |
| `results/derived/BW_P2_floor_documentation.csv` | Explicit 0.02–0.22 ban; correct partial_goal range |
| `results/derived/C3_corrections_changelog.md` | This file |

**Vault:** no matches for 80440, 0.02–0.22, N3 ρ=0.147, or “L1 survives the one-tower control” in `research-vault/` at correction time — nothing to edit there.

---

## C1 hygiene — 2026-09-25

**Scope.** Audit section B (`docs/audit/REPO_AUDIT_2026-09-25.md`): drop mock / column-shifted rows from derived rescored CSVs; normalise ALGO `verifier_function`; fix Claude display name; report ALGO ID width (no rename).

### 1. Mock + column-shifted rows (derived only; raw append-only)

- Script: `scripts/consolidate/rescore_p1.py` now **omits** rows with `model` in `{mock, the answer is 42.}` / `variant_type=MOCK` / answer-like model text from derived `*_rescored.csv`.
- Detects the BW column-shift pattern (4 rows in `results/raw/BW_P1_behavioral.csv` where `model == "The answer is 42."` and `notes` holds `planning_suite`); repairs in-memory then drops as mock. Raw unchanged.
- Regenerated all P1 `*_rescored.csv`. Drops: BW combined 62 (58 mock + 4 shifted), ALGO claude 3, ALGO llama 2.
- **Included metrics unchanged** (0 diffs on model×variant n/acc). Non-rescored `results/derived/` and `results/paper/` MD5 unchanged.

### 2. `verifier_function` normalisation (`data/problems/question_bank_algo.csv`)

| Before | After (by subtype) |
|---|---|
| `verify_coinchange` | `verify_coinchange` |
| `verify_sp`, `Dijkstra`, `Bellman-Ford` | `verify_sp` |
| `verify_wis`, `veryify_WIS` | `verify_wis` |

- 50/690 bank rows changed. Generator typo `veryify_WIS` fixed in `scripts/generation/stage1_generate_algo.py`.
- `probes/contamination/algo_instance_metrics.py`: SP optimal cost uses Bellman-Ford when any edge weight is negative (so SP_004 still works after losing the `Bellman-Ford` verifier string).
- Guard: `tests/test_bank_schema.py`.
- Rebuilt `site/data/pipeline_explorer.json` (verifier field sync).

### 3. Display name

- `configs/models.yaml`: `claude-sonnet-4` → `Claude Sonnet 4`.
- Figure label maps in `ALGO_P1/P2/P3_FIG_generate.py`, `BW_P2_SCR_generate_figures.py` updated to match. (Figure binaries not regenerated this pass.)

### 4. ALGO ID width — report only (no rename)

See `docs/audit/C1_ALGO_ID_REPORT.md`. **Proposal (awaiting approval):** canonical form `{PREFIX}_{NNN}` zero-padded to 3 digits (`CC_01` → `CC_001`). Do not apply until approved.
---

## C2 — W3b relabel + Alice substring fix — 2026-09-25
**Scope.** HP-23 steps 1–2 only (no W3a generation, no model calls).
### 1. Relabel
Added columns `w3_kind=isomorph` and `variant_subtype=W3b` on every existing W3 row. `variant_type` unchanged (`W3`). Schema: `probes/common/io.py`.
| Family | W3 rows labelled W3b |
|---|---:|
| GSM | 44 |
| BW | 65 |
| ALGO | 110 |

### 2. Substring / article-collision fix
`apply_mapping` now whole-word replaces with an article-context guard for keys `a`/`an`/`the` (and single-letter alpha keys): skips `You are a …`, `at a …`, `with a …`.
Detected and regenerated **18** mechanical BW/MBW W3 rows. Of these, **12** had `You are Alice robot arm`.

Before/after heads:
- `BW_496`
  - before: `You are Alice robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), assign_to_base X (place X on t`
  - after: `You are a robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), assign_to_base X (place X on the t`
- `BW_498`
  - before: `You are Alice robot arm. Available actions: hire X (X must be clear and on the table, hand must be empty), assign_to_base X (place X on the `
  - after: `You are a robot arm. Available actions: hire X (X must be clear and on the table, hand must be empty), assign_to_base X (place X on the tabl`
- `BW_499`
  - before: `You are Alice robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), dismiss X (place X on the tabl`
  - after: `You are a robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), dismiss X (place X on the table), `
- `BW_500`
  - before: `You are Alice robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), dismiss X (place X on the tabl`
  - after: `You are a robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), dismiss X (place X on the table), `
- `BW_501`
  - before: `You are Alice robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), dismiss X (place X on the tabl`
  - after: `You are a robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), dismiss X (place X on the table), `
- `BW_503`
  - before: `You are Alice robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), assign_to_base X (place X on t`
  - after: `You are a robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), assign_to_base X (place X on the t`
- `BW_504`
  - before: `You are Alpha robot arm. Available actions: mobilize X (X must be clear and on the table, hand must be empty), deploy X (place X on the tabl`
  - after: `You are a robot arm. Available actions: mobilize X (X must be clear and on the table, hand must be empty), deploy X (place X on the table), `
- `BW_510`
  - before: `You are Alpha robot arm. Available actions: mobilize X (X must be clear and on the table, hand must be empty), deploy X (place X on the tabl`
  - after: `You are a robot arm. Available actions: mobilize X (X must be clear and on the table, hand must be empty), deploy X (place X on the table), `
- `BW_513`
  - before: `You are Alice robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), assign_to_base X (place X on t`
  - after: `You are a robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), assign_to_base X (place X on the t`
- `BW_514`
  - before: `You are Alice robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), assign-to-base X (place X on t`
  - after: `You are a robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), assign-to-base X (place X on the t`
- `BW_515`
  - before: `You are Alice robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), dismiss X (place X on the tabl`
  - after: `You are a robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), dismiss X (place X on the table), `
- `BW_E_001`
  - before: `You are Alice robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), dismiss X (place X on the tabl`
  - after: `You are a robot arm. Available actions: recruit X (X must be clear and on the table, hand must be empty), dismiss X (place X on the table), `
- `BW_E_004`
  - before: `You are Alice robot arm. Available actions: hire X (X must be clear and on the table, hand must be empty), assign_to_base X (place X on the `
  - after: `You are a robot arm. Available actions: hire X (X must be clear and on the table, hand must be empty), assign_to_base X (place X on the tabl`
- `MBW_496`
  - before: `You are Alpha robot arm. Available actions: deploy X (requires harmony, province X, planet X to be true), surrender X (requires pain X to be`
  - after: `You are a robot arm. Available actions: deploy X (requires harmony, province X, planet X to be true), surrender X (requires pain X to be tru`
- `MBW_497`
  - before: `You are Alice robot arm. Available actions: terminate X (requires harmony, province X, planet X to be true), resign X (requires pain X to be`
  - after: `You are a robot arm. Available actions: terminate X (requires harmony, province X, planet X to be true), resign X (requires pain X to be tru`
- `MBW_498`
  - before: `You are Alpha robot arm. Available actions: deploy X (requires harmony, province X, planet X to be true), surrender X (requires pain X to be`
  - after: `You are a robot arm. Available actions: deploy X (requires harmony, province X, planet X to be true), surrender X (requires pain X to be tru`
- `MBW_499`
  - before: `You are Alpha robot arm. Available actions: secure X (requires harmony, province X, planet X to be true), surrender X (requires pain X to be`
  - after: `You are a robot arm. Available actions: secure X (requires harmony, province X, planet X to be true), surrender X (requires pain X to be tru`
- `MBW_500`
  - before: `You are Alpha_Base robot arm. Available actions: deploy_forces X (requires harmony, province X, planet X to be true), surrender_position X (`
  - after: `You are a robot arm. Available actions: deploy_forces X (requires harmony, province X, planet X to be true), surrender_position X (requires `

### 3. Reruns
Listed in `docs/trackT/T0_PENDING_RERUNS.md` (OpenRouter key present; plain `--resume` cannot force re-score under append-only raw).
