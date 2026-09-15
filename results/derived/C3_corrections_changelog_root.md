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
