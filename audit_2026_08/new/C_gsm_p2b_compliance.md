# C. GSM Phase-2B compliance

**Not computable.** The four-way taxonomy (compliant / partial / refusal / format-ignored) is the ALGO Phase-2B `response_type` classifier in `scripts/ALGO_P2_SCR_run_phase2.py:parse_decision_reason` (Decision:/Reason: format). It does not exist for GSM.

GSM Phase-2B raw injection-step text is not in the released logs, so the ALGO classifier cannot be reapplied.

**GSM 2B fields that ARE available** (`results/raw/GSM_P2_cci.csv`):
`problem_id, model, cci_score, cci_matched, cci_total, valid_divergence, tep_score, inject_at_step, injected_value, session_b_correct, correct_answer, contamination_pole, difficulty, problem_subtype`

**Additional Phase-1/session fields** (`GSM_P2_phase1_*.csv`, including o4-mini):
`problem_id, model, contamination_pole, difficulty, phase1_steps_json, phase1_final_answer, phase1_parseable, cci_score, cci_matched, cci_total, inject_at_step, injected_value, true_value_at_injection, tep_score, tep_diverged_steps, tep_total_steps, session_b_correct, correct_answer`

| model | n in GSM_P2_cci.csv | n in phase1 file | phase1 file |
|---|---:|---:|---|
| Claude | 44 | 44 | yes |
| GPT-4o | 44 | 44 | yes |
| Llama | 44 | 44 | yes |
| Gemini | 44 | 44 | yes |
| o4-mini | 0 | 44 | yes |

o4-mini is in `GSM_P2_phase1_o1mini.csv` (n=44) but **not** in `GSM_P2_cci.csv` (the 4-model table the paper plots).

**Flags:** no `response_type`, `raw_response` (2B), or `parse_status` on any GSM P2 file. Closest 2B outcomes already stored: `tep_score`, `tep_diverged_steps`, `tep_total_steps`, `session_b_correct`, `inject_at_step`, `injected_value`.
