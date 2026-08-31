# B. ALGO TEP per model

Appendix D: TEP = fraction of **post-injection** steps whose numeric/symbolic content differs from the uninjected run.

`ALGO_P2_phase2_injected*.csv` columns: `problem_id, model, subtype, instance_type, step_index, injection_applied, critical_step_index, true_state, injected_state, raw_response, response_type, parsed_decision, parse_status, reasoning_text, reasoning_type, diverged_from_normal, post_injection_correct`.

**Is ALGO TEP computable?** Yes, but **not from the injected file alone**.
- Injected files have **no `tep_score` column** (that is the GSM P2 column name).
- `diverged_from_normal` is **not** TEP: it is `step >= critical_step_index` (a Boolean “after injection” flag written at run time, not a content comparison).
- TEP **is** computable by joining `parsed_decision` on `(problem_id, model, step)` against `ALGO_P2_phase2_normal.csv` + `ALGO_P2_phase2_normal_gemini.csv`. That join is what this file reports.

| model | n sessions | mean TEP | median TEP |
|---|---:|---:|---:|
| Claude | 61 | 0.185 | 0.000 |
| GPT-4o | 61 | 0.221 | 0.000 |
| Llama | 61 | 0.601 | 0.800 |
| Gemini | 61 | 0.206 | 0.000 |
| o4-mini | 61 | 0.344 | 0.000 |

Session-level rows: 305 (plausible injection only; implausible file excluded as a different condition). Gemini taken from the dedicated rerun, not the mixed main file.

**Flags:** `305` sessions with at least one paired post-injection step. Sessions with `critical_step_index < 0` or no overlapping post-injection steps are dropped (TEP undefined), not imputed.
