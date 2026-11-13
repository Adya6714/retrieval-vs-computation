# v1 audit and what v1.1 changes

Audit date: 13 Nov 2026. Scope: the behavioural pipeline (banks, variant text, prompts, clients,
sweeps, verifiers, P2). Every count below was measured on the committed files and can be
reproduced with `python -m rvc11 build-bank` (see `data/v11/bank_summary.json`, `quarantine.csv`,
`stripped_instructions.csv`).

No real model was called. Model ids in `configs/v11/models.yaml` are placeholders until
`make v11-models` confirms them.

## 1. Defects that change v1 numbers

| # | Defect | Evidence | Effect |
|---|---|---|---|
| S1 | The Blocksworld sweep drops the `notes` column, so the W3 action-name mapping never reaches the verifier (`scripts/BW_P1_SCR_run_behavioral_sweep.py`, `row.get("notes")` on a dict that has no `notes`) | 25 of 50 Blocksworld W3 gold answers fail the verifier when called as the sweep calls it | W3 accuracy is biased down at sweep time. `scripts/consolidate/rescore_p1.py` rescores later, so there are two scoring paths |
| S2 | 40 Mystery Blocksworld gold answers do not simulate | `data/problems/mystery_invalid_gold.csv` (already known), reproduced | Those rows are excluded downstream, which shrinks Mystery to 65 of 105 rows |
| S3 | 25 WIS W6 rows show the canonical problem text with a different instance's parameters and gold | Text identical to canonical; `difficulty_params` and `correct_answer` differ | A correct answer is scored wrong |
| S4 | 8 more W6 rows and 5 W5 rows are copies of the canonical text | 3 Blocksworld + 5 Mystery W6, 5 Mystery W5 | W6/W5 partly measure nothing |
| S5 | Verification is stricter on W3 than on canonical for coin change | Canonical path accepts a count with no coin list; the W3 "scoops" path requires the exact gold list (`verify_coinchange` vs `verify_coinchange_scoops`) | Inflates the W3 drop |
| S6 | Lenient fallbacks accept partial answers | Shortest path accepts a correct cost with no path; WIS accepts a correct total with no selection; `_verify_numeric` takes the first number in the response, the GSM verifier takes the last | False positives that depend on answer style, which depends on the instruction |
| S7 | Planning init and goal are recovered by regex from the prompt prose, including LLM-written rewrites | `probes/contamination/verify.py::_parse_blocksworld_state` | A partial parse gives a smaller goal, which is easier to satisfy. Unparseable action lines are skipped, not failed |
| S8 | API failures are stored as a response string `ERROR: ...` and scored wrong | `OpenRouterClient.complete` returns the error as `response`; the sweep verifies it | Errors count as model failures unless a later resume retries them |
| S9 | No finish reason is logged, and `max_tokens` is 4096 to 8192 (1024 in the direct Anthropic client) | `probes/behavioral/*_client.py` | Truncated answers are invisible and scored wrong |
| S10 | Three clients with different settings serve different probes | `ModelClient` (no token cap, no retry), `OpenRouterClient`, `AnthropicClient` (direct API, 1024 tokens) | The same model is not the same system across P1 and P2 |
| S11 | One sample per item at temperature 0 | all sweeps | No noise estimate; per-item claims are not possible |

## 2. Defects in the item bank

| # | Defect | Evidence |
|---|---|---|
| B1 | Answer-format and verbosity instructions sit inside `problem_text` and differ by variant | GSM: "just the final answer" in 19/44 W3 and 19/44 W4, 0/44 canonical. Blocksworld: "no explanation" in 65/65 canonical, 51/65 W1, 35/65 W2. 55 distinct instruction tails in the planning bank alone |
| B2 | GSM W2 and W4 add solution content | 33 of 88 rows contain numbers that are not in the canonical problem (derived quantities such as "the remaining 66 minutes"); 5 state a solution step in words ("rounded up") |
| B3 | GSM W5 is a different question (the inverse problem), not a surface variant | 44/44 |
| B4 | Planning W5 is mostly "put every block on the table" | 47 of the 55 usable W5 planning rows |
| B5 | Planning W3 mixes two treatments | 35 true cover stories, 30 rename-only rows that keep "robot arm" and "block" |
| B6 | Blocksworld problems are near-trivial | 62 of 65 canonical problems start with every block on the table; all 65 goals are a single tower |
| B7 | 6 coin-change W3 rows never state the target amount | e.g. CC_031 W3: "the required total postage needed for a package" |
| B8 | 10 Mystery W5/W6 rows are unsolvable or under-specified as written | goal or initial facts missing |
| B9 | Metadata errors | `problem_family` is "coin change" on 10 rows and "algorithmic" elsewhere; shortest-path notes say "bidirectional" on directed graphs; WIS W3 gold names "Item A" while the prompt names "House 0"; 5 WIS problems store non-overlapping intervals although the task is a conflict graph |
| B10 | The default rewrite model is also an evaluated model | `configs/models.yaml: default_openrouter_model: anthropic/claude-sonnet-4` |
| B11 | Licence | v2 docs record GSM-Symbolic as CC-BY-NC-ND. Not checked here. If correct, derived variants cannot be redistributed |

## 3. What v1.1 does about each

| Problem | Change |
|---|---|
| B1 | `rvc11/bank.py` strips every instruction from the task text (removed text is logged per item). `rvc11/prompts.py` appends one programmatic answer-format block and one instruction level. A QA gate proves the non-task part of the prompt is identical across variants |
| S1, S5, S6, S7 | One verifier per answer type in `rvc11/`, used for every variant. It reads a formal spec stored in the bank (facts, graph, weights, denominations), requires a complete answer, and accepts any optimal answer |
| S2, S3, S4, B7, B8 | 54 rows quarantined with a reason (`data/v11/quarantine.csv`). 14 Mystery gold plans replaced by a plan from an independent reference planner |
| S7 | Canonical planning specs are cross-checked against PlanBench PDDL where the file resolves (54 of 65 agree; 0 disagree) |
| S8, S9, S10 | One client. Errors are records with `status=error`, retried on resume, never scored. Finish reason, token counts, provider and request settings are logged. Truncated wrong answers are excluded and counted |
| S11 | k samples per item (5 by default), cluster bootstrap over problems, paired tests, Holm correction, pass^k, a dedicated noise-floor experiment |
| B2 to B6 | Not repairable without rewriting items. Each row carries a flag and a tier. 1,179 rows are `verified` (text machine-checked against the spec), 200 are `unverified`. The report prints every P1 table twice: all usable rows, and verified only |

## 4. Redundant or superseded for the rerun (kept as archive, not deleted)

- `probes/behavioral/{model_client,openai_client,anthropic_client}.py`: replaced by `rvc11/client.py`.
- `scripts/{BW,GSM,ALGO}_P1_SCR_*`, `scripts/run_full_sweep.py`: three family-specific sweeps, replaced by `rvc11 run p1_surface`.
- `scripts/consolidate/` (60 scripts), `rebuild/`, `audit_2026_08/`, `results/paper/AUDIT/`: successive repair and rescoring layers over the same raw logs. Four places hold "final" numbers. v1.1 has one: `results/v11/<run>/scored.csv`.
- `probes/contamination/verify.py`, `verify_algo.py`: used only at bank-build time as prose parsers, never for scoring.
- `scripts/llama_greedy_*`, `scripts/algo_llama_greedy_*`, `scripts/ALGO_P1_SCR_rerun_sp_w5.py`: one-off reruns.
- `scripts/test_target_token_resolver.py`: a test outside `tests/`.
- `tests/`: 163 tests pass, and none exercises the sweep scoring path where S1 lives.

This list is by directory. It is not a file-by-file review of all 1,111 files.

## 5. Not done

Living checklist of ordered tests, next steps, and blockers: [README.md](README.md).

- No real model run. Costs and frontier ceiling effects are unknown until `make v11-pilot`.
- The 200 unverified rewrites were not regenerated. Doing that needs a rewrite model that is not evaluated, plus the same gates.
- GSM W6 gold (regenerated numbers) is taken from the legacy bank, not re-derived.
- P2 was rebuilt for arithmetic only. The Blocksworld and ALGO P2 scripts (`BW_P2_*`, `ALGO_P2_*`, `MBW_P2_*`) were read, not ported.
- P3 (infini-gram contamination), the mechanistic scripts, the Colab notebooks, the paper and the site were not touched.
- The `FINAL ANSWER:` marker is a new format. It is constant across variants, so it cannot create a variant effect, but v1.1 absolute accuracies are not comparable with v1.
