# Rebuild report

Every number below is from `rebuild/NUMBERS.csv`, computed from `results/raw/` under the frozen filters in `rebuild/FROZEN_FILTERS.md`. `results/` and `paper/` were not modified.

## Frozen definitions (applied everywhere)

- GSM bank-valid canonical IDs: n=44 from `data/problems/question_bank_gsm.csv`.
- ALGO adversarial pool: 34 SP + 10 CC + 17 WIS = 61 (not bank `instance_type`).
- BW bank: n=65 PlanBench IDs (50 standard + 15 mystery).
- `model == 'mock'` dropped explicitly before `keep='last'`.
- VRI = mean(W1, W2, W4) − W3, per problem, 0/1 correctness.
- BW W3 = entity+action rename; BW W5 = init/goal swap (confirmed from generator).

## Probe 1 — headline

P1.1 replaces Table 7. Per-(model, family, subtype, variant) accuracy + Wilson 95% CI is in NUMBERS.csv ids `P1.1.*`.

| model | GSM Acc_can | GSM Acc_W3 | GSM R_W3 | ALGO SP-chall Acc_can | ALGO SP-chall Acc_W3 | BW Acc_W3 (rename) | BW Acc_W5 (init/goal swap) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude | 0.841 | 0.750 | 0.892 | 0.647 | 0.000 | 0.138 | 0.523 |
| GPT-4o | 0.850 | 0.300 | 0.353 | 0.412 | 0.265 | 0.169 | 0.246 |
| Llama | 0.800 | 0.150 | 0.187 | 0.059 | 0.000 | 0.108 | 0.000 |
| Gemini | 0.909 | 0.523 | 0.575 | 0.676 | 0.324 | 0.108 | 0.569 |
| o4-mini | 0.841 | 0.841 | 1.000 | 1.000 | 0.882 | 0.185 | 0.769 |

P1.5 all-pairs inversion: `rebuild/p1_pairwise_inversion.csv` (3 subtypes × 10 pairs × 2 definitions, Fisher + bootstrap 10k).
P1.6 within-model φ: ids `P1.6.*`.

## Probe 2 — headline

| model | GSM Acc_P2A | GSM CCI mean | GSM CCI med | GSM TEP | ALGO Acc_P2A | ALGO CCI mean | ALGO TEP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Claude | 0.864 | 0.231 | 0.216 | 0.539 | 0.500 | 0.410 | 0.185 |
| GPT-4o | 0.705 | 0.108 | 0.000 | 0.598 | 0.500 | 0.339 | 0.221 |
| Llama | 0.455 | 0.167 | 0.000 | 0.773 | 0.218 | 0.339 | 0.601 |
| Gemini | 0.886 | 0.270 | 0.250 | 0.652 | 0.309 | 0.257 | 0.206 |
| o4-mini | 0.955 | 0.220 | 0.143 | 0.628 | 0.436 | NOT_COMPUTABLE | 0.344 |

BW Probe 2 covers **3 models** (strict-PDDL) and **3 models** (NL-tolerant), not 5. See P2.5 rows.

P2.1 declaration parse / empty / diverged (GSM):

| model | parse_rate | n_empty | n_diverged | empty Acc | diverged Acc |
|---|---:|---:|---:|---:|---:|
| Claude | 0.909 | 2 | 12 | 0.000 | 0.917 |
| GPT-4o | 0.455 | 17 | 12 | 0.706 | 0.667 |
| Llama | 0.455 | 13 | 11 | 0.615 | 0.364 |
| Gemini | 0.818 | 3 | 7 | 0.333 | 0.857 |
| o4-mini | 0.977 | 1 | 16 | 0.000 | 0.938 |

## Probe 3 infini-gram — headline

Paper §4.3 labels an instance-level correlation as template-level. Both are reported, labelled.

| model | instance r vs VRI | template r vs VRI | instance partial r | n |
|---|---:|---:|---:|---:|
| Claude | 0.444 | -0.366 | 0.418 | 61 |
| GPT-4o | 0.364 | 0.278 | 0.380 | 61 |
| Llama | 0.114 | 0.268 | 0.124 | 61 |
| Gemini | 0.121 | -0.047 | 0.119 | 61 |
| o4-mini | -0.171 | -0.045 | -0.171 | 61 |

P3.4 within-ALGO gradient: ids `P3.4.*` (mean template proximity, canonical acc, W3 acc, per subtype per model).

## Probe 3 mechanistic — inventory only

See `rebuild/mechanistic_inventory.csv`. No new mechanistic claims were computed (P3.7).

## Triangulation

Executed rule is in `rebuild/triangulation_rule.py` (named constants). It is the 5-field AND from `ALGO_P3_SCR_triangulation.py`, **not** the appendix three-signal print.

- 4-model (paper scope, n=440): retrieval=8 computation=4 mixed=157 ambiguous=271.
- Flags on that panel: parse_failure=127 missing_phase2=196 missing_core=0.
- Reproduces 8/4/157/271? **yes**
  YES — reproduces 8/4/157/271 from raw.
- 5-model under the same rule: retrieval=8 computation=4 mixed=157 ambiguous=381.
- Appendix three-signal on 5 models: retrieval=8 computation=0 mixed=356 ambiguous=186. That printed rule is **not defensible as the paper default** because it was not the function that produced the published 8/4/157/271.
- 270-config sweep (same missing-data flags as the from-raw panel): 270 cells; 21 match 8/4/157/271. CSV: `rebuild/triangulation_270_sweep.csv`.

## New analyses

N.1 intrusion rates: ids `N.1.*`; examples in `rebuild/intrusion_examples.csv`.

| model | GSM intrusion rate | GSM n_err | ALGO intrusion rate | ALGO n_err |
|---|---:|---:|---:|---:|
| Claude | 0.000 | 11 | 0.020 | 100 |
| GPT-4o | 0.000 | 14 | 0.022 | 91 |
| Llama | 0.000 | 17 | 0.000 | 108 |
| Gemini | 0.000 | 21 | 0.049 | 82 |
| o4-mini | 0.000 | 7 | 0.116 | 43 |

N.2 ALGO TEP by model: ids `N.2.ALGO.*` (same values as P2.3 TEP; previously unreported per-model).

## Where rebuilt numbers differ from the paper

- Table 7 GSM -- GPT-4o W6: paper=0.800 rebuilt=NOT_COMPUTABLE (no valid rows after bank+ERROR filter)
- Table 7 GSM -- Llama W6: paper=0.450 rebuilt=NOT_COMPUTABLE (no valid rows after bank+ERROR filter)
- Table 7 ALGO SP-chall. GPT-4o W6: paper=0.258 rebuilt=0.265 (n=34; ALGO_P1_behavioral_gpt4o.csv)
- Table 7 ALGO SP-chall. Llama W6: paper=0.065 rebuilt=0.059 (n=34; ALGO_P1_behavioral_llama.csv)
- Table 7 ALGO SP-std. GPT-4o W6: paper=0.368 rebuilt=0.381 (n=21; ALGO_P1_behavioral_gpt4o.csv)
- Table 7 ALGO SP-std. Llama W6: paper=0.105 rebuilt=0.095 (n=21; ALGO_P1_behavioral_llama.csv)
- §4.3 proximity pool n: paper=64 rebuilt=61 (frozen adversarial list is 61, not 64)
- §4.3 o4-mini r: paper=-0.094 rebuilt=-0.17129223374405897 (n=61)
- §4.2 GPT-4o empty-declaration Acc: paper=0.69 rebuilt=0.7058823529411765 (n=17)
- §4.2 GPT-4o diverged Acc: paper=0.73 rebuilt=0.6666666666666666 (n=12)
- Table 6 NL-tolerant covers all five: paper=5 rebuilt=3.0 (n=3)

## NOT_COMPUTABLE

42 rows. Reasons:

- `P1.1.GSM.GPT-4o.W6` (accuracy GPT-4o): no valid rows after bank+ERROR filter
- `P1.1.GSM.Llama.W6` (accuracy Llama): no valid rows after bank+ERROR filter
- `P1.1.ALGO.CC-chall.Claude.W5` (accuracy Claude): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.CC-chall.Claude.W6` (accuracy Claude): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.CC-std.Claude.W5` (accuracy Claude): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.WIS-chall.Claude.W5` (accuracy Claude): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.WIS-std.Claude.W5` (accuracy Claude): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.CC-chall.GPT-4o.W5` (accuracy GPT-4o): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.CC-std.GPT-4o.W5` (accuracy GPT-4o): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.WIS-chall.GPT-4o.W5` (accuracy GPT-4o): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.WIS-std.GPT-4o.W5` (accuracy GPT-4o): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.CC-chall.Llama.W5` (accuracy Llama): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.CC-std.Llama.W5` (accuracy Llama): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.WIS-chall.Llama.W5` (accuracy Llama): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.WIS-std.Llama.W5` (accuracy Llama): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.CC-chall.Gemini.W5` (accuracy Gemini): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.CC-chall.Gemini.W6` (accuracy Gemini): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.CC-std.Gemini.W5` (accuracy Gemini): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.WIS-chall.Gemini.W5` (accuracy Gemini): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.WIS-std.Gemini.W5` (accuracy Gemini): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.CC-chall.o4-mini.W5` (accuracy o4-mini): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.CC-chall.o4-mini.W6` (accuracy o4-mini): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.CC-std.o4-mini.W5` (accuracy o4-mini): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.WIS-chall.o4-mini.W5` (accuracy o4-mini): variant not present for this slice (W5/W6 holes are real)
- `P1.1.ALGO.WIS-std.o4-mini.W5` (accuracy o4-mini): variant not present for this slice (W5/W6 holes are real)
- `P1.5.SP.Llama_vs_Gemini.canonically-matched.fisher_p` (fisher_p Llama|Gemini): empty ID intersection
- `P1.5.WIS.Claude_vs_Llama.canonically-matched.fisher_p` (fisher_p Claude|Llama): empty ID intersection
- `P1.6.ALGO.o4-mini.phi` (phi o4-mini): degenerate 2x2 (zero variance)
- `P2.1.ALGO.o4-mini.parse_rate` (declaration_parse_rate o4-mini): No ALGO Phase-1 declaration file for this model
- `P2.1.ALGO.o4-mini.n_empty_declarations` (n_empty_declarations o4-mini): no phase1
- `P2.3.GSM.four_way_compliance` (four_way_compliance ): ALGO four-way taxonomy is Decision:/Reason: parse of injection-step raw_response. GSM Phase-2B logs have no injection-step raw_response and no response_type.
- `P2.2.ALGO.o4-mini.cci_mean` (CCI_mean o4-mini): o4-mini has no Phase-1 file so CCI cannot be built
- `P2.4.ALGO.GPT-4o.fisher` (correct_given_complied_vs_refused GPT-4o): n_compliant=57 n_refusal=0 — Fisher undefined
- `P2.4.ALGO.Llama.fisher` (correct_given_complied_vs_refused Llama): n_compliant=24 n_refusal=0 — Fisher undefined
- `P2.4.ALGO.Gemini.fisher` (correct_given_complied_vs_refused Gemini): n_compliant=0 n_refusal=0 — Fisher undefined
- `P2.4.ALGO.o4-mini.fisher` (correct_given_complied_vs_refused o4-mini): n_compliant=61 n_refusal=0 — Fisher undefined
- `P2.5.BW.strict_pddl.Gemini.abort_rate` (abort_rate Gemini): model not in this BW P2 file (coverage is 3 models, not 5)
- `P2.5.BW.strict_pddl.o4-mini.abort_rate` (abort_rate o4-mini): model not in this BW P2 file (coverage is 3 models, not 5)
- `P2.5.BW.nl_tolerant.Gemini.abort_rate` (abort_rate Gemini): model not in this BW P2 file (coverage is 3 models, not 5)
- `P2.5.BW.nl_tolerant.o4-mini.abort_rate` (abort_rate o4-mini): model not in this BW P2 file (coverage is 3 models, not 5)
- `P3.3.o4-mini.template_contamination_score_vs_CCI` (pearson_template_contamination_score_vs_CCI o4-mini): n<4 or zero variance
- `P3.3.o4-mini.instance_contamination_score_vs_CCI` (pearson_instance_contamination_score_vs_CCI o4-mini): n<4 or zero variance

## Files written

- `rebuild/NUMBERS.csv` — frozen number file for the paper
- `rebuild/FROZEN_FILTERS.md` — exact ID lists
- `rebuild/triangulation_rule.py` — executed label rule with named constants
- `rebuild/triangulation_270_sweep.csv`
- `rebuild/triangulation_panel.csv`, `triangulation_4model_labels.csv`, `triangulation_5model_labels.csv`
- `rebuild/p1_vri_per_problem.csv`, `p1_pairwise_inversion.csv`
- `rebuild/algo_cci_per_instance.csv`, `algo_tep_sessions.csv`
- `rebuild/mechanistic_inventory.csv`, `intrusion_examples.csv`

