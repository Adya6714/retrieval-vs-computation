# Audit summary — 2026-08-29

Scope: `paper/main.tex`, `paper/appendix.tex`, Tables 3, 4, 5, 7, 9, §§4.1–4.5, Conclusion bullets.
Recompute source: `results/raw/` plus frozen labels / derived triangulation where the paper caption cites them.
`rederive_all_metrics.py` stdout: `rederive_stdout.txt` (104 lines, not truncated).
No files outside `audit_2026_08/` were left modified (rederive outputs were snapshotted and restored).

## MISMATCH

### Table 7 BW Claude W5
- section: Table 7
- paper: `.523`  recomputed: `0.566667`
- raw: `BW_P1_behavioral.csv / BW_P1_behavioral_claude.csv`
- filter: filter_p1_to_bank(BW) n=65 PlanBench IDs; drop mock; exclude variant_not_transformed
- note: 34/60

### Table 7 BW GPT-4o W5
- section: Table 7
- paper: `.246`  recomputed: `0.266667`
- raw: `BW_P1_behavioral.csv / BW_P1_behavioral_gpt4o.csv`
- filter: filter_p1_to_bank(BW) n=65 PlanBench IDs; drop mock; exclude variant_not_transformed
- note: 16/60

### Table 7 BW Gemini W5
- section: Table 7
- paper: `.569`  recomputed: `0.616667`
- raw: `BW_P1_behavioral.csv / BW_P1_behavioral_gemini.csv`
- filter: filter_p1_to_bank(BW) n=65 PlanBench IDs; drop mock; exclude variant_not_transformed
- note: 37/60

### Table 7 BW o4-mini W5
- section: Table 7
- paper: `.769`  recomputed: `0.833333`
- raw: `BW_P1_behavioral.csv / BW_P1_behavioral_o1mini.csv`
- filter: filter_p1_to_bank(BW) n=65 PlanBench IDs; drop mock; exclude variant_not_transformed
- note: 50/60

### §4.2 GPT-4o empty-declaration Acc
- section: §4.2
- paper: `0.69`  recomputed: `0.705882`
- raw: `GSM_P2_cci.csv`
- filter: 17/29
- note: 

### §4.2 GPT-4o diverged Acc
- section: §4.2
- paper: `0.73`  recomputed: `0.666667`
- raw: `GSM_P2_cci.csv`
- filter: 12/29
- note: 

### §4.2 Claude post-inj comparison figure (paper second number)
- section: §4.2
- paper: `54.1%`  recomputed: `0.52459`
- raw: `implausible file OR pooled 54.1%`
- filter: implausible last-row acc=0.525; paper Claude 54.1% is pooled implausible aggregate n=122
- note: 

### §4.2 o4-mini post-inj comparison figure (paper second number)
- section: §4.2
- paper: `40.9%`  recomputed: `0.42623`
- raw: `ALGO_P2_phase2_injected_implausible.csv`
- filter: implausible last-row acc=0.426; paper Claude 54.1% is pooled implausible aggregate n=122
- note: 

### §4.2 Gemini plausible vs implausible Δpp
- section: §4.2
- paper: `-3.3`  recomputed: `-1.63934`
- raw: `injected + implausible CSVs`
- filter: post_injection_correct last row
- note: 

### §4.3 GPT-4o proximity-VRI r (AUDIT file)
- section: §4.3
- paper: `+0.37`  recomputed: `0.36`
- raw: `results/paper/AUDIT/contamination_vri_algo_adversarial.csv (from P3+P1)`
- filter: n=61 (paper says n=64; this file n=61 = 34+10+17)
- note: 

### §4.3 Llama proximity-VRI r (AUDIT file)
- section: §4.3
- paper: `0.12`  recomputed: `0.11`
- raw: `results/paper/AUDIT/contamination_vri_algo_adversarial.csv (from P3+P1)`
- filter: n=61 (paper says n=64; this file n=61 = 34+10+17)
- note: 

### §4.3 proximity pool n
- section: §4.3
- paper: `64`  recomputed: `61`
- raw: `frozen adversarial union 34 SP + 10 CC + 17 WIS = 61`
- filter: paper 64 ≠ frozen 61; bank adversarial is 71
- note: 

### §4.4 / Table 6 NL-tolerant covers all five models
- section: §4.4 / Table 6
- paper: `all five models`  recomputed: `anthropic/claude-sonnet-4,meta-llama/llama-3.1-8b-instruct,openai/gpt-4o`
- raw: `BW_P2_cci_nl.csv`
- filter: models present in NL rerun
- note: NL files contain 3 models: ['anthropic/claude-sonnet-4', 'meta-llama/llama-3.1-8b-instruct', 'openai/gpt-4o']

### §4.4 rename attributed to NL-tolerant Probe-2 protocol
- section: §4.4
- paper: `NL-tolerant protocol`  recomputed: `Probe-1 BW behavioral (OpenRouter), n=109 Claude/GPT/Llama and n=65 Gemini/o4-mini`
- raw: `BW_P1_behavioral.csv + gemini/o1mini files; NOT BW_P2_cci_nl.csv`
- filter: unfiltered P1 paired can∩W5
- note: Numbers match P1 W5; NL-tolerant P2 files have only 3 models and do not produce these Acc_can/Acc_W5 pairs

### Appendix o4-mini excluded because W3=1.00
- section: Appendix triangulation
- paper: `1.00`  recomputed: `0.609091`
- raw: `ALGO_P1_behavioral_o1mini.csv`
- filter: overall W3 n=110
- note: Exclusion applied (omini not in 440), but stated reason W3=1.00 is false; Acc_W3=0.609, Acc_can=1.00

## NOT_RECOMPUTABLE

### Table 4 Claude Acc_P2A
- section: Table 4
- paper: `.864`  recomputed: ``
- note: either=0.864

### Table 4 GPT-4o Acc_P2A
- section: Table 4
- paper: `.705`  recomputed: ``
- note: either=0.705

### Table 4 Llama Acc_P2A
- section: Table 4
- paper: `.455`  recomputed: ``
- note: either=0.455

### Table 4 Gemini Acc_P2A
- section: Table 4
- paper: `.886`  recomputed: ``
- note: either=0.886

### Table 4 o4-mini Acc_P2A
- section: Table 4
- paper: `.955`  recomputed: ``
- note: either=0.955

### Table 5 CC mean template proximity
- section: Table 5
- paper: `0.468`  recomputed: `0.468`
- note: Infini-gram not re-queried; figure script hardcodes [0.468, 0.147, 0.000]

### Table 5 SP mean template proximity
- section: Table 5
- paper: `0.147`  recomputed: `0.147`
- note: Infini-gram not re-queried; figure script hardcodes [0.468, 0.147, 0.000]

### Table 5 WIS mean template proximity
- section: Table 5
- paper: `0.000`  recomputed: `0.000`
- note: Infini-gram not re-queried; figure script hardcodes [0.468, 0.147, 0.000]

### Table 9 liberal v2 retrieval %
- section: Table 9
- paper: `27.3%`  recomputed: `0.273 (from derived triangulation_v2_summary.md, not raw)`
- note: Value taken from triangulation_v2_summary.md; raw vote matrix not re-aggregated here

### Table 9 liberal v2 computation %
- section: Table 9
- paper: `30.4%`  recomputed: `0.304 (derived summary)`
- note: Confirmed in derived summary; not re-derived from results/raw/

### Table 9 liberal v2 strong total %
- section: Table 9
- paper: `57.7%`  recomputed: `0.577 (derived summary)`
- note: 

### Table 9 liberal v2 ambiguous %
- section: Table 9
- paper: `37.9%`  recomputed: `0.379 (derived summary)`
- note: 

### §4.1 universally W3-fragile 26/110 (12 SP, 11 WIS, 3 CC)
- section: §4.1
- paper: `26/110`  recomputed: `not uniquely recoverable`
- note: All-4-models W3=False yields 68 (26 SP, 28 WIS, 14 CC). Frozen-adv subset yields 38 (20 SP, 16 WIS, 2 CC). scripts/runs/cross_probe_patterns.py requires >=4 models canon-correct AND W3-collapse (4 problems across families, not 26 ALGO). No released definition reproduces 12/11/3.

### §4.2 GPT-4o Acc_P2A 70.5%
- section: §4.2 / Conclusion
- paper: `70.5%`  recomputed: ``
- note: 

### §4.3 Claude partial r +0.41 p=0.0007
- section: §4.3
- paper: `+0.41`  recomputed: ``
- note: AUDIT file has n=61 Pearson only

### §4.3 GPT-4o partial r +0.39 p=0.002
- section: §4.3
- paper: `+0.39`  recomputed: ``
- note: 

### §4.3 Claude CCI proximity r +0.31 p=0.044 n=42
- section: §4.3
- paper: `+0.31`  recomputed: ``
- note: join not uniquely specified in raw

### §4.3 o4-mini proximity r -0.094 p=0.46 n=64
- section: §4.3
- paper: `-0.094`  recomputed: ``
- note: 

### §4.2 / Conclusion elicitation raises invocation 10–50×
- section: §4.2 / Conclusion
- paper: `10-50×`  recomputed: `rederive algorithm_invocation=0.0 for all models (response_type contains 'algo' mean)`
- note: Need algorithm_invocation_clean.csv / reasoning_type==algorithm_invocation rates, not rederive's response_type metric

### ~20,000 API calls
- section: Abstract / Conclusion
- paper: `20000`  recomputed: ``
- note: Would require summing all raw rows across probes; not verified here as a billed-call total

### Appendix population Spearman r=+0.147 p=0.46 n=28
- section: Appendix
- paper: `+0.147`  recomputed: ``
- note: Depends on triangulation instance_type map (disagrees with frozen 34/10/17); prior camera-ready audit matched by re-running the figure script

## Counts

| status | n |
|--------|--:|
| MATCH | 373 |
| MISMATCH | 15 |
| NOT_RECOMPUTABLE | 21 |
| **total ledger rows** | **409** |

## Filters (where they live; which rows they drop)

### 1. Bank-valid GSM ID list

**Lives in** `data/problems/question_bank_gsm.csv` canonical `problem_id`s; applied by `scripts/runs/coverage_audit.py:filter_p1_to_bank` and then `_accuracy` in `rederive_all_metrics.py` (drops `ERROR:`).

- Bank canonical IDs (n=44): `GSM_001, GSM_002, GSM_003, GSM_004, GSM_005, GSM_006, GSM_007, GSM_008, GSM_009, GSM_010, GSM_011, GSM_012, GSM_013, GSM_014, GSM_015, GSM_016, GSM_017, GSM_018, GSM_019, GSM_020, GSM_041, GSM_042, GSM_043, GSM_044, GSM_045, GSM_046, GSM_047, GSM_048, GSM_049, GSM_050, GSM_051, GSM_052, GSM_053, GSM_054, GSM_055, GSM_056, GSM_057, GSM_058, GSM_059, GSM_060, GSM_061, GSM_062, GSM_063, GSM_064`
- GSM_001–020 (n=20): in bank
- GSM_041–064 (n=24): in bank, but GPT-4o/Llama raw rows are OpenRouter **402 Payment Required** placeholders → excluded from n_valid
- GSM_021–040: **not in the bank**. Present in GPT-4o/Llama raw files as **real model outputs** (duplicate reruns of 001–020). Excluded by `filter_p1_to_bank`.

| model | n_valid canonical after filters | excluded |
|-------|-------------------------------:|----------|
| Claude, Gemini, o4-mini | 44 | none |
| GPT-4o, Llama | 20 (GSM_001–020) | GSM_021–040 off-bank (140 rows each); GSM_041–064 ERROR:402 (168 rows each) |

Per-model CSV: `bank_valid_gsm_ids_per_model.csv`.

### 2. Adversarial ALGO ID list (paper expects 34 SP, 10 CC, 17 WIS)

**This is a frozen list, not `question_bank_algo.csv` `instance_type`, and not a consistent `difficulty_params_instance_type` column across model files.**

| source | CC adv | SP adv | WIS adv |
|--------|-------:|-------:|--------:|
| `question_bank_algo.csv` `instance_type` | 10 | 31 | 30 |
| Paper / frozen labels n / claude·gemini·o1mini `difficulty_params_instance_type` | 10 | 34 | 17 |
| gpt4o + llama `difficulty_params_instance_type` | 0 | 31 | 15 |

Frozen SP/CC/WIS IDs (paper Table 5/7 challenging cells) from claude `difficulty_params_instance_type`:
- CC (10): `CC_01, CC_02, CC_03, CC_04, CC_05, CC_06, CC_07, CC_08, CC_09, CC_10`
- SP (34): `SP_003, SP_004, SP_005, SP_019, SP_020, SP_021, SP_023, SP_024, SP_026, SP_027, SP_028, SP_029, SP_030, SP_037, SP_038, SP_039, SP_040, SP_042, SP_044, SP_045, SP_046, SP_047, SP_048, SP_062, SP_063, SP_064, SP_065, SP_066, SP_068, SP_069, SP_070, SP_071, SP_072, SP_073`
- WIS (17): `WIS_003, WIS_004, WIS_013, WIS_014, WIS_015, WIS_016, WIS_017, WIS_018, WIS_019, WIS_020, WIS_023, WIS_024, WIS_025, WIS_026, WIS_027, WIS_028, WIS_029`

gpt4o/llama disagree on SP by missing `SP_003, SP_004, SP_005` (31 vs 34) and on WIS by missing `WIS_003, WIS_004` (15 vs 17); **all 25 CC rows are labelled `standard`** in those two files (0 vs 10).
Full comparison: `adversarial_id_lists.csv`.
Table 7 caption points at `results/derived/ALGO_P1_4model_frozen_labels.csv` (n=34/10/17), which matches the claude-side column, **not** the bank column.

### 3. `model=='mock'`

| file | n mock | keys | dropped? |
|------|-------:|------|----------|
| `ALGO_P1_behavioral_claude.csv` | 3 | CC_01 canonical, W1, W2 | yes — real Claude rows come **after** mock, so `drop_duplicates(keep='last')` keeps Claude |
| `ALGO_P1_behavioral_llama.csv` | 2 | CC_01 canonical, W1 | **not dropped by keep='last'** — mock rows are *after* Llama, so rederive would keep mock True on those two keys unless an explicit `model!='mock'` filter is applied |
| other ALGO P1 files | 0 | — | — |

Paper Table 7 uses frozen labels (mock already out). This audit’s regenerated VAR files **explicitly drop mock**. Confirmed 3+2 mock rows exist as stated. Details: `mock_rows.csv`.

## GSM_021–040 vs GSM_041–064 (task 4)

See `gsm_p1_id_slices.md`. Short version:
- **GSM_021–040:** 140 rows / 20 IDs per file; no `parse_status`/`verified`/`model_answer`; `behavioral_correct` mixed True/False; **real CoT model outputs** (not placeholders). Off-bank; excluded from paper n=20.
- **GSM_041–064:** 168 rows / 24 IDs per file; every `raw_response` is `ERROR: 402 Payment Required ... Insufficient credits` (length 154). Placeholders. In the bank, so `filter_p1_to_bank` keeps them, but `_accuracy` drops ERROR: → n_valid=20.

## Table 7 vs `results/paper/AUDIT/*_VAR_5model.csv`

Old GSM_VAR uses n=44 for GPT-4o/Llama (includes off-bank and/or ERROR-as-False). Paper Table 7 uses n=20 bank-valid. Regenerated files are in this directory. See `VAR_DIFF.md`.

## rederive_all_metrics.py notes

- Coverage matrix prints **ALGO_P1 and GSM_P1 n_valid=0** because `coverage_matrix()` calls `filter_p1_to_bank(df, "BW")` on every per-model P1 file. Probe-1 accuracy tables themselves use the correct family.
- Llama mock can leak into ALGO P1 via `keep='last'` (see filters).
- o4-mini GSM P2 CCI mean on **all 44** is 0.215; paper Table 4 uses **parseable 43/44** (0.220). Rederive step [4/6] prints the unfiltered mean.

