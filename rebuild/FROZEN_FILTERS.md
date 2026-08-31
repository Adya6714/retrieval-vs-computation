# Frozen filters

These lists are applied everywhere in `rebuild/NUMBERS.csv`.
Mock rows (`model == 'mock'`) are dropped explicitly before `drop_duplicates`.

## GSM bank-valid IDs

Source: `data/problems/question_bank_gsm.csv` canonical `problem_id`s. n=44.

```
GSM_001, GSM_002, GSM_003, GSM_004, GSM_005, GSM_006, GSM_007, GSM_008, GSM_009, GSM_010, GSM_011, GSM_012, GSM_013, GSM_014, GSM_015, GSM_016, GSM_017, GSM_018, GSM_019, GSM_020, GSM_041, GSM_042, GSM_043, GSM_044, GSM_045, GSM_046, GSM_047, GSM_048, GSM_049, GSM_050, GSM_051, GSM_052, GSM_053, GSM_054, GSM_055, GSM_056, GSM_057, GSM_058, GSM_059, GSM_060, GSM_061, GSM_062, GSM_063, GSM_064
```

Notes:
- GSM_001–020 (n=20) are in the bank.
- GSM_041–064 (n=24) are in the bank. GPT-4o/Llama raw rows for these IDs are `ERROR: 402 Payment Required` placeholders and are dropped by the ERROR: filter, so those two models have n_valid=20.
- GSM_021–040 are **not** in the bank (duplicate reruns of 001–020 in GPT-4o/Llama files). Excluded by the bank filter.

## ALGO adversarial pool (frozen, 61)

Paper Table 5/7 challenging cells. **Not** `question_bank_algo.csv` `instance_type`.
Taken from Claude P1 `difficulty_params_instance_type == adversarial`.
34 SP + 10 CC + 17 WIS = 61.
Paper §4.3 says n=64; the released frozen list is 61.

### CC (10)

```
CC_01, CC_02, CC_03, CC_04, CC_05, CC_06, CC_07, CC_08, CC_09, CC_10
```

### SP (34)

```
SP_003, SP_004, SP_005, SP_019, SP_020, SP_021, SP_023, SP_024, SP_026, SP_027, SP_028, SP_029, SP_030, SP_037, SP_038, SP_039, SP_040, SP_042, SP_044, SP_045, SP_046, SP_047, SP_048, SP_062, SP_063, SP_064, SP_065, SP_066, SP_068, SP_069, SP_070, SP_071, SP_072, SP_073
```

### WIS (17)

```
WIS_003, WIS_004, WIS_013, WIS_014, WIS_015, WIS_016, WIS_017, WIS_018, WIS_019, WIS_020, WIS_023, WIS_024, WIS_025, WIS_026, WIS_027, WIS_028, WIS_029
```

## BW bank (65 PlanBench IDs)

Source: `data/problems/question_bank_bw.csv` canonical IDs. n=65 (50 standard BW_ + 15 mystery MBW_).

### Standard (BW_)

```
BW_001, BW_002, BW_010, BW_011, BW_014, BW_022, BW_080, BW_120, BW_137, BW_155, BW_172, BW_203, BW_227, BW_282, BW_310, BW_331, BW_350, BW_408, BW_467, BW_495, BW_496, BW_497, BW_498, BW_499, BW_500, BW_501, BW_502, BW_503, BW_504, BW_505, BW_506, BW_507, BW_508, BW_509, BW_510, BW_511, BW_512, BW_513, BW_514, BW_515, BW_E002, BW_E015, BW_E017, BW_E019, BW_E100, BW_E_001, BW_E_002, BW_E_003, BW_E_004, BW_E_005
```

### Mystery (MBW_)

```
MBW_001, MBW_002, MBW_010, MBW_014, MBW_037, MBW_10, MBW_100, MBW_127, MBW_185, MBW_495, MBW_496, MBW_497, MBW_498, MBW_499, MBW_500
```

## BW W3 vs W5 (generator confirmation)

- **W3** = entity + action rename. Prompt templates: `scripts/generation/utils/variant_prompts.py` `W3_BW_MAPPING_SYSTEM` / `W3_BW_MAPPING_USER` (`entity_mapping` + `action_mapping`).
- **W5** = init/goal swap. Implementation: `scripts/generation/utils/variant_utils.py` `swap_pddl_init_goal` ("W5 reversal: start from the original goal tower ... plan to the original flat init state").
- These are **different columns** in the behavioral CSVs (`variant_type == W3` vs `W5`). They must not be pooled.

## Mock drop

Every loader runs `_drop_mock` (`model` stripped, case-insensitive `mock`) **before** `drop_duplicates(..., keep='last')`.
Llama ALGO P1 has mock rows *after* real rows on two keys; keep-last without the mock drop would retain mock.
