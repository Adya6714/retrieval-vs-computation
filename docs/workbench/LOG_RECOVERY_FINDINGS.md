# Log & archive recovery findings

Checked: `results/raw/new_model_sweep_logs/` (25 files) and `results/raw/_failed_archive/` (2 files).

## Short answer

**Most planned API gaps still need real API calls.** The logs confirm one major recovery (o4-mini GSM P2) and show Step 16 **attempted** GSM P1 but only wrote 402 ERROR rows — not hidden successes.

---

## What IS recoverable from logs/archives (no new API)

| Gap | Status | Source | Notes |
|-----|--------|--------|-------|
| GSM P2 o4-mini | ✅ **Recovered** | `GSM_P2_phase1_o1mini.csv` | `finish_o4mini.log` 44/44 (2026-05-24). Failed archive was all-parse-fail; **superseded** |
| BW P1 o4-mini | ✅ **Already complete** | `BW_P1_behavioral_o1mini.csv` | Failed archive = 795× ERROR; current = 455/455 good |
| Gemini BW/ALGO/GSM P2 | ✅ Complete per logs | `t1`–`t3b` logs | No hidden gaps |

**`_failed_archive/`** — only superseded snapshots; do not use over current raw files.

---

## What is NOT in logs (still needs API when wallet restored)

### GSM P1 GPT-4o / Llama — GSM_041–064

| Check | Result |
|-------|--------|
| `step16_gsm_p1_gpt4o_llama.log` | Ran 2026-05-30; says `processed=168, errors=0` |
| Actual CSV content | **All 168 new rows are `ERROR: 402 Payment Required`** |
| Any other log with GSM_041 success | **None** |
| `_failed_archive` | No GSM P1 files |

**Bank-valid today:** **20/44** (GSM_001–020).  
**Off-bank noise:** GSM_021–040 have responses but are **not in the bank** — exclude.  
**Still need:** 24 problems × 7 variants ≈ **168 calls/model** (same as Step 16 queue).

⚠️ Step 16 log `errors=0` is misleading — script counts HTTP responses written to CSV, not non-ERROR content.

### ALGO P2A elicited — Claude / Gemini / Llama (49/110 missing)

| Model | Elicited problems | Normal track (non-elicited) |
|-------|-------------------|----------------------------|
| Claude | 61/110 | 110/110 (49 problems never elicited) |
| Gemini | 61/110 | 110/110 |
| Llama | 61/110 | 110/110 |
| GPT-4o / o4-mini | 110/110 | 110/110 |

Missing IDs start at **CC_031** through **WIS_032** (49 problems).  
**No separate elicited sweep log** — `t3b_gemini_algo_p2.log` only ran normal + injected for Gemini, not elicited track.  
**No alternate CSV** — only `ALGO_P2_phase2_normal_elicited.csv`.

**Still need:** ~49 sessions × 4 steps × 3 models ≈ **588 calls**.

### BW P2 Gemini / o4-mini

No sweep logs (`t1`–`t5` never ran BW P2 for these models). **~2000 calls/model** if pursued (Step 20).

---

## Log index (`new_model_sweep_logs/`)

| Log | What it ran | Useful for recovery? |
|-----|-------------|---------------------|
| `finish_o4mini.log` | o4-mini GSM P1/P2, BW P1, ALGO P1 | ✅ GSM P2 o4-mini |
| `step16_gsm_p1_gpt4o_llama.log` | GSM P1 gap fill | ❌ all 402 |
| `t3_gemini_p2.log` | Gemini GSM P2 + ALGO P2 phase1 | ✅ complete |
| `t3b_gemini_algo_p2.log` | Gemini phase2 normal + injected | ✅ not elicited |
| `t2_gemini_algo_p1.log` | Gemini ALGO P1 | ✅ complete |
| `t1_gemini_bw.log` | Gemini BW P1 (resume skip) | ✅ already done |
| `wait_and_finish.log` | Credit wait loop + metrics | Context only |
| `t4/t5 o4mini/o1mini` | Partial / terminated | Incomplete runs |

---

## Action when API returns

1. **GSM P1:** `--resume` on existing CSVs — retries only `ERROR:` rows (GSM_041–064 variants).
2. **ALGO elicited:** Run phase2 elicited for missing 49 problem IDs × 3 models.
3. **Do not** relabel GSM_021–040 → GSM_041–060 (invalid).
4. Re-run `python scripts/runs/pre_api_master_audit.py` after successful fills.
