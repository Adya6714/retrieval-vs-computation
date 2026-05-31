# Pre-API recovery audit

Before spending on API runs: check **existing raw files** and **sweep logs**.
Regenerate: `python scripts/runs/pre_api_master_audit.py`

**Total API calls avoided so far:** ~616 (GSM P2 o4-mini recovered)

## Revival checklist

| Gap | Status | Observed | Source | Log / note | Action | API saved |
|-----|--------|----------|--------|------------|--------|-----------|
| GSM P2 o4-mini | `recovered_no_api` | 44/44 | `results/raw/GSM_P2_phase1_o1mini.csv` | results/raw/new_model_sweep_logs/finish_o4mini.log (44/44 complete 2026-05-24) | Merge into GSM P2 derivations; coverage now counts o4-mini from this file | 616 |
| GSM P1 GPT-4o | `remap_invalid` | 20/44 bank-valid | `results/raw/GSM_P1_behavioral_gpt4o.csv` | No archived logs for GSM_041–064 | Exclude 20 duplicate IDs (GSM_021–040 = GSM_001–020); API still needed for 24 missing IDs (GSM_041–064) | 0 |
| GSM P1 Llama | `remap_invalid` | 20/44 bank-valid | `results/raw/GSM_P1_behavioral_llama.csv` | No archived logs for GSM_041–064 | Exclude 20 duplicate IDs (GSM_021–040 = GSM_001–020); API still needed for 24 missing IDs (GSM_041–064) | 0 |
| ALGO P2A elicited Claude | `needs_api` | 61/110 problems | `results/raw/ALGO_P2_phase2_normal_elicited.csv` | No separate elicited logs; 61/110 pilot (CC_031+ missing) | ~49 sessions × 4 steps ≈ 196 API calls | 0 |
| ALGO P2A elicited Gemini | `needs_api` | 61/110 problems | `results/raw/ALGO_P2_phase2_normal_elicited.csv` | No separate elicited logs; 61/110 pilot (CC_031+ missing) | ~49 sessions × 4 steps ≈ 196 API calls | 0 |
| ALGO P2A elicited Llama | `needs_api` | 61/110 problems | `results/raw/ALGO_P2_phase2_normal_elicited.csv` | No separate elicited logs; 61/110 pilot (CC_031+ missing) | ~49 sessions × 4 steps ≈ 196 API calls | 0 |
| BW P2 Gemini | `needs_api` | 0/65 | `results/raw/BW_P2_tep.csv (pilot)` | No BW P2 sweep logs for these models | ~1000 calls/model for 50-problem pilot (Step 20 optional) | 0 |
| BW P2 o4-mini | `needs_api` | 0/65 | `results/raw/BW_P2_tep.csv (pilot)` | No BW P2 sweep logs for these models | ~1000 calls/model for 50-problem pilot (Step 20 optional) | 0 |

## Log locations

- `results/raw/new_model_sweep_logs/finish_o4mini.log` — o4-mini GSM P1/P2/BW/ALGO resume (2026-05-24)
- `results/raw/new_model_sweep_logs/wait_and_finish.log` — credit-limit wait loop
- `results/raw/_failed_archive/` — superseded failed CSV snapshots (compare before discard)

## Status key

- `recovered_no_api` — data exists; update derivations only
- `remap_invalid` — do not relabel IDs; duplicates or wrong bank mapping
- `needs_api` — no recoverable rows in raw or logs
- `resume_not_full_rerun` — use `--resume` on existing output to retry ERROR rows

