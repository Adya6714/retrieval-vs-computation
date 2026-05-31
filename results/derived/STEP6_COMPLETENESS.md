# Step 6 — Results completeness

**Status:** regenerate pipeline OK
**Coverage:** 34/40 model×probe slices bank-complete
**Raw files indexed:** 46 CSVs in `results/raw/`

## Step 6 actions (this run)

- [x] Re-ran `rederive_all_metrics.py`, `deep_metrics_analysis.py`, `triangulation_v2.py`, `scientific_filewise_audit.py`
- [x] BW + ALGO P1 derivations now **filter to question bank** (`filter_p1_to_bank`)
- [x] Tagged API backlog → `api_backlog_tagged.csv`
- [x] Raw manifest → `results_manifest.csv`

## Fix types (no API for Step 6 itself)

| fix_type | Cells | API calls |
|----------|-------|-----------|
| derivation_cleanup | 0 | 0 |
| remap_and_or_api (GSM P1) | 2 | see backlog (missing IDs only) |
| api_* (runs deferred) | 4 | ~1204 estimated |

## Incomplete slices

```
family        probe   model          fix_type  est_api_calls  observed_canonical_n  bank_canonical_n
   GSM           P2 o4-mini    api_full_probe            616                     0                44
  ALGO P2A_elicited  Claude api_partial_probe            196                    61               110
  ALGO P2A_elicited  Gemini api_partial_probe            196                    61               110
  ALGO P2A_elicited   Llama api_partial_probe            196                    61               110
   GSM           P1  GPT-4o  remap_and_or_api            168                    20                44
   GSM           P1   Llama  remap_and_or_api            168                    20                44
```

## GSM P1 remap note

GPT-4o/Llama raw contains GSM_021–040 where bank expects GSM_041–064. Investigate whether rows can be **ID-remapped** without re-querying before spending ~168 calls/model on missing bank IDs.

## Phase 2 ready?

Yes — existing complete slices are bank-filtered in derivations. Proceed to Step 8 (TEP) without API spend.

