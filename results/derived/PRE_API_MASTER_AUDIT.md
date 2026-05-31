# Pre-API master audit

Single inventory of **missing data**, **unusable/excluded data**, and **estimated API calls**.
Regenerate: `python scripts/runs/pre_api_master_audit.py`

## Executive summary

| Metric | Value |
|--------|-------|
| Coverage slices (master table) | **35/40** bank-complete |
| Missing ID rows (long form) | **483** |
| Unusable / scoped flags | **33** |
| Est. API calls — **core** (GSM+ALGO, Steps 16–18) | **~924** |
| Est. API calls — **BW P2 pilot** (Step 20, optional) | **~2300** |
| Est. API calls — all api_* rows | **~3224** |
| **Recovered without API** | GSM P2 o4-mini **44/44** in `GSM_P2_phase1_o1mini.csv` → saves **~616** calls |
| GSM P1 remap | **Invalid** — GSM_021–040 duplicate GSM_001–020; still need API for GSM_041–064 |

See **`PRE_API_RECOVERY_AUDIT.md`** for log/file revival checklist.

## 1. Missing data — needs runs or remap

```
family        probe  model  observed_canonical_n  bank_canonical_n coverage_label
   GSM           P1 GPT-4o                    20                44        partial
   GSM           P1  Llama                    20                44        partial
  ALGO P2A_elicited Claude                    61               110        partial
  ALGO P2A_elicited Gemini                    61               110        partial
  ALGO P2A_elicited  Llama                    61               110        partial
```

### Counts by cell

```
family        probe  model  missing_canonical_n          run_type priority
   GSM           P1 GPT-4o                   24  partial_p1_rerun       P1
   GSM           P1  Llama                   24  partial_p1_rerun       P1
  ALGO P2A_elicited Claude                   49 partial_probe_run       P1
  ALGO P2A_elicited Gemini                   49 partial_probe_run       P1
  ALGO P2A_elicited  Llama                   49 partial_probe_run       P1
```

## 2. Uncanny / excluded from analysis (exists but don't use blindly)

| Issue | Where | Action |
|-------|-------|--------|
| **GSM_021–040 in P1 raw** | GPT-4o, Llama | Duplicate reruns of GSM_001–020 — **exclude**; not remap to 041–060 |
| **GSM P2 o4-mini** | `GSM_P2_phase1_o1mini.csv` | **44/44 exists** — merge into derivations (Step 16 not needed) |
| **BW P1 GSM contamination** | `BW_P1_behavioral.csv` | Filter to bank in derivations (done) |
| **BW P2 pilot** | 50/65 problems, 3/5 models | Scope claims; Step 20 protocol fix |
| **BW TEP 87% missing** | 468/536 sessions | Aborted sessions — not blank rows |
| **BW final_ok always False** | P2 TEP slice | Spearman undefined |
| **ALGO P2B n=61** | Injection CSVs | By design — label denominator |
| **P3 mechanistic** | Qwen 0.5B only | Exploratory — not five-model |
| **GSM P2 five-model** | o4-mini in phase1_o1mini file | Five-model P2 OK after merge into loaders |

Full flag list → `pre_api_unusable_flags.csv`

## 3. BW P2 (not in master coverage table)

```
   probe   model  observed_canonical_n  bank_canonical_n coverage_label
  P2_tep  Claude                    50                65        partial
  P2_tep  Gemini                     0                65  missing_model
  P2_tep  GPT-4o                    50                65        partial
  P2_tep   Llama                    50                65        partial
  P2_tep o4-mini                     0                65  missing_model
  P2_cci  Claude                    50                65        partial
  P2_cci  Gemini                     0                65  missing_model
  P2_cci  GPT-4o                    50                65        partial
  P2_cci   Llama                    50                65        partial
  P2_cci o4-mini                     0                65  missing_model
P2_plans  Claude                    50                65        partial
P2_plans  Gemini                     0                65  missing_model
P2_plans  GPT-4o                    50                65        partial
P2_plans   Llama                    50                65        partial
P2_plans o4-mini                     0                65  missing_model
```

## 4. TEP / P2 analysis usability

```
family  probe   model  n_sessions  tep_valid_n  tep_missing_n  final_ok_rate              issue     analysis_use
  ALGO P2_tep  Claude          61           58              3          0.541                                  ok
  ALGO P2_tep  GPT-4o          61           61              0          0.557                                  ok
  ALGO P2_tep  Gemini          61           61              0          0.311                                  ok
  ALGO P2_tep   Llama          61           61              0          0.230                                  ok
  ALGO P2_tep o4-mini          61           61              0          0.443                                  ok
    BW P2_tep  Claude         177           47            130          0.000 tep_mostly_missing exclude_or_scope
    BW P2_tep  GPT-4o         180           13            167          0.000 tep_mostly_missing exclude_or_scope
    BW P2_tep   Llama         179            8            171          0.000 tep_mostly_missing exclude_or_scope
   GSM P2_tep  Claude          44           44              0          0.864                                  ok
   GSM P2_tep  GPT-4o          44           44              0          0.705                                  ok
   GSM P2_tep  Gemini          44           44              0          0.886                                  ok
   GSM P2_tep   Llama          44           44              0          0.455                                  ok
   GSM P2_tep o4-mini          44           44              0          0.955                                  ok
```

## 5. API budget (estimated)

```
priority family        probe      model                                         task  units  est_api_calls          fix_type
      P1   ALGO P2A_elicited     Claude                   ALGO P2A elicited sessions     49            196 api_partial_probe
      P1   ALGO P2A_elicited     Gemini                   ALGO P2A elicited sessions     49            196 api_partial_probe
      P1   ALGO P2A_elicited      Llama                   ALGO P2A elicited sessions     49            196 api_partial_probe
      P1     BW           P2     Gemini      BW P2 full model run (50-problem pilot)     50           1000    api_full_probe
      P1     BW           P2    o4-mini      BW P2 full model run (50-problem pilot)     50           1000    api_full_probe
      P1    GSM           P1     GPT-4o P1 behavioral missing bank IDs (GSM_041–064)     24            168    api_partial_p1
      P1    GSM           P1      Llama P1 behavioral missing bank IDs (GSM_041–064)     24            168    api_partial_p1
      P2     BW           P2 all_models      BW P2 extend to full bank (65 problems)     15            300 api_partial_probe
```

### Priority interpretation

- **P0:** ~~GSM P2 o4-mini~~ **recovered** from existing file — wire merged loader only
- **P1 core:** ALGO P2A elicited ×3 models (~588 calls); GSM P1 GPT-4o/Llama missing GSM_041–064 (~336 calls)
- **P2 optional:** BW P2 pilot extension (~2300 calls — Step 20; defer until protocol fixed)

## 6. Output files

| File | Contents |
|------|----------|
| `pre_api_slice_inventory.csv` | All slices + BW P2 pilot rows |
| `pre_api_missing_ids.csv` | Every missing canonical/session ID |
| `pre_api_unusable_flags.csv` | Exclusion / uncanny flags |
| `pre_api_api_budget.csv` | API estimates by task |

