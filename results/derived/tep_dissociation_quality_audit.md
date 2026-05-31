# TEP dissociation — data quality audit

Rows with blank `problem_id` or `model` are dropped on load. All-empty CSV rows are removed. Spearman is computed only when `corr_eligible=True` (≥3 TEP-valid sessions, TEP and final_ok both vary).

## Session inventory

- Total sessions after validation: **1061**
- Empty/all-blank rows dropped at load: see per-family loaders
- TEP-valid: **590** (55.6%)
- Analysis-ready (TEP + final_ok vary within slice): see correlation table

### By family

```
family   n  tep_valid  tep_missing  final_ok_rate
  ALGO 305        302            3          0.416
    BW 536         68          468          0.000
   GSM 220        220            0          0.773
```

### TEP missing — expected causes

- **ALGO (3/305 missing):** no post-critical parseable steps (ALGO); not blank-row corruption.
- **BW (468/536 missing):** session aborted or empty cascade (BW); not blank-row corruption.
- **GSM (0/220 missing):** all rows have tep_score (GSM); not blank-row corruption.

### Spearman exclusions

```
family  model  n_tep_valid corr_exclude_reason
    BW Claude           47   final_ok_constant
    BW GPT-4o           13        tep_constant
    BW  Llama            8   final_ok_constant
```

BW slices excluded because **final_ok is always False** (zero variance) — correlation undefined, not a parsing error.

### Column nulls (expected, not bugs)

- `cci`: GSM-only
- `instance_type`, `tep_inclusive`: ALGO-only
- `first_response_class`, `session_status`: BW-only
- `subtype`: blank for BW (no subtype in probe)

