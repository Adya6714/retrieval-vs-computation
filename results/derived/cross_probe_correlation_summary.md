# Cross-probe correlation analysis (Step 11)

Links P1 W3 fragility, P2 CCI, and P3 contamination on shared instances.

**Instance rows (excl. mock):** 1249

## Model-level Spearman (5 models × family means)

```
family                      pair  n_models  spearman_rho  p_value
  ALGO      p1_w3_drop vs p2_cci         4           NaN      NaN
  ALGO   p1_w3_drop vs p3_contam         5           NaN      NaN
  ALGO       p2_cci vs p3_contam         4           NaN      NaN
  ALGO p1_w3_retention vs p2_cci         4           NaN      NaN
    BW      p1_w3_drop vs p2_cci         3           NaN      NaN
    BW   p1_w3_drop vs p3_contam         5           NaN      NaN
    BW       p2_cci vs p3_contam         3           NaN      NaN
    BW p1_w3_retention vs p2_cci         3           NaN      NaN
   GSM      p1_w3_drop vs p2_cci         5        -0.500   0.3910
   GSM   p1_w3_drop vs p3_contam         5         0.866   0.0577
   GSM       p2_cci vs p3_contam         5        -0.866   0.0577
   GSM p1_w3_retention vs p2_cci         5         0.500   0.3910
```

## Per-model instance Spearman (selected pairs)

```
family   model                      pair  n  spearman_rho  p_value
  ALGO  GPT-4o P1_W3_retention vs P2_CCI 26        0.5641   0.0027
   GSM   Llama      P1_W3_drop vs P2_CCI 20        0.3857   0.0931
    BW   Llama      P1_W3_drop vs P2_CCI 11       -0.3591   0.2781
   GSM   Llama P1_W3_retention vs P2_CCI 16       -0.3575   0.1740
  ALGO  Claude       P2_CCI vs P3_contam 61       -0.3426   0.0069
    BW   Llama       P2_CCI vs P3_contam 11        0.3420   0.3032
    BW  Claude   P1_W3_drop vs P3_contam 65        0.3245   0.0084
    BW o4-mini      VRI_gap vs P3_contam 65       -0.2751   0.0266
   GSM  Gemini P1_W3_retention vs P2_CCI 40        0.2662   0.0968
    BW  Claude       P2_CCI vs P3_contam  8       -0.2535   0.5446
   GSM   Llama   P1_W3_drop vs P3_contam 20       -0.2454   0.2970
   GSM   Llama      VRI_gap vs P3_contam 20       -0.2454   0.2970
   GSM o4-mini P1_W3_retention vs P2_CCI 37       -0.2423   0.1485
   GSM  Claude       P2_CCI vs P3_contam 44        0.2407   0.1155
    BW o4-mini   P1_W3_drop vs P3_contam 65       -0.2351   0.0594
   GSM o4-mini      P1_W3_drop vs P2_CCI 44        0.2262   0.1399
  ALGO   Llama       P2_CCI vs P3_contam 61       -0.1873   0.1484
   GSM  Gemini      P1_W3_drop vs P2_CCI 44       -0.1766   0.2515
  ALGO  Gemini P1_W3_retention vs P2_CCI 34       -0.1748   0.3229
   GSM  GPT-4o      P1_W3_drop vs P2_CCI 20       -0.1600   0.5005
```

## Triple-probe agreement (threshold flags)

- P1 fragile: w3_drop≥0.5 or vri_gap>0.5 · P2 low: cci≤0.3 · P3 high: contam≥0.6

```
family   model  n_instances  triple_retrieval_agree_pct  triple_computation_agree_pct  triple_mixed_pct  p2_cci_valid_n
  ALGO  Claude          110                         0.0                         0.009             0.991              61
  ALGO  GPT-4o          110                         0.0                         0.036             0.964              61
  ALGO  Gemini          110                         0.0                         0.009             0.991              61
  ALGO   Llama          110                         0.0                         0.000             1.000              61
  ALGO o4-mini          110                         0.0                         0.000             1.000               0
    BW  Claude          109                         0.0                         0.000             1.000               8
    BW  GPT-4o          109                         0.0                         0.000             1.000              10
    BW  Gemini           65                         0.0                         0.000             1.000               0
    BW   Llama          109                         0.0                         0.000             1.000              11
    BW o4-mini           65                         0.0                         0.000             1.000               0
   GSM  Claude           44                         0.0                         0.136             0.864              44
   GSM  GPT-4o           40                         0.0                         0.000             1.000              20
   GSM  Gemini           44                         0.0                         0.136             0.864              44
   GSM   Llama           40                         0.0                         0.000             1.000              20
   GSM o4-mini           44                         0.0                         0.091             0.909              44
```

**All-family totals:** retrieval-agree 0/1249 (0.0%) · computation-agree 22/1249 (1.8%)

## Accuracy vs W3 retention (across models, refreshed)

```
probe  n_models  spearman_rho  p_value                        models_used
 ALGO         5        1.0000   0.0000 Claude,GPT-4o,Gemini,Llama,o4-mini
   BW         5       -0.1000   0.8729 Claude,GPT-4o,Gemini,Llama,o4-mini
  GSM         5        0.6325   0.2522 Claude,GPT-4o,Gemini,Llama,o4-mini
```

## Coverage caveats

- **BW:** P2 CCI sparse in triangulation labels — merged from `BW_P2_cci.csv` where available.
- **GSM:** P3 contam missing for ~40 instances; P2 CCI missing for ~84.
- **ALGO:** P2 CCI only on adversarial subset (~244/550 instances).

## Files

- `cross_probe_instance_frame.csv`
- `cross_probe_spearman_by_model.csv`
- `cross_probe_spearman_model_level.csv`
- `cross_probe_agreement_instances.csv`
- `cross_probe_acc_vs_w3retention.csv`

