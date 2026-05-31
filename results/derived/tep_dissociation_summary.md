# TEP dissociation analysis (Step 8)

TEP = trajectory divergence after perturbation. **Dissociation** = high TEP (≥0.5) but final answer still correct — suggests re-derivation or answer independent of corrupted chain.

**Sessions analyzed:** 1061 ({'BW': 536, 'ALGO': 305, 'GSM': 220})
**TEP-valid sessions:** 590 (55.6%)

ALGO TEP uses paper-compliant steps; when empty, falls back to parseable (`format_ignored`) steps. BW TEP recomputed from `cascade_sequence_json` when CSV column is blank.

## Headline: TEP vs final correctness

```
family   model  n_tep_valid  corr_eligible  mean_tep  final_ok_rate  high_tep_final_ok_rate  spearman_tep_vs_final
  ALGO  Claude           58           True     0.155          0.552                   0.455                 -0.097
  ALGO  GPT-4o           61           True     0.229          0.557                   0.444                 -0.204
  ALGO  Gemini           61           True     0.183          0.311                   0.231                 -0.073
  ALGO   Llama           61           True     0.590          0.230                   0.154                 -0.180
  ALGO o4-mini           61           True     0.344          0.443                   0.357                 -0.012
    BW  Claude           47          False     0.064          0.000                   0.000                    NaN
    BW  GPT-4o           13          False     0.000          0.000                     NaN                    NaN
    BW   Llama            8          False     0.625          0.000                   0.000                    NaN
   GSM  Claude           44           True     0.539          0.864                   0.852                 -0.096
   GSM  GPT-4o           44           True     0.598          0.705                   0.833                  0.374
   GSM  Gemini           44           True     0.652          0.886                   0.879                 -0.071
   GSM   Llama           44           True     0.773          0.455                   0.514                  0.261
   GSM o4-mini           44           True     0.637          0.955                   0.933                 -0.232
```

**Interpretation:** Weak or positive Spearman on GSM/ALGO for some models → high TEP does not imply wrong finals.

## Dissociation rate (high TEP + correct final)

```
family   model       subtype  n_sessions  mean_tep  final_ok_rate  pct_dissociated_high_tep_correct
  ALGO  Claude   coin_change          10     0.200          0.200                             0.100
  ALGO  Claude shortest_path          34     0.147          0.559                             0.088
  ALGO  Claude           wis          17     0.143          0.706                             0.059
  ALGO  GPT-4o   coin_change          10     0.333          0.500                             0.200
  ALGO  GPT-4o shortest_path          34     0.245          0.412                             0.147
  ALGO  GPT-4o           wis          17     0.135          0.882                             0.059
  ALGO  Gemini   coin_change          10     0.167          0.000                             0.000
  ALGO  Gemini shortest_path          34     0.132          0.118                             0.000
  ALGO  Gemini           wis          17     0.294          0.882                             0.176
  ALGO   Llama   coin_change          10     0.700          0.200                             0.100
  ALGO   Llama shortest_path          34     0.534          0.088                             0.029
  ALGO   Llama           wis          17     0.635          0.529                             0.235
  ALGO o4-mini   coin_change          10     0.367          1.000                             0.400
  ALGO o4-mini shortest_path          34     0.309          0.059                             0.029
  ALGO o4-mini           wis          17     0.402          0.882                             0.294
    BW  Claude         (all)         177     0.064          0.000                             0.000
    BW  GPT-4o         (all)         180     0.000          0.000                             0.000
    BW   Llama         (all)         179     0.625          0.000                             0.000
   GSM  Claude         (all)          44     0.539          0.864                             0.523
   GSM  GPT-4o         (all)          44     0.598          0.705                             0.568
   GSM  Gemini         (all)          44     0.652          0.886                             0.659
   GSM   Llama         (all)          44     0.773          0.455                             0.409
   GSM o4-mini         (all)          44     0.637          0.955                             0.636
```

## Quadrant counts (all families)

- **tep_missing**: 471 (44.4%)
- **dissociated_low_tep_wrong_final**: 177 (16.7%)
- **dissociated_high_tep_correct_final**: 155 (14.6%)
- **aligned_low_tep_correct_final**: 141 (13.3%)
- **aligned_high_tep_wrong_final**: 117 (11.0%)

## ALGO injection recovery (post-injection correct)

```
       mode model_short       subtype inj_reasoning_type  critical_step  n  post_ok_rate
  plausible     o4-mini shortest_path       local_greedy              0 30         0.000
implausible     o4-mini shortest_path       local_greedy              0 30         0.033
  plausible      GPT-4o shortest_path            unclear              0 29         0.379
implausible      Claude shortest_path            unclear              0 28         0.571
implausible      Gemini shortest_path            unclear              0 28         0.143
implausible      GPT-4o shortest_path            unclear              0 28         0.500
  plausible      Claude shortest_path            unclear              0 27         0.630
  plausible      Gemini shortest_path            unclear              0 26         0.154
implausible       Llama shortest_path            unclear              0 21         0.143
  plausible       Llama shortest_path            unclear              0 20         0.100
implausible      GPT-4o           wis       local_greedy              0 16         0.938
implausible      Claude           wis       local_greedy              0 13         0.846
  plausible      GPT-4o           wis       local_greedy              0 13         0.923
  plausible      Claude           wis       local_greedy              0 12         0.750
implausible      Gemini           wis            unclear              0  9         0.778
  plausible       Llama           wis            unclear              0  8         0.500
implausible     o4-mini           wis       local_greedy              0  8         0.875
implausible     o4-mini           wis            unclear              0  8         1.000
  plausible      Gemini           wis            unclear              0  8         0.875
  plausible     o4-mini           wis       local_greedy              0  8         0.875
```

## Mechanistic hypotheses (for paper discussion)

1. **Re-derivation:** model recomputes from problem statement after chain corruption (GSM numeric steps).
2. **Terminal correction:** wrong intermediate steps but correct final aggregation (coin-change).
3. **Format compliance without state:** ALGO `compliant` steps after injection may diverge in token but still reach correct final state.
4. **BW protocol noise:** many BW TEP rows reflect parser/session abort — interpret BW separately.

## Files

- `tep_dissociation_sessions.csv` — per-session TEP, final_ok, dissociation label
- `tep_dissociation_by_slice.csv` — family × model × subtype aggregates
- `tep_dissociation_correlations.csv` — Spearman TEP vs final_ok
- `tep_injection_recovery.csv` — ALGO injection recovery by reasoning type
- `tep_dissociation_scatter.csv` — TEP-valid rows for scatter plots
- `tep_dissociation_quality_audit.md` — row validation and exclusion log

