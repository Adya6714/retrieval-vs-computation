# Triangulation exploratory analysis (Step 10)

Characterization only — **do not** treat as official paper thresholds (Step 13).

**Instances (excl. mock):** 1249

## Default v2 label mix (all families)

```
tri_v2_label
insufficient        457
computation         338
weak_retrieval      168
retrieval           160
mixed                67
weak_computation     59
```

## Label distribution by family × model (top slices)

```
family   model             subtype   tri_v2_label  n   pct
    BW   Llama         blocksworld   insufficient 49 0.980
    BW  GPT-4o         blocksworld   insufficient 46 0.920
    BW o4-mini         blocksworld weak_retrieval 43 0.860
  ALGO o4-mini       shortest_path    computation 41 0.745
    BW  Claude         blocksworld   insufficient 40 0.800
   GSM o4-mini        gsm_symbolic    computation 32 0.821
    BW  Claude               (all)    computation 29 0.659
    BW   Llama               (all)   insufficient 26 0.441
    BW  Gemini         blocksworld   insufficient 25 0.500
   GSM  Claude        gsm_symbolic    computation 25 0.641
  ALGO   Llama       shortest_path   insufficient 23 0.418
    BW  GPT-4o               (all)   insufficient 23 0.390
    BW  Gemini         blocksworld weak_retrieval 22 0.440
  ALGO  Claude       shortest_path   insufficient 21 0.382
   GSM  Gemini        gsm_symbolic    computation 20 0.513
    BW  GPT-4o               (all)    computation 20 0.339
  ALGO  GPT-4o       shortest_path    computation 19 0.345
    BW   Llama               (all)      retrieval 19 0.322
  ALGO o4-mini         coin_change    computation 17 0.680
    BW  GPT-4o               (all)      retrieval 16 0.271
    BW  Claude mystery_blocksworld   insufficient 15 1.000
  ALGO   Llama         coin_change   insufficient 15 0.600
  ALGO o4-mini                 wis          mixed 15 0.500
    BW  GPT-4o mystery_blocksworld   insufficient 15 1.000
  ALGO  Gemini       shortest_path      retrieval 15 0.273
```

## Vote signals — highest fire rates by family

**ALGO:**
```
      vote_signal  fire_rate
    p3_contam_low      0.718
     p2_impl_fail      0.327
p1_rename_fragile      0.258
      p1_vri_high      0.258
 p2_impl_recovery      0.227
```

**BW:**
```
      vote_signal  fire_rate
    p3_contam_low      0.667
p1_rename_fragile      0.248
      p1_vri_high      0.248
       p1_w3_keep      0.154
 p1_multi_variant      0.152
```

**GSM:**
```
      vote_signal  fire_rate
    p3_contam_low      0.811
      p2_cci_retr      0.561
       p1_w3_keep      0.486
 p1_multi_variant      0.486
p1_rename_fragile      0.363
```

## Stability (reference config set)

```
                           metric  value  n_instances    pct                                       note
               instances_any_flip    492         1249 0.3939 Distinct labels across 6 reference configs
               instances_all_same    757         1249 0.6061      Stable label across reference configs
flip_default_vs_best_sweep_id_204    301         1249 0.2410     Instance label changed between configs
      flip_default_vs_min_votes_3    231         1249 0.1849     Instance label changed between configs
    flip_default_vs_vote_margin_1    294         1249 0.2354     Instance label changed between configs
```

### Sweep: mean Δ strong-label rate between adjacent param values

```
                 param  from_val  to_val  mean_pct_strong_from  mean_pct_strong_to  mean_delta_strong  n_configs_from  n_configs_to
           vote_margin      1.00     2.0                0.5047              0.3087            -0.1961             972           972
             min_votes      2.00     3.0                0.4716              0.3419            -0.1297             972           972
  contam_retrieval_min      0.50     0.6                0.4128              0.4040            -0.0088             648           648
contam_computation_max      0.40     0.5                0.4041              0.4120             0.0079             648           648
   cci_computation_min      0.40     0.5                0.4057              0.4068             0.0011             648           648
   cci_computation_min      0.50     0.6                0.4068              0.4076             0.0008             648           648
  contam_retrieval_min      0.60     0.7                0.4040              0.4033            -0.0008             648           648
     cci_retrieval_max      0.25     0.3                0.4062              0.4070             0.0008             648           648
     cci_retrieval_max      0.30     0.4                0.4070              0.4069            -0.0001             648           648
      w3_retrieval_max      0.20     0.3                0.4067              0.4067             0.0000             648           648
      w3_retrieval_max      0.30     0.5                0.4067              0.4067             0.0000             648           648
    w3_computation_min      0.50     0.7                0.4067              0.4067             0.0000             972           972
```

## Legacy vs v2 overlap (ALGO, n=330 with legacy label)

**Bucket summary:**
```
legacy_bucket    v2_bucket   n   pct
    ambiguous insufficient 121 0.220
    ambiguous        mixed   6 0.011
    ambiguous       strong  35 0.064
    ambiguous         weak  43 0.078
        mixed insufficient   4 0.007
        mixed        mixed  13 0.024
        mixed       strong  56 0.102
        mixed         weak  42 0.076
        other insufficient  32 0.058
        other        mixed  20 0.036
        other       strong 114 0.207
        other         weak  54 0.098
       strong       strong  10 0.018
```

**Full crosstab (top rows):**
```
legacy_label     tri_v2_label   n  pct_of_overlap
   ambiguous     insufficient 121           0.220
         NaN      computation  85           0.155
       mixed      computation  47           0.085
         NaN   weak_retrieval  33           0.060
         NaN     insufficient  32           0.058
         NaN        retrieval  29           0.053
   ambiguous   weak_retrieval  27           0.049
       mixed weak_computation  22           0.040
   ambiguous      computation  22           0.040
         NaN weak_computation  21           0.038
       mixed   weak_retrieval  20           0.036
         NaN            mixed  20           0.036
   ambiguous weak_computation  16           0.029
       mixed            mixed  13           0.024
   ambiguous        retrieval  13           0.024
```

## Interpretation notes

- **~37% insufficient** under default thresholds — many instances lack enough firing votes.
- **P3 contam_low** and **P1 w3_keep / multi_variant** (GSM) fire most often; **p2_match_first** fires rarely.
- Legacy strong labels (~3%) vs v2 strong (~58% at best sweep) — legacy is much stricter.
- Label flips most when **`vote_margin`** (1→2) or **`min_votes`** (2→3) change — Step 13 must lock these.

## Files

- `triangulation_label_distribution.csv`
- `triangulation_vote_fire_rates.csv`
- `triangulation_sweep_stability.csv`
- `triangulation_legacy_v2_overlap.csv`

