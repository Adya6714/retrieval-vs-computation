# Regenerated VAR vs old AUDIT files vs paper Table 7

Paper filters used for regeneration:
- Drop `model==mock`
- `filter_p1_to_bank` (question bank problem_id×variant)
- Drop rows whose raw_response/model_answer starts with `ERROR:`
- GSM GPT-4o/Llama: n=20 because GSM_041-064 are 402 placeholders (still in bank, excluded by ERROR filter)
- ALGO family-level file uses all 110 bank IDs (Table 7 ALGO is *sliced*; see ALGO_VAR_5model_table7_slices.csv)
- BW n=65 bank IDs, `behavioral_correct`

## Diff vs old results/paper/AUDIT/{GSM,BW,ALGO}_VAR_5model.csv

### GSM: 22 field mismatches
family  model      field   old       new
   GSM claude  canonical 0.818  0.840909
   GSM claude         W2 0.682  0.772727
   GSM claude         W4 0.682  0.636364
   GSM claude         W5 0.864  0.818182
   GSM claude         W6 0.792  0.750000
   GSM  gpt4o n_problems    44 20.000000
   GSM  gpt4o  canonical 0.818  0.850000
   GSM  gpt4o         W1 0.818  0.750000
   GSM  gpt4o         W2 0.591  0.300000
   GSM  gpt4o         W3 0.477  0.300000
   GSM  gpt4o         W4 0.477  0.200000
   GSM  gpt4o         W5 0.545  0.300000
   GSM  gpt4o         W6 0.792  0.800000
   GSM  llama n_problems    44 20.000000
   GSM  llama  canonical 0.773  0.800000
   GSM  llama         W1 0.727  0.850000
   GSM  llama         W2 0.455  0.250000
   GSM  llama         W3 0.409  0.150000
   GSM  llama         W4 0.318  0.300000
   GSM  llama         W5  0.25  0.050000
   GSM  llama         W6 0.625  0.450000
   GSM o4mini  canonical 0.879  0.840909

### BW: 23 field mismatches
family  model      field   old       new
    BW claude         W5 0.523  0.566667
    BW claude         W6 0.508       NaN
    BW  gpt4o  canonical 0.089  0.061538
    BW  gpt4o         W1 0.085  0.092308
    BW  gpt4o         W2 0.067  0.092308
    BW  gpt4o         W3 0.244  0.169231
    BW  gpt4o         W4 0.044  0.076923
    BW  gpt4o         W5 0.178  0.266667
    BW  gpt4o         W6 0.215       NaN
    BW  llama  canonical 0.022  0.015385
    BW  llama         W1   0.0  0.030769
    BW  llama         W2   0.0  0.015385
    BW  llama         W3 0.156  0.107692
    BW  llama         W6 0.031       NaN
    BW gemini         W5 0.569  0.616667
    BW gemini         W6 0.338       NaN
    BW o4mini n_problems       65.000000
    BW o4mini  canonical        0.769231
    BW o4mini         W1        0.753846
    BW o4mini         W2        0.738462
    BW o4mini         W3        0.184615
    BW o4mini         W4        0.415385
    BW o4mini         W5        0.833333

### ALGO: 11 field mismatches
family  model      field   old        new
  ALGO claude         W6   0.1        NaN
  ALGO  gpt4o         W6   0.2        NaN
  ALGO  llama         W6 0.056        NaN
  ALGO gemini  canonical 0.109   0.463636
  ALGO gemini         W6 0.144        NaN
  ALGO o4mini n_problems    20 110.000000
  ALGO o4mini         W1  0.95   0.963636
  ALGO o4mini         W2  0.95   0.254545
  ALGO o4mini         W3   0.7   0.609091
  ALGO o4mini         W4  0.85   0.736364
  ALGO o4mini         W5         0.040000

## Diff vs paper Table 7

### GSM (regenerated GSM_VAR_5model.csv vs Table 7)

### BW (regenerated BW_VAR_5model.csv vs Table 7)
- Claude W5: paper .523  regenerated 0.5666666666666667  **MISMATCH**
- GPT-4o W5: paper .246  regenerated 0.26666666666666666  **MISMATCH**
- Gemini W5: paper .569  regenerated 0.6166666666666667  **MISMATCH**
- o4-mini W5: paper .769  regenerated 0.8333333333333334  **MISMATCH**

### ALGO family-level regenerated file vs Table 7
Structural mismatch: old and regenerated `ALGO_VAR_5model.csv` are **overall 110-problem** accuracies;
Table 7 is **6 subtype×difficulty slices** (CC/SP/WIS × chall/std). Sliced regeneration is in `ALGO_VAR_5model_table7_slices.csv` (frozen labels).
Old ALGO_VAR o4mini n_problems=['20']
New ALGO_VAR o4-mini n=[110] canonical=[1.0]
