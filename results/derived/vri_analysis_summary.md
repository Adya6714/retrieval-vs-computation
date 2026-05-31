# VRI analysis (Step 9)

VRI = (Acc_W1 + Acc_W2 + Acc_W4) / 3 − Acc_W3. Positive VRI → W3 hurts more than vocabulary-preserving variants.

**Data quality:** GSM GPT-4o/Llama computed on **20/44 bank-valid** IDs only — interpret with caution.

## VRI vs W3 retention (by family × model)

```
family   model  acc_canonical  acc_w3    vri  w3_retention      bank_status
  ALGO  Claude          0.364   0.091  0.403         0.250        full_bank
  ALGO  GPT-4o          0.427   0.173  0.158         0.404        full_bank
  ALGO  Gemini          0.464   0.255  0.121         0.549        full_bank
  ALGO   Llama          0.064   0.018  0.052         0.286        full_bank
  ALGO o4-mini          1.000   0.609  0.042         0.609        full_bank
    BW  Claude          0.154   0.138 -0.036         0.900        full_bank
    BW  GPT-4o          0.062   0.169 -0.082         2.750        full_bank
    BW  Gemini          0.385   0.108 -0.015         0.280        full_bank
    BW   Llama          0.015   0.108 -0.092         7.000        full_bank
    BW o4-mini          0.769   0.185  0.451         0.240        full_bank
   GSM  Claude          0.841   0.750  0.000         0.892        full_bank
   GSM  GPT-4o          0.850   0.300  0.117         0.353 20/44 bank-valid
   GSM  Gemini          0.909   0.523  0.121         0.575        full_bank
   GSM   Llama          0.800   0.150  0.317         0.187 20/44 bank-valid
   GSM o4-mini          0.841   0.841 -0.053         1.000        full_bank
```

## VRI by subtype (ALGO)

```
family                 model       subtype    vri  acc_w3  acc_canonical  n_problems
  ALGO       claude-sonnet-4   coin_change  0.173   0.360          0.440          25
  ALGO       claude-sonnet-4 shortest_path  0.648   0.018          0.400          55
  ALGO       claude-sonnet-4           wis  0.144   0.000          0.233          30
  ALGO                gpt-4o   coin_change  0.413   0.040          0.400          25
  ALGO                gpt-4o shortest_path  0.091   0.327          0.527          55
  ALGO                gpt-4o           wis  0.067   0.000          0.267          30
  ALGO                gpt-4o           NaN    NaN     NaN            NaN           0
  ALGO llama-3.1-8b-instruct   coin_change  0.120   0.000          0.080          25
  ALGO llama-3.1-8b-instruct shortest_path  0.085   0.000          0.055          55
  ALGO llama-3.1-8b-instruct           wis -0.067   0.067          0.033          30
  ALGO llama-3.1-8b-instruct           NaN    NaN     NaN            NaN           0
```

## Proximity × VRI_gap (ALGO adversarial instances)

```
family                 model  n_adversarial  spearman_proximity_vri_gap  p_value  mean_vri_gap
  ALGO       claude-sonnet-4             71                     -0.1509   0.2089        0.0986
  ALGO llama-3.1-8b-instruct             71                     -0.0786   0.5147        0.0423
  ALGO                gpt-4o             71                      0.0637   0.5978        0.2817
  ALGO                pooled            213                     -0.0429   0.5332        0.1408
```

## Highest VRI (rename-specific fragility)

- **BW o4-mini**: VRI=0.451 (W3 retention=0.240)
- **ALGO Claude**: VRI=0.403 (W3 retention=0.250)
- **GSM Llama**: VRI=0.317 (W3 retention=0.187)
- **ALGO GPT-4o**: VRI=0.158 (W3 retention=0.404)
- **ALGO Gemini**: VRI=0.121 (W3 retention=0.549)

## Files

- `vri_by_model.csv`
- `vri_by_subtype.csv`
- `vri_proximity_correlation.csv`

