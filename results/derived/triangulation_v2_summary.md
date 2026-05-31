# Triangulation v2 summary

k-of-n vote labels from P1 variants, P2 CCI/injection, P3 contamination/mechanistic.

## Default thresholds (recommended starting point)

- `min_votes=2`, `vote_margin=2`
- contamination high ≥0.6 / low ≤0.4
- CCI computation ≥0.5 / retrieval ≤0.3

## Label distribution (default params)

```
family     tri_v2_label  size
  ALGO      computation   158
  ALGO     insufficient   158
  ALGO   weak_retrieval    80
  ALGO weak_computation    59
  ALGO        retrieval    57
  ALGO            mixed    39
    BW     insufficient   339
    BW   weak_retrieval   110
    BW      computation    78
    BW        retrieval    42
   GSM      computation   106
   GSM        retrieval    61
   GSM            mixed    28
   GSM   weak_retrieval     9
   GSM     insufficient     8
```

## P2A decision normalization

Raw prose↔token match **0.0%**; normalized match **27.5%** (see `deep_p2a_decision_schema_audit.csv`).

## Threshold sweep (best strong-label rate)

- param_id=204: **57.7%** strong labels (retrieval 27.3%, computation 30.4%)
- mixed 0.0%, insufficient 37.9%
- ALGO vs legacy: legacy strong 3.0%, v2 strong 57.6%

