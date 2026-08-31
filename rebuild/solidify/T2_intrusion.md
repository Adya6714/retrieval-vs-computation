# T2 — DS-02 intrusion errors (paper-ready)

Canonical-answer intrusion: among W3 **errors**, the model’s W3 response encodes the **canonical** gold (pre-rename identifiers) and does **not** encode the W3 gold. Matching is structured (SP path tokens, CC coin multiset, WIS selected set, GSM last number, BW action list), same detector as `rebuild/compute_rebuild.py`.

GSM W3 is name-substitution that preserves the numeric gold, so a W3 error that still equals the canonical number is almost impossible unless verifier and extractor disagree. ALGO W3 relabels nodes/items (0,1,2 → Hub A,B,C); emitting the canonical identifiers on the renamed instance is the intrusion.

## Rates

| model | family | n_W3_errors | n_intrusions | intrusion_rate | Wilson 95% CI |
|---|---|---:|---:|---:|---|
| Claude | ALGO | 100 | 2 | 0.020 | [0.006, 0.070] |
| GPT-4o | ALGO | 91 | 2 | 0.022 | [0.006, 0.077] |
| Llama | ALGO | 108 | 0 | 0.000 | [0.000, 0.034] |
| Gemini | ALGO | 82 | 4 | 0.049 | [0.019, 0.119] |
| o4-mini | ALGO | 43 | 5 | 0.116 | [0.051, 0.245] |
| Claude | GSM | 11 | 0 | 0.000 | [0.000, 0.259] |
| GPT-4o | GSM | 14 | 0 | 0.000 | [0.000, 0.215] |
| Llama | GSM | 17 | 0 | 0.000 | [0.000, 0.184] |
| Gemini | GSM | 21 | 0 | 0.000 | [0.000, 0.155] |
| o4-mini | GSM | 7 | 0 | 0.000 | [0.000, 0.354] |
| Claude | BW | 56 | 0 | 0.000 | [0.000, 0.064] |
| GPT-4o | BW | 54 | 0 | 0.000 | [0.000, 0.066] |
| Llama | BW | 58 | 0 | 0.000 | [0.000, 0.062] |
| Gemini | BW | 58 | 0 | 0.000 | [0.000, 0.062] |
| o4-mini | BW | 53 | 0 | 0.000 | [0.000, 0.068] |

ALGO pooled: **13/424**. GSM pooled: **0/70**. BW pooled: **0/279**.

## Fisher exact: o4-mini vs each other model (ALGO W3 errors)

2×2 of (intrusion, non-intrusion) among W3 errors. Two-sided Fisher. Odds ratio > 1 means o4-mini has a higher intrusion odds than the comparison model.

| comparison | o4-mini | other | OR | Fisher p |
|---|---|---|---:|---:|
| o4-mini vs Claude | 5/43 | 2/100 | 6.45 | 0.026 |
| o4-mini vs GPT-4o | 5/43 | 2/91 | 5.86 | 0.035 |
| o4-mini vs Llama | 5/43 | 0/108 | ∞ | 0.002 |
| o4-mini vs Gemini | 5/43 | 4/82 | 2.57 | 0.273 |

## Verbatim examples (true intrusions only)

Up to five true hits per model. Llama has none. Full W3 traces are in `T2_intrusion_examples.csv`; below: canonical gold, then the span of the W3 response that reproduced it.

### Claude (n=2)

**1. ALGO SP_004**

- Canonical gold: `Path: 0 → 3 → 6 → 4 → 5 → 9, Cost: 8`
- W3 gold: `Path: Hub A → Hub D → Hub G → Hub E → Hub F → Hub J, Cost: 8`
- W3 response (reproducing span): `Path: 0 -> 3 -> 6 -> 4 -> 5 -> 9, Cost:`

**2. ALGO WIS_001**

- Canonical gold: `Selected: {0, 2, 4, 6, 9}, Total: 40`
- W3 gold: `Selected: {Item A, Item C, Item E, Item G, Item J}, Total: 40`
- W3 response (reproducing span): `Selected: {0, 2, 4, 6, 9}, Total: 40`

### GPT-4o (n=2)

**1. ALGO SP_002**

- Canonical gold: `Path: 16 → 12 → 9 → 19 → 18 → 6 → 5 → 10 → 15 → 21 → 11, Cost: 40`
- W3 gold: `Path: Hub Q → Hub M → Hub J → Hub T → Hub S → Hub G → Hub F → Hub K → Hub P → Hub V → Hub L, Cost: 40`
- W3 response (reproducing span): `Path: 16 -> 12 -> 9 -> 19 -> 18 -> 6 -> 5 -> 10 -> 15 -> 21 -> 11, Cost:`

**2. ALGO SP_004**

- Canonical gold: `Path: 0 → 3 → 6 → 4 → 5 → 9, Cost: 8`
- W3 gold: `Path: Hub A → Hub D → Hub G → Hub E → Hub F → Hub J, Cost: 8`
- W3 response (reproducing span): `Path: 0 -> 3 -> 6 -> 4 -> 5 -> 9, Cost:`

### Llama (n=0)

No canonical-answer intrusions.

### Gemini (n=4)

**1. ALGO SP_002**

- Canonical gold: `Path: 16 → 12 → 9 → 19 → 18 → 6 → 5 → 10 → 15 → 21 → 11, Cost: 40`
- W3 gold: `Path: Hub Q → Hub M → Hub J → Hub T → Hub S → Hub G → Hub F → Hub K → Hub P → Hub V → Hub L, Cost: 40`
- W3 response (reproducing span): `Path: 16 -> 12 -> 9 -> 19 -> 18 -> 6 -> 5 -> 10 -> 15 -> 21 -> 11, Cost:`

**2. ALGO SP_004**

- Canonical gold: `Path: 0 → 3 → 6 → 4 → 5 → 9, Cost: 8`
- W3 gold: `Path: Hub A → Hub D → Hub G → Hub E → Hub F → Hub J, Cost: 8`
- W3 response (reproducing span): `Path: 0 -> 3 -> 6 -> 4 -> 5 -> 9, Cost:`

**3. ALGO WIS_001**

- Canonical gold: `Selected: {0, 2, 4, 6, 9}, Total: 40`
- W3 gold: `Selected: {Item A, Item C, Item E, Item G, Item J}, Total: 40`
- W3 response (reproducing span): `Selected: {0, 2, 4, 6, 9}, Total: 40`

**4. ALGO WIS_002**

- Canonical gold: `Selected: {1, 3, 7, 10, 11}, Total: 52`
- W3 gold: `Selected: {Item B, Item D, Item H, Item K, Item L}, Total: 52`
- W3 response (reproducing span): `Selected: {1, 3, 7, 10, 11}, Total: 52`

### o4-mini (n=5)

**1. ALGO WIS_001**

- Canonical gold: `Selected: {0, 2, 4, 6, 9}, Total: 40`
- W3 gold: `Selected: {Item A, Item C, Item E, Item G, Item J}, Total: 40`
- W3 response (reproducing span): `Selected: {0, 2, 4, 6, 9}, Total: 40`

**2. ALGO WIS_002**

- Canonical gold: `Selected: {1, 3, 7, 10, 11}, Total: 52`
- W3 gold: `Selected: {Item B, Item D, Item H, Item K, Item L}, Total: 52`
- W3 response (reproducing span): `Selected: {1, 3, 7, 10, 11}, Total: 52`

**3. ALGO WIS_003**

- Canonical gold: `Selected: {1, 2, 3, 4, 10, 11, 13}, Total: 55`
- W3 gold: `Selected: {Item B, Item C, Item D, Item E, Item K, Item L, Item N}, Total: 55`
- W3 response (reproducing span): `Selected: {1,2,3,4,10,11,13}, Total: 55`

**4. ALGO WIS_004**

- Canonical gold: `Selected: {0, 1, 3, 6, 9, 11, 12}, Total: 37`
- W3 gold: `Selected: {Item A, Item B, Item D, Item G, Item J, Item L, Item M}, Total: 37`
- W3 response (reproducing span): `Selected: {0, 1, 3, 6, 9, 11, 12}, Total: 37`

**5. ALGO WIS_005**

- Canonical gold: `Selected: {2, 5, 7, 8, 11, 14}, Total: 67`
- W3 gold: `Selected: {Item C, Item F, Item H, Item I, Item L, Item O}, Total: 67`
- W3 response (reproducing span): `Selected: {2, 5, 7, 8, 11, 14}, Total: 67`

Unparsed or empty W3 answers are counted as errors and as non-intrusions (never labelled intrusion without a structured canonical match).

## Files

- `T2_intrusion_rates.csv`
- `T2_intrusion_fisher_algo.csv`
- `T2_intrusion_examples.csv` (full W3 traces)
- `T2_intrusion_detail.csv` (every W3 error)

