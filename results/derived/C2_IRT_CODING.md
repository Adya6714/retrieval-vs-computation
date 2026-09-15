# C2 / DS-01 IRT coding

## Person / item definition

- **Person (respondent):** `(model, problem_id)` instance from rescored Probe-1.
- **Item (IRT item):** variant correctness indicators: ['canonical', 'W1', 'W2', 'W3', 'W4', 'W5', 'W6'].
- **Response:** binary `rescored_correct` (1=correct, 0=incorrect), complete-case rows only.

This treats variant surfaces as the measurement instrument for *strategy*, matching
Mislevy & Verhelst (strategy classes with class-specific item parameters). It is
**not** a problem-bank difficulty ranking across `GSM` items; that ranking is
blocked when O10 item variance is near zero (see `C2_identifiability_check.csv`).

## Models

- **1PL / 2PL baseline:** `girth.onepl_mml` / `girth.twopl_mml` on the person×variant matrix.
- **Mixture Rasch (K=2,3,4):** Rost / Mislevy–Verhelst style — class mixing weights `π_c`,
  class-specific Rasch difficulties `b_{c,j}` (sum-to-zero within class), latent ability
  `θ ~ N(0,1)` integrated by Gauss–Hermite quadrature (Q=15).
- **Selection:** minimum BIC among mixture K∈(1, 2, 3, 4) (K=1 is the single-class Rasch null).
  Bootstrap LRT (B=40) for adjacent increments as a second criterion.
  If BIC and LRT disagree, **BIC is primary**; disagreement is reported in `C2_irt_fit_comparison.csv`.
  Do not prefer a larger K for narrative fit.

## Class labels (post-hoc profile rules)

Applied after fit; not used in estimation:

- `w3_collapse`: P(canonical)−P(W3) ≥ 0.20
- `w6_collapse`: P(canonical)−P(W6) ≥ 0.20
- `surface_invariant`: mean(W1,W2,W4) ≥ 0.80 and P(W3) ≥ 0.70
- `low_accuracy`: P(canonical) < 0.40
- else `mixed_other`
