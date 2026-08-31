# T3 — DS-14 double dissociation (formal)

Source: `rebuild/p1_pairwise_inversion.csv`, recomputed at item level from the same frozen ALGO adversarial IDs and P1 loaders as the rebuild. **Definition = canonically-matched**: the ID must be in the frozen subtype list and both models must be canonical-correct. Effect size is W3 accuracy of model A minus model B; 95% CI is a paired bootstrap (10,000), same as the rebuild.

A **genuine crossover** is a model pair with significant effects in **opposite directions** on two different subtypes. Single dissociations (one subtype significant, the other not) can be a difficulty artefact; a crossover cannot.

## Strict vs suggestive

**Strict crossover** (all of):

1. Canonically-matched definition.
2. Two different subtypes.
3. Opposite signs of Δ (acc_A − acc_B).
4. Fisher exact two-sided p < 0.05 on **both** subtypes.
5. Bootstrap 95% CI of Δ excludes 0 on **both** subtypes.
6. Combined 2×2×2 interaction is significant: Breslow–Day p < 0.05 **or** log-linear three-way G² p < 0.05 **or** bootstrap 95% CI of the difference-of-differences excludes 0.

The 2×2×2 table is model (A, B) × subtype (S1, S2) × W3 outcome (correct, incorrect). Breslow–Day tests homogeneity of the two subtype odds ratios. The Poisson log-linear G² tests the three-way term against the all-two-way model (1 df). Zero cells get Haldane–Anscombe +0.5 before the log-linear fit. Difference-of-differences = Δ_S1 − Δ_S2, items resampled independently within subtype, pairing within subtype preserved.

**Suggestive**: opposite-signed Δ on two subtypes, but (4)–(6) not all met (one Fisher non-significant, a CI covering 0, or the interaction not significant).

α = 0.05, uncorrected. 10 model pairs × 3 subtype-pair slots = 30 implicit tests.

Opposite-signed subtype pairs found: **1**. Strict: **1**. Suggestive: **0**.

## Pairs that meet the strict crossover criterion

### Claude vs GPT-4o — SP × CC

- SP (n=11): Claude 0/11 vs GPT-4o 5/11, Δ = -0.455 [-0.727, -0.182], Fisher p = 0.035, OR = 0.00
- CC (n=5): Claude 4/5 vs GPT-4o 0/5, Δ = 0.800 [0.400, 1.000], Fisher p = 0.048, OR = ∞
- Combined 2×2×2: Breslow–Day = 13.05, p = 0.0003; log-linear G² = 12.26, p = 0.0005 (Haldane-Anscombe +0.5 continuity (zero cell)); ΔΔ = -1.255 [-1.636, -0.764].

**Verdict: strict crossover.** Relative to GPT-4o, Claude is worse on SP and better on CC; both simple effects and the interaction survive the criterion. Caveat: canonically-matched n on CC is small (matched IDs require both models canonical-correct).

## Pairs that are only suggestive

No other model pair has opposite-signed canonically-matched Δ on two subtypes.

## Pairs with no opposite-signed subtype pair

Every other canonically-matched pair is either same-sign across subtypes, zero on the second subtype, or missing a matched ID intersection. Those are single dissociations or nulls, not crossovers. Full cell table: `T3_canonically_matched_cells.csv`.

## Files

- `T3_crossover.csv`
- `T3_canonically_matched_cells.csv`

