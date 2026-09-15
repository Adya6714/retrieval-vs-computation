# C1 / DS-02 Intrusion-Error Analysis — Pre-registration

**Frozen before execution.** Date: 2026-09-03.  
Analysis: `scripts/consolidate/c1_intrusion_error_analysis.py`  
Spec: research vault `DS-02_Intrusion_Error_Analysis` / user C1 brief.

## Hypothesis (directional)

Among Probe-1 errors on gold-changing variants (W3 entity rename, W6 parameter regen), a non-zero fraction of wrong answers equal the **canonical** gold (canonical-answer intrusion). Intrusion rate should correlate **positively** with Infini-gram `contamination_score` within (family, model).

## Pre-registered null (publishable)

1. Intrusion rate ≤ explicit chance baseline (see below), and/or
2. Association of intrusion with contamination has CI including 0 (or wrong sign).

A null result is a first-class outcome and must be reported without post-hoc threshold fishing.

## Eligibility (denominator for rates)

Row `(family, model, variant, problem_id)` enters the error set iff **all** hold:

1. Source: rescored Probe-1 CSVs (`results/derived/*_P1_*rescored.csv`), `included=True`, not mock, bank exclusions applied.
2. `variant ∈ {W3, W6}`.
3. Variant gold **differs** from canonical gold under the family structured comparator (see methods note).
4. **Surface-incorrect:** the response’s structured extract does **not** equal the variant gold’s structured extract **without** applying W3 reverse mappings. (Empty / unparseable → surface-incorrect.)

### Amendment (2026-09-03, before primary run)

`rescored_correct` alone is **not** used as the error gate. The ALGO/BW W3 verifiers reverse-map entity/action renames and accept canonical-identifier answers as correct, which would remove the proactive-interference fingerprint from the denominator. Surface-incorrect recovers those cases. Column `rescored_correct` is retained on each error row for audit.

Rows with empty / unparseable responses remain in the denominator as `OTHER_ERROR` (never labelled INTRUSION without a structured match).

## Error classes (mutually exclusive, priority order)

1. **INTRUSION** — structured match to canonical gold (and not to variant gold).
2. **PARTIAL_INTRUSION** — not INTRUSION; canonical gold is a subsequence of the normalized response, **or** overlap with canonical plan steps / solution tokens **> 0.50**.
3. **OTHER_ERROR** — neither.

Exact matchers and the 0.50 threshold are frozen in `C1_INTRUSION_METHODS.md`. No additional fuzzy thresholds.

## Primary estimands

| Estimand | Definition |
|----------|------------|
| `intrusion_rate` | `n_INTRUSION / n_errors` per `(family, model, variant)` |
| `chance_baseline` | Mean, over eligible errors, of the fraction of **other** incorrect responses from the same `(family, model, variant)` (different `problem_id`) that would score INTRUSION against **this** item’s canonical gold |
| `intrusion_vs_contam` | Per `(family, model)`, Spearman ρ between per-instance `is_intrusion` (1 if INTRUSION else 0) and Infini-gram `contamination_score` among eligible errors; cluster-bootstrap CI + p (clone_family; seed 42, B=5000) |

Secondary (reported, not decision criteria): PARTIAL_INTRUSION rates; W3-only and W6-only splits; pooled-within-family correlations.

## Chance baseline (explicit; not assumed zero)

For each eligible error *i* with canonical gold *g_i*:

```
chance_i = (# incorrect answers from same family×model×variant, pid≠pid_i,
            that match g_i under the INTRUSION rule)
           / (# such other incorrect answers)
```

`chance_baseline` = mean of `chance_i` over eligible errors in the cell. If a cell has <2 errors, chance is NA.

## Kill / non-interpretation rules

- Do not pool families (different matchers and gold-change rates).
- Do not treat PARTIAL_INTRUSION as primary evidence of retrieval.
- Do not invent intermediate GSM gold steps; only numbers present in bank gold / response text.
- Stop if raw response text is absent (blocker). Confirmed present: ALGO `model_answer`, BW/GSM `raw_response`.
