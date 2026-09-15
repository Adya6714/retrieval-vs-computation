# C7 reconciliation note — variant concordance vs item-level shared hard

**Date.** 2026-09-03  
**Criterion (preregistered, auditable).** A (family, model) ranker is degenerate if
the range of its W1–W6 accuracies is `< 0.1`, or its mean accuracy
`> 0.95`, or its mean accuracy `< 0.1`. Filtering is applied only
after this rule; full-sample W is always reported beside filtered W.

## Degenerate rankers

- `ALGO` / `meta-llama/llama-3.1-8b-instruct`: mean=0.0617, range=0.0893, reasons=`range_lt_0.10|mean_lt_0.10`

ALGO `openai/o4-mini` motivated the check (pairwise ρ −0.09 to −0.31 vs other models; W3 overall 0.963, SP-subset post-fix ≈0.982) but **does not** meet the preregistered criterion (mean=0.7718, range=0.7333). The only ALGO exclusion is Llama (floor + flat).

## Kendall's W (O2 within-row null, 10000 perms, seed=42)

- **ALGO**: full W=0.4194 (p=0.0491, m=5); filtered W=0.5071 (p=0.0495, m=4); excluded=meta-llama/llama-3.1-8b-instruct
- **BW**: full W=0.6786 (p=0.0001, m=6); filtered W=0.6786 (p=0.0001, m=6); excluded=(none)
- **GSM**: full W=0.6603 (p=0.0419, m=3); filtered W=0.6603 (p=0.0419, m=3); excluded=(none)

ALGO's full-sample p≈0.049 sits on the α=0.05 knife-edge. Dropping the only
degenerate ranker (Llama) **raises** W (more concordance among informative
rankers) but leaves p≈0.05 — so the marginal ALGO result is fragile, not a
clear artifact of a ceiling ranker deflating W. o4-mini was never excluded by
the criterion. BW stays decisive with or without filtering.

## Shared-hard item counts (J6 / `P1_failure_patterns.csv`)

- ALGO shared-hard (fail all five paper models): 0/110
- GSM shared-hard (fail all five on paper n=20): 0/20
- BW shared-hard (fail all five): 14/64

## Reconciliation (N4 vs J6)

These two claims are **not** in conflict once the unit of agreement is named:

1. **N4 (variant level).** Within a family, models tend to agree on which
   *surface transforms* are hard (e.g. BW Kendall W=0.6786,
   p=0.0001).
   That is agreement over six variant aggregates, not over items.

2. **J6 (item level).** Fragility does **not** concentrate on a shared hard-item
   core: ALGO 0/110, GSM 0/20,
   BW only 14/64 items fail for all five paper models
   on canonical.

**One sentence for the paper.** Models can share a family's *variant difficulty
ordering* while still failing different *items* under the same transform —
variant-level concordance without item-level shared hardness.

Sources: `P1_variant_ordering_v2.csv` / this script for W; `P1_failure_patterns.csv`
(`shared_hard_canonical`, `fail_all_five_paper_models`) for J6 counts;
`C7_concordance_filtered.csv` for the audit trail.
