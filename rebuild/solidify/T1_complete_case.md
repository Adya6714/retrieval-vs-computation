# T1 — Triangulation complete-case

The executed rule (`rebuild/triangulation_rule.py`) marks an instance **ambiguous** when any of `missing_core`, `parse_failure_or_missing`, or `missing_phase2` is true. Retrieval and computation are assigned only on the complement. Mixed is everything else. That means the published 8 / 4 / 157 / 271 split **confounds signal disagreement with signal absence**.

## Complete-case definition

An instance is complete-case when all three named signals are present **and** W3 actually parsed:

- W3 correctness: `VAR_W3` not NA
- CCI: `ACI` not NA (`missing_phase2` is false)
- proximity: `instance_contamination_score` not NA
- parse succeeded: `parse_failure_or_missing` is false

On the 440-row 4-model panel: parse_failure = 127, missing_phase2 = 196, missing_core = 0.
Three-signal intersection (W3 + CCI + proximity): n = 244.
Of those, 75 still have a parse failure.
**n_complete = 169** after requiring a successful parse — the subset on which the executed rule can assign mixed / retrieval / computation.

Contamination ranks are inherited from the 440-panel (a problem-level property). A reranked sensitivity is reported below.

## Default rule — full panel vs complete-case

Confident-label rate = (n_retrieval + n_computation) / n.

| panel | n | retrieval | computation | mixed | ambiguous | confident-label rate |
|---|---:|---:|---:|---:|---:|---:|
| full 440 (signals never collected **or** disagree) | 440 | 8 | 4 | 157 | 271 | 0.0273 (12/440) |
| three signals present (parse flag still applied) | 244 | 8 | 4 | 157 | 75 | 0.0492 (12/244) |
| complete-case (parse succeeded; remaining mixed = disagreement) | 169 | 8 | 4 | 157 | 0 | 0.0710 (12/169) |

Side by side (headline): **0.0273** on the full panel vs **0.0710** on complete-case.

Retrieval and computation counts do not move (8 and 4). Those labels already required all flags clear. What moves is the denominator: 271 of 440 were never eligible because a signal was missing. On complete-case, ambiguous drops to 0 and the 157 mixed labels are genuine three-signal disagreement.

## 270-config sweep — maximum confident-label rate

Same grid as the rebuild: CCI ∈ {0.05, 0.10, …, 0.90} × W3 cutoff ∈ {0.0, 0.25, 0.5, 0.75, 1.0} × contamination percentile ∈ {50, 75, 90}. Missing-data flags still force ambiguous, so on the full panel they pin n_ambiguous = 271 in every cell. On complete-case that pin is gone; the sweep only reallocates the 169 collected instances among retrieval / computation / mixed.

| panel | n | default confident rate | **max** confident rate over 270 | config at max (CCI, W3, contam pct) | retrieval / computation / mixed / ambiguous at max |
|---|---:|---:|---:|---|---|
| full 440 | 440 | 0.0273 | **0.0523** (23/440) | 0.05, 0.25, 50 | 8 / 15 / 146 / 271 |
| complete-case | 169 | 0.0710 | **0.1361** (23/169) | 0.05, 0.25, 50 | 8 / 15 / 146 / 0 |

Side by side: **max confident-label rate 0.0523** (full panel) vs **0.1361** (complete-case).

Even at the most generous cell of the 270-grid, complete-case confident labels stay a minority. The dominant complete-case label is mixed: the three signals were collected and they disagree.

## Sensitivity: re-rank contamination on the subset

Default rule after re-ranking: retrieval=19, computation=3, mixed=147, ambiguous=0, confident rate=0.1302.
Sweep max after re-ranking: 0.1479 (25/169) at CCI=0.05, W3=0.25, contam pct=50.

Primary numbers use inherited 440-panel ranks.

## Files

- `T1_complete_case_counts.csv`
- `T1_270_sweep_complete_case.csv`
- `T1_270_sweep_complete_case_reranked.csv`
- `T1_panel_flags.csv`

