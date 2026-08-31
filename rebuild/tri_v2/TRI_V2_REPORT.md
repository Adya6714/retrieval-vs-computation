# Triangulation v2 — rule comparison

Executed rule = `label_default()` (asymmetric 5-field AND). Symmetric rule = `label_appendix_three_signal()` (signed votes on W3, CCI bands 0.10/0.67, contamination floor vs p75). Panels: 440-row 4-model, and the 169 complete-case subset from `rebuild/solidify/` (W3 + CCI + proximity present, parse succeeded). Contamination ranks for the executed rule are inherited from the 440 panel.

Sign convention for the symmetric rule: **−1 = retrieval-ward**, **+1 = computation-ward**, **0 = neutral or missing**. Tuple order is `(W3, CCI, proximity)`.

## A. Label counts

Appendix contamination p75 / floor on 440: 0.5 / 0. On 169 (recomputed): 0 / 0.

| panel | rule | retrieval | computation | mixed | ambiguous | n |
|---|---|---:|---:|---:|---:|---:|
| full 440 | executed | 8 | 4 | 157 | 271 | 440 |
| full 440 | appendix symmetric | 8 | 0 | 299 | 133 | 440 |
| complete-case 169 | executed | 8 | 4 | 157 | 0 | 169 |
| complete-case 169 | appendix symmetric | 38 | 0 | 27 | 104 | 169 |
| complete-case 169 | appendix, p75 frozen from 440 | 5 | 0 | 137 | 27 | 169 |

Executed headline: **8 / 4 / 157 / 271 (n=440)** vs **8 / 4 / 157 / 0 (n=169)**.
Symmetric headline: **8 / 0 / 299 / 133 (n=440)** vs **38 / 0 / 27 / 104 (n=169)**.

On complete-case the executed rule has **ambiguous = 0** (missing-data flags are gone). The symmetric rule can still call instances ambiguous when votes are one-sided or neutral (CCI in (0.10, 0.67), contamination between floor and p75).

**p75 collapse.** Recomputing p75 on the 169 subset yields p75 = floor = 0. Every non-negative contamination score is then ≥ p75, so every proximity vote becomes −1 (retrieval-ward). That is why appendix retrieval jumps 8 → 38 and mixed collapses 299 → 27. The frozen-p75 row (5 / 0 / 137 / 27) is the same vote thresholds as the 440 panel, subsetted — use that row to compare rules rather than thresholds.

## B. Symmetric-rule vote contingency

Mixed under the symmetric rule is **defined** as at least one −1 and at least one +1. So every mixed instance is a genuine sign conflict. Neutral/missing votes go to **ambiguous**, not mixed. That is the opposite of the executed rule, whose mixed pile is a residual.

### Full 440 — pattern kinds

| kind | n |
|---|---:|
| unanimous retrieval (−1,−1,−1) | 8 |
| unanimous computation (+1,+1,+1) | 0 |
| conflict, all three nonzero | 61 |
| conflict, one vote neutral (0) | 238 |
| one-sided or neutral (no conflict) | 133 |
| all zeros | 0 |

### Mixed sign patterns (symmetric rule)

| panel | vote (W3, CCI, prox) | kind | n |
|---|---|---|---:|
| full_440 | `(-1,0,1)` | conflict_partial | 223 |
| full_440 | `(-1,-1,1)` | conflict_full | 47 |
| full_440 | `(1,0,-1)` | conflict_partial | 15 |
| full_440 | `(1,-1,1)` | conflict_full | 10 |
| full_440 | `(-1,1,1)` | conflict_full | 3 |
| full_440 | `(1,-1,-1)` | conflict_full | 1 |
| complete_case_169 | `(1,0,-1)` | conflict_partial | 17 |
| complete_case_169 | `(1,-1,-1)` | conflict_full | 7 |
| complete_case_169 | `(-1,1,-1)` | conflict_full | 3 |
| complete_case_169_p75_frozen | `(-1,0,1)` | conflict_partial | 92 |
| complete_case_169_p75_frozen | `(-1,-1,1)` | conflict_full | 33 |
| complete_case_169_p75_frozen | `(1,-1,1)` | conflict_full | 7 |
| complete_case_169_p75_frozen | `(-1,1,1)` | conflict_full | 3 |
| complete_case_169_p75_frozen | `(1,0,-1)` | conflict_partial | 2 |

Full 27-cell occupancy (only cells with n>0): `B_appendix_vote_contingency.csv`.

### Complete-case 169 — pattern kinds (p75 recomputed = 0; proximity vote is always −1)

| kind | n |
|---|---:|
| unanimous retrieval | 38 |
| unanimous computation | 0 |
| conflict, all three nonzero | 10 |
| conflict, one vote neutral | 17 |
| one-sided or neutral | 104 |
| all zeros | 0 |

Frozen-p75 mixed patterns are in the same table under `complete_case_169_p75_frozen`. Dominant 440 mixed cell: **(−1, 0, +1) = 223** — W3 fail vs contamination at floor, CCI silent. That is a two-signal conflict with a missing CCI vote, not a three-way fight.

## C. Executed-rule mixed: which conjunction failed

Executed mixed n = **157** (all 157 sit inside the 169 complete-case subset).

Retrieval needs ALL of: canonical > 0.5, W3 < 0.2, rank > 0.5, greedy_succeeds. Computation needs ALL of: W3 > 0.5, ACI > 0.5, rank ≤ 0.5. Mixed = not ambiguous and neither conjunction.

| condition that failed | n of 157 mixed | fraction |
|---|---:|---:|
| retrieval: canonical ≤ 0.5 | 102 | 0.650 |
| retrieval: W3 not < 0.2 | 20 | 0.127 |
| retrieval: contamination rank not > 0.5 | 146 | 0.930 |
| retrieval: greedy_succeeds is not True | 26 | 0.166 |
| computation: W3 not > 0.5 | 137 | 0.873 |
| computation: ACI not > 0.5 | 115 | 0.732 |
| computation: contamination rank not ≤ 0.5 | 11 | 0.070 |

### Same-direction but still mixed

Three signals = W3, CCI, proximity. Two sign conventions:

- **Executed-native:** W3 < 0.2 → −1, W3 > 0.5 → +1; ACI > 0.5 → +1 else −1; rank > 0.5 → −1 else +1.
- **Appendix votes:** CCI bands 0.10/0.67; contamination floor / p75.

- Mixed with executed-native unanimous retrieval (−1,−1,−1): **8**.
- Mixed with executed-native unanimous computation (+1,+1,+1): **0** (must be 0 — that conjunction *is* executed computation).
- Mixed with appendix unanimous retrieval: **3**.
- Mixed with appendix unanimous computation: **0**.
- Mixed with W3 in the asymmetric gap (0.2, 0.5]: **0** (binary W3 makes this empty).
- Mixed with retrieval-side W3=0 and high contamination, blocked only by extra retrieval fields: n with that side = **9**; failed greedy only = **0**; failed canonical only = **8**; failed both = **1**.
- Mixed that satisfy the full computation conjunction: **0** (must be 0).
- **Structural hole:** W3 = 0 and contamination rank ≤ 0.5: **128** of 157. Retrieval needs high contamination; computation needs W3 = 1. Neither conjunction can fire.

**Same-direction but still mixed: 8 instances** (executed-native all three retrieval-ward). All 8 fail `canonical > 0.5`; 1 of those also fails `greedy_succeeds`. W3-cut asymmetry (0.2 vs 0.5) creates **zero** mixed labels: VAR_W3 is 0/1. The 8 are mixed because of the extra retrieval fields (canonical, and in one case greedy), not because the three named signals disagreed.

The bulk of mixed (128/157) is the structural hole, not greedy and not W3-cut asymmetry.

Failure signatures (which subset of conditions failed): `C_executed_mixed_failure_signatures.csv`.

## D. Sweep after collapsing identical W3 cutoffs

Nominal grid: 18 CCI × 5 W3 × 3 contamination percentiles = **270**.

VAR_W3 is binary. `label_sweep_cell` uses one cutoff on both sides (`W3 < cut` retrieval, `W3 > cut` computation):

- cut = 0.0: retrieval never (W3 < 0 is empty); computation iff W3 = 1.
- cut ∈ {0.25, 0.50, 0.75}: retrieval iff W3 = 0; computation iff W3 = 1. **These three are identical.**
- cut = 1.0: computation never (W3 > 1 is empty); retrieval iff W3 = 0.

**Distinct configurations after collapsing W3: 162** = 18 × 3 × 3.
Empirical check that 0.25/0.50/0.75 produce identical label vectors at every (CCI, contam) cell: mismatches = 0.
Empirically unique label vectors among the 270: 8. Unique count-tuples: 8. Unique label vectors among the collapsed 162: 8.

## E. Instances that change label

- Full 440: **195** / 440 change label (44.3%).
- Complete-case 169, appendix p75 recomputed: **144** / 169 (85.2%). This includes the p75-collapse artefact.
- Complete-case 169, appendix p75 frozen from 440: **30** / 169 (17.8%). Prefer this for rule-vs-rule.

Rows = executed, columns = appendix symmetric.

### Full 440

| executed \ appendix | retrieval | computation | mixed | ambiguous |
|---|---:|---:|---:|---:|
| retrieval | 2 | 0 | 0 | 6 |
| computation | 0 | 0 | 0 | 4 |
| mixed | 3 | 0 | 137 | 17 |
| ambiguous | 3 | 0 | 162 | 106 |

### Complete-case 169

| executed \ appendix | retrieval | computation | mixed | ambiguous |
|---|---:|---:|---:|---:|
| retrieval | 2 | 0 | 0 | 6 |
| computation | 0 | 0 | 4 | 0 |
| mixed | 36 | 0 | 23 | 98 |
| ambiguous | 0 | 0 | 0 | 0 |

### Complete-case 169, appendix p75 frozen from 440

| executed \ appendix | retrieval | computation | mixed | ambiguous |
|---|---:|---:|---:|---:|
| retrieval | 2 | 0 | 0 | 6 |
| computation | 0 | 0 | 0 | 4 |
| mixed | 3 | 0 | 137 | 17 |
| ambiguous | 0 | 0 | 0 | 0 |

Per-instance labels: `E_per_instance_labels.csv`.

## Files

- `A_label_counts.csv`
- `B_appendix_vote_contingency.csv`, `B_appendix_mixed_sign_patterns.csv`
- `C_executed_mixed_failed_conditions.csv`, `C_executed_mixed_failure_signatures.csv`, `C_executed_mixed_same_direction.csv`, `C_executed_mixed_instances.csv`
- `D_sweep_collapse.csv`, `D_sweep_270.csv`
- `E_label_crosstab.csv`, `E_label_transitions.csv`, `E_per_instance_labels.csv`

