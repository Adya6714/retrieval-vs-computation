# F. Five-model triangulation under a non-retention rule

Paper default used **retention** (W3/canonical, undefined if canonical=0) and dropped o4-mini as “degenerate at ceiling canonical accuracy”.
This rerun uses **raw W3 correctness (0/1)** on all five models × 110 ALGO problems.

Appendix label rules (three-signal conjunction):
- W3 at 0 (retrieval) or 1 (computation)
- CCI ≤ 0.10 (retrieval) or ≥ 0.67 (computation), per-instance `cci_composite`
- instance contamination at floor (0.000) or ≥ 75th percentile (0.500) of the 110-problem scores

Strong retrieval / computation require **all three** signals aligned. Mixed = at least one retrieval-direction and one computation-direction signal. Ambiguous = remainder (including missing CCI).

| model | n | retrieval | computation | mixed | ambiguous | CCI available |
|---|---:|---:|---:|---:|---:|---:|
| Claude | 110 | 5 | 0 | 72 | 33 | 61 |
| GPT-4o | 110 | 3 | 1 | 73 | 33 | 61 |
| Llama | 110 | 4 | 0 | 78 | 28 | 61 |
| Gemini | 110 | 3 | 0 | 77 | 30 | 61 |
| o4-mini | 110 | 0 | 0 | 57 | 53 | 0 |

**Pooled (5 models):** retrieval=15, computation=1, mixed=357, ambiguous=177.

Paper 4-model legacy counts for comparison: retrieval=8, computation=4, mixed=157, ambiguous=271 (440 instances). Those used the 5-field AND in `ALGO_P3_SCR_triangulation.py` (canonical>0.5, W3<0.2, contamination top-half, greedy_succeeds, plus missing-data → ambiguous), **not** the three thresholds printed in the appendix. Replacing retention with raw W3 on the *printed* three-signal rule does not reproduce 8/4.

**Flags:** o4-mini has **no per-instance CCI** (`ALGO_P2_per_instance_cci.csv` is 4 models; no Phase-1 file to build `cci_composite`). o4-mini rows therefore cannot form a strong conjunction and land in mixed/ambiguous from W3×contamination only. CCI itself exists for only 61 adversarial problems × 4 models, so most of the 110×5 grid is CCI-missing → ambiguous.
