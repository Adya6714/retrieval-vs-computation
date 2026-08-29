# HP-05 — Difficulty-Matched WIS Bank
addresses: [[EF-03_Probe3_Exposure]] weakness #2 · phase: 0 (design) → API run · needs: repo verifiers, modest API budget

PROMPT:
WIS currently confounds low exposure with high difficulty (all models: canonical 1.00 → W3 0.00; but WIS canonicals are also harder). Build a matched bank.
Steps:
1. Define target difficulty band: canonical accuracy 60–70% per model (from existing CC/SP results, pick the band where CC and SP sit).
2. Generate candidate WIS instances varying n(intervals), weight ranges, and overlap density; run the existing WIS verifier; then pilot canonical-only sweeps (cheapest model first) to estimate difficulty; select ~30 instances landing in-band for ≥3 of 5 models. Symmetrically, generate harder CC instances landing in the same band.
3. Run full W1–W6 on the matched sets, all 5 models.
4. Analysis: does the CC-vs-WIS W3-retention gap survive at matched canonical accuracy? Report alongside the IRT exposure coefficient from HP-04.
Output: data/problems/question_bank_wis_matched.csv (+cc_hard), raw CSVs, MATCHED_BANK_REPORT.md.
Validate: matched sets' canonical accuracies must be statistically indistinguishable (report test); every instance passes its verifier; document generation parameters for reproducibility.
