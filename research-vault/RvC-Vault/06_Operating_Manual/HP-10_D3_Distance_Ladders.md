# HP-10 — Continuous Transfer-Distance Ladders
addresses: [[D03_Continuous_Transfer_Distance]] H3, hosts [[D05_Cross_Linguistic]] points · phase: 3 · needs: API budget (~8–12k calls), repo generators

PROMPT:
Pre-register the distance metric BEFORE generating (write docs/D3_DISTANCE_PREREG.md): lexical distance = fraction of entities renamed × nonce grade (synonym=1, rare-word=2, nonce=3); structural distance = parse-graph edit ops; numeric distance = |Δparam| bins. Freeze, then:
1. Select ~40 problems stratified by family × canonical difficulty (all-5-models-correct canonicals preferred).
2. Generate ladders: rename {1 entity, 25%, 50%, 75%, 100%} × grade {synonym, rare, nonce} → 9–12 rungs/item (skip redundant cells); every rung passes the family verifier (answer preservation re-checked per rung — partial renames can create ambiguous referents; verifier failures get regenerated, and attrition reported).
3. Optional D5 points: for 15 GSM items, Hindi and German translations × {entities kept, entities localized} (2×2) with back-translation checks.
4. Sweep 5 models, T=0. Fit per-model decay: segmented regression (cliff) vs logistic (slope); report cliff-location CIs and cross-family consistency; per-model fingerprint figure.
Output: ladder bank CSV, raw results, D3_REPORT.md with decay-curve figures and the prereg doc.
Validate: metric frozen before any generation (git timestamp); monotonicity violations reported not smoothed; per-component curves (lexical vs structural) kept separate.
