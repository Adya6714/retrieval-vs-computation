# D03 — Continuous Transfer Distance
status: Tier 2 · execution: [[HP-10_D3_Distance_Ladders]]

**Claim tested.** H3: fragility decays with surface distance as a measurable dose-response curve; the cliff location (if any) is a per-model fingerprint stable across families.

**Distance definition (pre-register before generating).** Three orthogonal components per variant: lexical distance (fraction of entities renamed × nonce-ness grade: synonym → rare-word → nonce), structural distance (graph edit distance between problem parses; format changes), and numeric distance (parameter perturbation magnitude). W1–W6 become points in this space rather than unordered buckets.

**Design.** For each of ~40 selected problems (stratified across families and canonical difficulty): generate a graded W3 ladder — rename {1, 25%, 50%, 75%, 100%} of entities at each nonce-ness grade (9–12 points per item). Sweep 5 models. Fit per-model decay curves; test monotonicity, cliff vs slope (segmented regression), cross-family consistency of cliff location. Psychometric payoff: near/far transfer becomes a measured gradient — the Campbell/Piaget framing operationalized, not narrated.

**Why Tier 2 not 1.** Powerful upgrade to Probe 1 and the audit suite ([[D10_Structural_Audit_Suite]] gets a tunable severity knob), but it is still behavioral accumulation — it does not touch label validity. Sequence after D1-lite.

**Frontier (2026-07-07).** Accuracy-vs-perturbation-budget curves are standard in adversarial vision; graded-severity suites exist for VLM corruptions; text-reasoning perturbation work (RUPBench, MATH-Perturb, GSM-Plus) remains categorical. A continuous semantic-distance dose-response for reasoning, per-instance, appears open. D5 (language) and D6 (direction) slot in as far-distance points on the same axes.

**Risks.** Distance metric validity is contestable → pre-register, report per-component curves, and show robustness to metric choice. Partial renames may create ambiguous referents → verifier must re-check answer preservation per rung (already pipeline-standard).

evidence: [[P03_RUPBench_2024]] [[P05_MATH_Perturb_2025]] [[P04_GSM_Plus_2024]] [[P28_Berglund_Reversal_2023]]
