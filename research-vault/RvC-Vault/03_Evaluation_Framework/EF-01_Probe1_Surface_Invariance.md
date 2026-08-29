# EF-01 Probe 1 — Surface Invariance

**Question.** Does the model's verified final answer survive answer-preserving surface changes?

**Design.** Six variants per canonical: W1 paraphrase, W2 reformat, W3 entity rename to nonce tokens (the diagnostic), W4 formal notation, W5 direction reversal (answer changes; scored as RCS, excluded from CSS), W6 procedural regeneration (new numbers, same algorithm). All variants pass the family verifier before any model call. Zero-shot CoT, T=0.

**Primary readouts.** VAR(variant); W3 retention R_W3 = Acc_W3/Acc_can; CSS over {W1,W2,W3,W4,W6}; VRI = mean(W1,W2,W4) − W3 per the paper's App. C. Note: the repo's `css.py` computes VRI as mean(W2,W4) − W3 — a definition mismatch to reconcile in [[HP-04_Threshold_Prereg_and_MTMM]].

**Assumption this probe alone cannot discharge.** A W3 drop admits tokenization noise, residual difficulty, or parsing artifacts. In place: cross-tokenizer opposite-direction fragility (Claude vs GPT-4o on the same SP/CC items), length-matched nonce vocabulary. Missing: direct tokenizer ablation; graded-distance version ([[D03_Continuous_Transfer_Distance]]).

**Verified anchors.** WIS W3 = 0 for all five models incl. o4-mini at canonical 1.00; SP-adv inversion Claude 0/34 vs GPT-4o 9/34 (Fisher p=0.0021); GSM Claude R_W3=.892 vs o4-mini 1.000 at matched canonical .841.

**Extension hooks.** D3 distance continuum; D6 two-directional probe from W5; D5 language as extreme distance.

evidence: [[P01_GSM_Symbolic_2025]] [[P02_Wu_Counterfactual_2024]] [[P03_RUPBench_2024]] [[P05_MATH_Perturb_2025]] [[P27_VAR_MATH_2025]]
