# EF-03 Probe 3 — Training-Data Exposure

**Question.** Is behavior consistent with training-corpus proximity?

**Design.** Infini-gram n-gram overlap of canonical text with The Pile + DCLM, template vs instance scores, explicitly labeled a public-corpus proximity proxy for closed-weight models. The stronger internal control is structural: CC/SP/WIS all require the same DP technique but differ in public prevalence by orders of magnitude.

**Weaknesses (ranked).**
1. Pile/DCLM is not the training corpus of any tested model; the promised Llama alignment was never delivered.
2. WIS confounds low exposure with high difficulty; Table 5 admits it. Until [[HP-05_Difficulty_Matched_WIS_Bank]] exists, the gradient supports only "consistent with," never "shows."
3. n-gram overlap misses paraphrase-level exposure — exactly the exposure form Ruis et al. find matters most for reasoning (procedural documents, not answer strings; [[P18_Ruis_Procedural_2024]]).

**Upgrade path (where the science moves).**
- Ecological: OLMo + Dolma makes exposure countable in the actual corpus of a real model, at checkpoints ([[D04_Developmental_Checkpoints]]).
- Controlled: inject known exposure by construction and calibrate the probe ([[D01_Controlled_Exposure_Validation]]).
- Attributional: influence-function exposure estimates on open models (heavy; frontier option in D01).

evidence: [[P09_Razeghi_2022]] [[P10_Golchin_2024]] [[P11_Shi_MinK_2024]] [[P18_Ruis_Procedural_2024]] [[P20_OLMo3_2025]]
