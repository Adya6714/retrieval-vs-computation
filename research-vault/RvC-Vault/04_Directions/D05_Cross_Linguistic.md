# D05 — Cross-Linguistic Invariance
status: Tier 3 (fold into D3) · execution: inside [[HP-10_D3_Distance_Ladders]]

**Framing that works.** Not "multilingual robustness" (crowded: MGSM lineage — [[P29_MGSM_Shi_2022]]) but the far end of D03's distance axis: translation is the most extreme *meaningful* (non-nonce) surface transformation available. Prediction that distinguishes hypotheses: if W3 fragility is entity-identity binding, translated problems with *original entity names kept* should behave like W1; if it is surface-form dependency generally, translation should collapse like W3.

**Design (minimal viable).** 2–3 languages chosen for tokenizer diversity (e.g., Hindi, German, Japanese); professional-quality translation + back-translation verification + family verifier re-check; the 2×2 that matters: {translated, original-language} × {entities kept, entities localized}. ~30 items per family.

**Why not standalone.** Tokenizer confound is severe cross-lingually; without the 2×2 the result is uninterpretable, and with it, it is one clean figure inside the D3 paper rather than a thin standalone.

**Discovery gap.** A dedicated pass on 2025–26 multilingual reasoning-robustness work is still owed before running ([[P-00_Ingestion_Queue]]).

## Update 2026-09-25
- **Dependency on W3 construct.** The "entities kept vs localized" 2 × 2 must now be read against both W3a (nonce) and W3b (isomorph). See [[EF-07_W3_Construct_Audit]]. Without that, a translation effect cannot be attributed to binding vs domain transfer.
- **Language choice.** Hindi is added as the first language: tokenizer-diverse (Devanagari), India-relevant, and absent from per-instance retrieval/computation work as far as the 2026-09-25 scan went (scoop-check still owed, [[P-00_Ingestion_Queue]]).
- **Latent-language link.** Evidence that Llama-family models route through English-like intermediate representations (Wendler et al. 2024, quarantined) gives a prediction: R_translate should be near 1 for models that compute in a shared latent, and should collapse like W3b otherwise.
- **Open-weight execution.** Runs on the [[D13_Continued_Pretraining_Transfer]] models on a T4; no API needed.
