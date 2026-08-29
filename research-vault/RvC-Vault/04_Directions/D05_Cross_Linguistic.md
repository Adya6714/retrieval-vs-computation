# D05 — Cross-Linguistic Invariance
status: Tier 3 (fold into D3) · execution: inside [[HP-10_D3_Distance_Ladders]]

**Framing that works.** Not "multilingual robustness" (crowded: MGSM lineage — [[P29_MGSM_Shi_2022]]) but the far end of D03's distance axis: translation is the most extreme *meaningful* (non-nonce) surface transformation available. Prediction that distinguishes hypotheses: if W3 fragility is entity-identity binding, translated problems with *original entity names kept* should behave like W1; if it is surface-form dependency generally, translation should collapse like W3.

**Design (minimal viable).** 2–3 languages chosen for tokenizer diversity (e.g., Hindi, German, Japanese); professional-quality translation + back-translation verification + family verifier re-check; the 2×2 that matters: {translated, original-language} × {entities kept, entities localized}. ~30 items per family.

**Why not standalone.** Tokenizer confound is severe cross-lingually; without the 2×2 the result is uninterpretable, and with it, it is one clean figure inside the D3 paper rather than a thin standalone.

**Discovery gap.** A dedicated pass on 2025–26 multilingual reasoning-robustness work is still owed before running ([[P-00_Ingestion_Queue]]).
