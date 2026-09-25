<!-- APPEND to research-vault/RvC-Vault/04_Directions/D05_Cross_Linguistic.md -->

## Update 2026-09-25
- **Dependency on W3 construct.** The "entities kept vs localized" 2 × 2 must now be read against both W3a (nonce) and W3b (isomorph). See [[EF-07_W3_Construct_Audit]]. Without that, a translation effect cannot be attributed to binding vs domain transfer.
- **Language choice.** Hindi is added as the first language: tokenizer-diverse (Devanagari), India-relevant, and absent from per-instance retrieval/computation work as far as the 2026-09-25 scan went (scoop-check still owed, [[P-00_Ingestion_Queue]]).
- **Latent-language link.** Evidence that Llama-family models route through English-like intermediate representations (Wendler et al. 2024, quarantined) gives a prediction: R_translate should be near 1 for models that compute in a shared latent, and should collapse like W3b otherwise.
- **Open-weight execution.** Runs on the [[D13_Continued_Pretraining_Transfer]] models on a T4; no API needed.
