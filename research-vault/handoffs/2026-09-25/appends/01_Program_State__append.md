<!-- APPEND to research-vault/RvC-Vault/01_Program_State.md. Section C gets the hypothesis lines; Section D gets the issue lines; Section E gets the changelog entry. Do not rewrite existing lines. -->

## C (append): standing hypotheses added 2026-09-25 via [[THE_PLAN_AMENDMENT_A1]]
- H8 inference-stack noise floor bounds per-instance labels ([[D11_Inference_Stack_Noise_Floor]]). status: untested.
- H9 precision reduction degrades invariance before accuracy ([[D12_Precision_Compression_Invariance]]). status: untested.
- H10 code continued pretraining removes the W4 penalty, not the W3 cost ([[D13_Continued_Pretraining_Transfer]]). status: untested.
- H11 distillation transfers accuracy more than invariance ([[D14_Distillation_Invariance]]). status: untested.
- H12 surface ensembling beats self-consistency at matched calls ([[D16_Surface_Ensembling]]). status: untested.
- H13 isomorph cost exceeds nonce-rename cost ([[EF-07_W3_Construct_Audit]]). status: untested.
- H7b RL form of H7 ([[D15_RLVR_Surface_Diversity]]). status: gated behind G2.

## D (append): open issues found 2026-09-25 (verified-raw)
- **W3 construct mismatch.** Bank W3 is a real-word cover-story / domain isomorph; paper and vault describe a nonce rename. Blocks Paper I resubmission. See [[EF-07_W3_Construct_Audit]].
- **Rename bug.** 12/65 BW W3 rows contain "You are Alice robot arm" (substring replacement).
- **GSM-Symbolic positioning.** Vary Name is names only; our W3 is a domain swap. Direction of the number-vs-name effect relative to GSM-Symbolic must be checked in the primary source (quarantined recollection).
- **F3 prior art.** arXiv:2411.01790 is cited in HP-11 but not in Paper I.
- **Hygiene.** mock rows in derived rescored CSVs; one column-shifted BW row; inconsistent `verifier_function` names; mixed ALGO ID widths breaking CCI joins; `models.yaml` display name for claude-sonnet-4 says Claude 3.7 Sonnet. Full list: `docs/audit/REPO_AUDIT_2026-09-25.md`.
- **Compute reality updated.** Only free Colab T4 available. Phase 1 remains blocked. Track T defined to use the T4.

## E (append): changelog
- 2026-09-25: Repo + site review. Added [[THE_PLAN_AMENDMENT_A1]] (Track T), [[EF-07_W3_Construct_Audit]], [[EF-08_Position_On_Thinking]], directions [[D11_Inference_Stack_Noise_Floor]] to [[D16_Surface_Ensembling]], handoffs [[HP-16_Noise_Floor]] to [[HP-23_W3a_Nonce_Bank]], [[BI-04_Engineering_Relevance]]. Appended D05, D06, P01, P-00. Registered H8 to H13 and H7b. No gate status changed. G0 remains passed; G1 remains blocked.
