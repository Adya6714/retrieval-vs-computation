<!-- APPEND to research-vault/RvC-Vault/02_Decision_Memo.md as a new dated section. -->

## Addendum 2026-09-25: priority order under T4-only compute

Ranking rule: (value to a gated claim or to Paper I) ÷ (cost in available compute), with Paper I fixes first because an instrument-description error is a credibility risk.

1. **T0 W3 construct fix** ([[EF-07_W3_Construct_Audit]]). Required. Option S preferred.
2. **Hygiene fixes** (audit section B). Required before any number is regenerated.
3. **T1 noise floor** ([[D11_Inference_Stack_Noise_Floor]]). Cheapest result that strengthens every per-instance claim; already partly started (`results/raw/llama_determinism_check.csv`, 30 rows, same-config repeats only).
4. **T5 backward-prompting discriminator** ([[D06_Direction_Asymmetry]] append). Under 100 calls; explains F3.
5. **T2 continued-pretraining contrast** ([[D13_Continued_Pretraining_Transfer]]). Controlled open-weight natural experiment.
6. **T3 precision sweep** ([[D12_Precision_Compression_Invariance]]). Same models as T2, so marginal cost is low.
7. **T4 distillation contrast** ([[D14_Distillation_Invariance]]).
8. **T6 surface ensembling** ([[D16_Surface_Ensembling]]).
9. Phase 1 (G1) the moment an A100 is available. Nothing above substitutes for it.

What does not change: Phase 1 remains the priority spend; G1 remains the only route to per-instance labels and exposure claims.
