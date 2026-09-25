<!-- APPEND to PREREGISTRATION.md as a new top-level section. Commit this BEFORE any Track T run; record the commit hash in each T-report header. -->

## Track T (open-weight, compute-constrained), registered 2026-09-25

Scope. Probe-level results on open-weight models under controlled interventions. No per-instance labels. No exposure claims. Governed by THE_PLAN_AMENDMENT_A1.

Shared rules.
- Retention reported only where Acc_can ≥ .30 on ≥ 20 items for that model × family cell; otherwise the cell is reported as suppressed.
- Intervals: GSM Wilson 95%; ALGO percentile cluster bootstrap, 10,000 draws over near-duplicate families, seed 42.
- Decoding: greedy, fixed max_new_tokens per family, recorded. Stack configuration fixed to the most stable configuration from T1 for T2 to T6.
- No threshold, metric, or item set changes after the first model call of a direction. Deviations are logged as deviations, not silently applied.

T0 (H13), W3a nonce vs W3b isomorph.
- Items: canonical-correct subset (Acc_can ≥ .30 for at least two models).
- Primary contrast: Acc(W3a) − Acc(W3b), per model, paired by item.
- Support for H13: contrast > 0 with interval excluding 0 for at least 3 of 5 API models, or 2 of 3 open models.

T1 (H8), noise floor.
- Reference: batch 1, eager attention, float32.
- Primary metric: verdict flip rate vs reference across the 11 other configurations, per model.
- Kill criterion: flip rate > 10% of items suspends per-instance statements for that model.

T2 (H10), base vs Coder vs Math.
- Primary test: interaction Coder × W4 in a logistic mixed model, correctness ~ model × variant + (1 | item), on ALGO.
- Support: interaction coefficient reduces the W4 penalty relative to Instruct with 95% interval excluding 0.

T3 (H9), precision.
- Primary contrast: ΔR_W3 − (ΔAcc_can / Acc_can) vs fp16, cluster bootstrap.
- Support: interval excludes 0 (invariance loss exceeds relative accuracy loss) for at least 2 of 3 models at NF4 or INT4.

T4 (H11), distillation.
- Primary contrast: (ΔAcc_can) − (ΔR_W3) for distilled student vs base.
- Support: interval excludes 0 in the predicted direction.

T5 (F3 discriminator).
- Primary contrast: (Acc_B − Acc_A) / (Acc_C − Acc_A) per model (fraction of W5 gain recovered by backward instruction).
- Account S favoured if ≥ .5 for at least 2 models; account C favoured if ≤ .2 for at least 2 models and the per-item constraint covariate predicts the W5 gain (logistic, p < .05). Otherwise inconclusive.

T6 (H12), surface ensembling.
- Primary contrast: Acc(surface vote, 5 calls) − Acc(self-consistency, 5 calls).
- Kill criterion: fails on ≥ 2 of 3 models, report null, no rule tuning.
