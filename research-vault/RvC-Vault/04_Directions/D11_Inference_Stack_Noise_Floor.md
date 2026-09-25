# D11 Inference-stack noise floor

status: Track T, T1 · execution: [[HP-16_Noise_Floor]] · hypothesis: H8 · amendment: [[THE_PLAN_AMENDMENT_A1]]
evidence: [[P-00_Ingestion_Queue]] (batch-invariance write-up, 2025, quarantined) · prior repo work: `results/raw/llama_determinism_check.csv`

**Claim tested.** At T=0 a per-item verdict can flip when nothing about the model or prompt changes, only the inference stack: batch size, padding, attention backend, accumulation precision. Floating-point addition is not associative, so reduction order changes logits in the last bits, and near-tied tokens can flip.

**Why it matters for this programme.** The mission is per-instance diagnosis. If an item's correctness flips across stack configurations, any per-instance label on that item is partly kernel noise. No retrieval-vs-reasoning or contamination paper reports this floor. The CAISc camera-ready already withdrew a T=0 determinism claim, so the issue is live.

**What exists.** `llama_determinism_check.csv`: 30 rows, 10 items × 3 repeats, one configuration, all identical. That shows repeat stability, not configuration stability. D11 varies the configuration.

**Design.**
- Models: Qwen2.5-1.5B-Instruct (primary), Llama-3.2-1B-Instruct or Qwen2.5-0.5B-Instruct (replication). T4, HF transformers, greedy.
- Items: all GSM variants + ALGO coin_change variants (clears floor for small models).
- Configurations: batch size {1, 8, 16} with left padding; attention `eager` vs `sdpa`; dtype FP16 vs FP32. Full factorial = 12 configurations.
- Outputs per item × config: response hash, parsed answer, verifier verdict.

**Metrics.** Output-divergence rate (hash differs from batch-1 FP32 eager reference); verdict flip rate; flip rate conditional on canonical-correct. Report per variant: are W3/W4 items more flip-prone than canonical? A higher flip rate on transformed items would itself be a finding (transformed items sit nearer decision boundaries).

**Prediction.** Output divergence common (>20% of items); verdict flips rare but non-zero (1 to 5%), concentrated on transformed variants.

**Kill criterion.** Flip rate >10% for a model suspends per-instance statements for that model (see amendment).

**Cost.** A few T4 hours. No API.
