# D17 Commitment depth on a T4 (routing around the failed open-model pilot)

status: Track T, T7 · execution: [[HP-24_Commitment_Depth_T4]] · hypothesis: H14
serves: AC-1 precursor, cheap version of [[D08_Commitment_Depth]] · known blocker this routes around: earlier open-model pilot solved 1/60 and 0/61 renamed items (documented result, not repeatable as-is)

**Claim tested.** Using logit lens or tuned lens on canonical vs W3, find the layer where the answer token first becomes top-1 ("commitment depth"). Prediction: retrieval-like items commit earlier; renaming delays or prevents commitment.

**Why this needs its own direction and not just D08.** D08 sits under Phase 2, gated behind GPU access for the full mechanistic programme. But the layer tool itself (logit lens on a small model's residual stream) runs on a T4. The blocker was never compute, it was model choice: the earlier pilot picked models too weak to solve enough renamed items to have anything to look inside.

**Design.**
- Models: math-tuned Qwen2.5-Math-1.5B-Instruct and Qwen2.5-1.5B-Instruct (not the earlier pilot's models).
- Items: GSM canonical-correct subset only (Acc_can ≥ .30), paired canonical/W3.
- Method: logit lens per layer, per item; record first layer where gold token is top-1.
- Also run invariance depth: per-layer CKA or RSA between canonical and W3 hidden states, find where they reconverge.

**Metrics.** Distribution of commitment depth, canonical vs W3, per model. Correlation between commitment-depth shift and item-level rename survival.

**Primary contrast (pre-register before running).** Median commitment-depth shift (W3 minus canonical) > 0, tested with a paired sign test across items.

**Cost.** T4 hours. No API. Blocked only on GPU time, not on anything structural.
