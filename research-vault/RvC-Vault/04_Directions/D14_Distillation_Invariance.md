# D14 Distillation and invariance transfer

status: Track T, T4 · execution: [[HP-19_Distillation_Contrast]] · hypothesis: H11 · serves: CC-1; AC-3 precursor

**Claim tested.** Distillation from a reasoning teacher transfers canonical accuracy more fully than it transfers surface invariance.

**Existing signal.** F7: o4-mini's intrusion rate under rename on optimisation is .619 against ≤ .25 for other models. Reasoning-trained models carry a distinct fragility signature. R1-distill-70B is already in the roster (planning only).

**Design.**
- Student: DeepSeek-R1-Distill-Qwen-1.5B. Base: Qwen2.5-Math-1.5B (reported as the distillation base; verify in the DeepSeek-R1 report before citing). Third arm: Qwen2.5-Math-1.5B-Instruct (SFT without R1 traces).
- Items: GSM + coin_change, all variants. Long reasoning traces: cap max_new_tokens consistently and report truncation rate per arm.
- Add intrusion-error coding from DS-02 on W3 errors: does the distilled student inherit an intrusion signature like o4-mini's?

**Metrics.** Acc_can, R_W3, R_W4, φ, intrusion rate on W3 errors, per arm.

**Prediction.** Student gains most on canonical; retention gain smaller; intrusion rate rises.

**Training-law follow-up (Phase 3, gated).** Distil the same student on canonical-only traces vs traces spread over W1 to W4 at matched token count: diversity vs dose inside distillation (AC-3).

**Cost.** T4 hours (1.5B at FP16; long generations are the bottleneck).
