# EF-02 Probe 2 — Plan–Execution Coupling

**Question.** Does the model's process match its declaration, and does it recover from corrupted state?

**Design.** Phase 1: full plan declared in session A. Phase 2A: step-by-step execution, one fresh API session per step, no shared context; CCI = fraction of declared numeric steps matched at eps=0.01. Phase 2B: injected wrong intermediate state at a critical step; readouts = compliance taxonomy (compliant / partial / refusal / format-ignored), post-injection accuracy, TEP.

**What makes it distinct.** Hard cross-session isolation. Prior CoT-faithfulness work intervenes within a trace ([[P06_Turpin_2023]], [[P07_Lanham_2023]], [[P08_Tutek_2025]]) or on the model's own CoT ([[P13_vonRecum_2026]]); here the model has no declaration in context at execution time.

**Interpretive limits (be blunt).** CCI confounds reasoning inconsistency with format propensity and parser coverage: 17/29 of GPT-4o's zero-CCI GSM sessions had no parser-extractable declared steps at all. CCI is a fingerprint, not a score — the CAISc draft scopes it this way; keep it scoped. BW strict-PDDL aborts 84–100% of sessions: a measurement-protocol finding, never a per-model CCI claim.

**Verified anchors.** Compliance vs correctness dissociation (acceptance 88–100% for 4/5 models, post-injection accuracy near uninterrupted rates); plausibility null across 5 models; elicitation raises algorithm-naming 10–50x with no accuracy gain across 5 models.

**Extension hooks.** H5: patch at the injection position ([[D02_Causal_Patching]]) to test whether the answer direction pre-exists the injection — the mechanistic resolution of the "two behaviourally indistinguishable mechanisms." Top methodological fix: prompt-matched declaration/execution ablation.
