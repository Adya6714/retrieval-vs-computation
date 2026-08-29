# DS-05 — Machine Source Monitoring (does the model know it memorized?)
family: deeper measurement · cost: ≈ free as an HP-06 add-on

**Idea.** After D1-lite, ground truth exists for what the model saw. Ask the model, per item: "Have you encountered this problem before? Confidence 0–100?" Compute signal-detection sensitivity (d′ / meta-d′-style) of self-reports against true seen/unseen status, and calibration curves.

**Why deeper and distinct.** The "models know what they know" line (Kadavath-style P(IK)) measures knowledge of *correctness*. This measures knowledge of *source* — memory vs derivation — the source-monitoring construct from human memory research, which has never had a ground-truthed LLM test. Bonus dissociation: items the model *solves via retrieval but reports as novel* are exactly the dangerous class for deployment (confidently wrong provenance).

**Design cautions.** Self-report prompts can leak cues (ask before solving, separate session); instruct-tuned models may have trained dispositions to deny memorization — report base vs instruct (the unanalyzed 7B-Instruct sweeps make this comparison free); randomize question framing and report framing sensitivity.

**Output.** One figure: ROC of self-reported familiarity vs true exposure, per model arm. If d′ ≈ 0, models have no privileged access to their own memorization — itself a strong, quotable result for the audit thread ([[BI-02_Predeployment_Audit]]: you cannot just ask the model).
