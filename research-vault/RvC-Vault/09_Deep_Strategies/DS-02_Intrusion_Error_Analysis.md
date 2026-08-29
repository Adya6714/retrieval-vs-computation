# DS-02 — Intrusion-Error Analysis (read the wrong answers)
family: deeper measurement · cost: $0, existing data · execution: [[HP-13_Intrusion_Error_Analysis]]

**Idea.** Failures carry fingerprints. When a model fails W3 after succeeding on canonical, classify WHAT it answered: (a) **canonical-answer intrusion** — it outputs the canonical item's answer to the renamed problem (proactive interference; direct retrieval evidence); (b) **computational slip** — near-miss consistent with an arithmetic/step error; (c) **procedure intrusion** — applies a memorized technique that no longer fits (MATH-Perturb observed this qualitatively: models memorize problem-solving techniques and blindly apply them — [[P05_MATH_Perturb_2025]]); (d) **degenerate** — refusal, garbage, format break.

**Why deeper.** Every existing signal is about error *rate*; this is error *content* — a channel nobody in the perturbation literature quantifies per instance for reasoning. Memory research has used intrusion errors as the signature of interference for a century. Code-domain work computes output-similarity-under-perturbation ([[P33_Code_Memorization_MRI_2025]]) but no answer-level intrusion taxonomy exists for reasoning banks.

**Predictions.** Intrusion rate should be highest on retrieval-consistent items, correlate with proximity, and (the validation) be elevated on D1-lite SEEN items. If intrusions concentrate where CCI≈0 and proximity is high, three independent methods now converge on content-level evidence.

**First experiment.** Existing raw CSVs: join W3 failures to their canonical rows; exact/numeric-match test for (a); rule-based + manual audit of 100 sampled traces for (b)/(c)/(d); per-model intrusion rates with CIs; feed per-item intrusion as an indicator into DS-01.

**Scoop status (2026-07-07).** Perturbation-sensitivity detection exists (PSH, [[P32_Perturbation_Sensitivity_2025]]); C-BOD does rephrasing-based overfit detection; none analyzes wrong-answer content. Open.
