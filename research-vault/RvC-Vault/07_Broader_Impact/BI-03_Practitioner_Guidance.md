# BI-03 What Practitioners Can Use Today (with caveats)

Usable now, from verified findings:
1. Same-benchmark-score models are not interchangeable under surface change; if your pipeline renames/paraphrases user content, test W3-style variants of YOUR tasks before swapping vendors (SP-adv inversion: Claude 0.647→0.000 vs GPT-4o 0.412→0.265 on identical items).
2. Do not read step-by-step compliance as understanding: models accept injected wrong state 88–100% of the time yet answer as if uninterrupted. Audit outcomes, not narrated obedience.
3. Eliciting a correct strategy description predicts nothing about execution (elicitation null ×5 models). Strategy-level prompt engineering is not a robustness lever.
4. Strict output-format protocols can destroy measurement before the model fails the task (84–100% PDDL aborts) — separate protocol failures from capability failures in internal evals.
Caveat block for all of the above: zero-shot, T=0, three task families, 2026 model versions; per-paper practitioner_translation fields in 05_Papers feed this note as ingestion continues.
