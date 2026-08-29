# D01 — Controlled-Exposure Instrument Validation
status: Tier 1 · execution: [[HP-06_D1Lite_Finetune_Calibration]] (lite) → full ladder TBD after gate

**Claim being tested.** H1: the convergence labels actually track training exposure. Nothing in the program has ever checked this against ground truth.

**Design ladder (escalate only on signal).**
1. *D1-lite (1 GPU-day).* LoRA-fine-tune Llama-3.1-8B on a random 50% of ALGO canonicals (with CoT solutions); hold out the rest. Rerun full three-probe pipeline on both halves. Ground truth = seen/unseen. Report sensitivity/specificity of retrieval-consistent labels for "seen," computation-consistent for "unseen"; sweep thresholds to produce the ROC — then freeze thresholds at the calibrated point (feeds [[EF-04_Convergence_Labels_MTMM]]).
2. *D1-mid (continued pretraining).* Inject documents containing problem templates at controlled frequencies (0, 10, 10², 10³, 10⁴ occurrences) into continued pretraining of an open 1B model, paraphrase-diversity controlled per Ruis (procedural docs vs answer docs as separate arms — this distinguishes exposure *type*, not just count). Probe suite after; ask whether label strength is monotone in injected frequency.
3. *D1-full (pretraining from scratch).* Small model (160M–1B) on a synthetic corpus where every template's count is known exactly. Gold standard; only if D1-mid shows dose-response and budget exists.

**Why it beats the alternatives.** Every other direction adds signals; this one calibrates them. Reviewers cannot ask "how do you know your label means retrieval" of a labeled ROC curve.

**Frontier (checked 2026-07-07).** Xie et al. K&K ([[P17_Xie_KK_Memorization_2024]]): fine-tune-and-perturb per-sample memorization score — single perturbation-consistency signal, no execution/exposure probes, no instrument-calibration framing. Ruis et al. ([[P18_Ruis_Procedural_2024]]): influence functions show procedural documents (not answer strings) drive reasoning — dictates D1-mid's arm design. Procedural-pretraining modular structures ([[P19_Procedural_Pretraining_2025]]) supports feasibility at small scale. Grokking-based controlled-training studies ([[P30_Grokked_Transformers_2024]], abstract-only) are precedent for synthetic-corpus method, not for probe validation. Open lane confirmed.

**Risks / mitigations.** Small models floor on ALGO → use short DP instances or D1-lite's 8B base. Fine-tuning ≠ pretraining exposure (distribution shift objection) → present D1-lite as calibration lower bound; D1-mid closes the gap. LoRA may memorize differently than full FT → run one full-FT replication arm on 100 items.

evidence: [[P17_Xie_KK_Memorization_2024]] [[P18_Ruis_Procedural_2024]] [[P19_Procedural_Pretraining_2025]] [[P09_Razeghi_2022]] [[P30_Grokked_Transformers_2024]]
