# P17 On Memorization of LLMs in Logical Reasoning (Xie et al., 2024, arXiv:2410.23123) — verified-source (abstract+project page, 2026-07-07)
framing: is saturated benchmark performance memorization? Dynamic K&K puzzle generator; fine-tune models on puzzles, then perturb.
perturbs: local math-level (statement/leaf) and language-level (names, role pairs, order, role-flip) perturbations; fine-tuning as controlled "exposure."
measures: per-sample memorization score LiMem = Acc·(1−consistency-ratio); probing internals; cross-difficulty transfer; wrong-answer fine-tuning control.
granularity: per-sample.
uncontrolled: single task family; one consistency signal (no execution or corpus-proximity probes); fine-tuning ≠ pretraining exposure.
bears_on: **[[D01_Controlled_Exposure_Validation]] — the closest published neighbor to D1-lite.** Differentiators to state explicitly: we validate a *multi-probe* label set (not one score) against seen/unseen ground truth, across three families with a built-in prevalence gradient, and connect to mechanistic CD. Also relevant: their finding that fine-tuning improves generalization despite memorization pre-answers part of [[D09_Robustness_Finetune_Transfer]].
practitioner_translation: none shipped; framing only.
new_angle: their wrong-answer fine-tuning control is a clever falsification arm — adopt in D1-lite (fine-tune with corrupted answers; labels should NOT read "computation").
