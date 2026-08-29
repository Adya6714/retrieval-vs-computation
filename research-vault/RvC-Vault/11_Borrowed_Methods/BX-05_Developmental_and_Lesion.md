# BX-05 — Developmental Order & Lesion Methods

## Origin
DEVELOPMENTAL: Piaget's stages and the CONSISTENT ORDER OF ACQUISITION (children master number conservation before volume, always in that order) reveal the structure of cognition — what must be learned before what. LESION: the oldest neuroscience method — remove a part, see what breaks, infer what the part did. Together they answer WHEN an ability forms and WHERE it lives.

## The deep principle
The ORDER in which abilities emerge, and the parts whose removal breaks them, are structural fingerprints. If surface-robustness reliably emerges AFTER task accuracy across many training runs, that ordering is a law about how reasoning forms. If a specific circuit's removal selectively kills it, that localizes it.

## LLM translation
- **Developmental order (training checkpoints):** across OLMo/Pythia checkpoints, does surface-invariance ALWAYS emerge after raw accuracy? Is there a fixed order of acquisition across problem families (like Piaget's fixed conservation order)? A reliable ordering is a developmental law of LLM reasoning. This is [[D04_Developmental_Checkpoints]], reframed with the developmental-psychology backbone (order-of-acquisition, not just "curves go up").
- **Critical-period probes:** does exposure at different training phases have different lasting effects (variance experiments, [[DS-04_Exposure_Variance_Laws]])? Analog to developmental critical periods.
- **Lesion (ablation):** systematic component ablation mapped to which capability breaks — the lesion half of the double-dissociation program ([[BX-01_Dissociation_Logic]], [[D02_Causal_Patching]]).

## What it proves for the objective
Adds the time axis (when abilities form, in what order) and the localization axis (where they live) to the instrument — turning a static diagnostic into a developmental-plus-anatomical account. Generates [[DS-18_Order_of_Acquisition_Study]].

## Papers to be inspired by
- Pythia / OLMo checkpoint studies; emergent-abilities and grokking literature ([[P30_Grokked_Transformers_2024]]).
- BabyLM challenge and BabyReasoningBench (arXiv:2601.18933) — explicitly developmental LM evaluation.
- Biderman et al. memorization-across-training — order and timing of memorization.
