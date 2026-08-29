# BX-01 — Dissociation Logic (the crown jewel of neuropsychology)

## Origin
Teuber 1955 introduced DOUBLE DISSOCIATION; Shallice, Coltheart, Caramazza built cognitive neuropsychology on it. The logic: if lesion A breaks function X but spares Y, and lesion B breaks Y but spares X, then X and Y are SEPARATE systems, not one graded ability. It is the gold standard for inferring that mental functions are architecturally independent, and it works by inference-to-the-best-explanation, not by looking inside. (Verified: multiple sources confirm it as the cornerstone method for specifying cognitive architecture.)

## The deep principle
Correlation of abilities is cheap; SEPARABILITY is the real structural claim. You establish separability by finding a manipulation that moves one ability while leaving the other untouched, AND a second manipulation that does the reverse. A single dissociation could be a difficulty artifact (X is just harder). The DOUBLE dissociation kills that: difficulty cannot explain a crossover.

## LLM translation (this is genuinely novel for our field)
Our program already stumbled onto a single dissociation without naming it: same benchmark accuracy, different rename-fragility (Claude vs o4-mini). Formalize and hunt DOUBLE dissociations:
- **Behavioral double dissociation across manipulations:** find manipulation M1 (e.g., entity rename) that breaks model/item class A but not B, and M2 (e.g., compositional depth increase) that breaks B but not A. That proves surface-robustness and compositional-robustness are SEPARATE capabilities, not one "reasoning" scalar. This directly attacks the field's habit of collapsing everything into one reasoning score.
- **Lesion double dissociation (open models):** ablate circuit component P → rename-fragility appears but arithmetic intact; ablate Q → arithmetic breaks but rename-robustness intact. That localizes and separates the mechanisms. Ties to [[D02_Causal_Patching]] and the symbolic-circuit work ([[P35_Emergent_Symbolic_Mechanisms_2025]]).
- **Task-impurity control:** neuropsychology's known caveat. Match tasks on everything but the target ability (our difficulty-matched banks, [[HP-05_Difficulty_Matched_WIS_Bank]], are exactly this instinct).

## What it proves for the objective
Converts "models behave differently" into "reasoning is not one thing; it is N separable capabilities with N separable mechanisms, and here is the crossover evidence." That is an architecture-of-cognition claim, the kind that defines a research line. Generates [[DS-14_Double_Dissociation_Battery]].

## Papers to be inspired by / differentiate from
- Cognitive neuropsychology canon (Shallice, "From Neuropsychology to Mental Structure"; Coltheart dual-route reading model built entirely on dissociations).
- ML side is nearly empty: some interpretability work shows circuit ablations, but almost nobody frames LLM capability claims as formal double dissociations. Confirm with a fresh search before drafting; this looks like open ground.
