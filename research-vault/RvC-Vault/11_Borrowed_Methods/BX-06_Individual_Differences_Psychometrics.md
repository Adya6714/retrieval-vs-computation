# BX-06 — Individual Differences & Psychometrics (the population view)

## Origin
Spearman invented FACTOR ANALYSIS to discover that many cognitive tests share a latent "g" factor. The whole individual-differences tradition asks: across a POPULATION of test-takers, what latent DIMENSIONS explain the pattern of who succeeds where? Item Response Theory (IRT) separates item difficulty from person ability. Campbell & Fiske's MTMM validates that a measure captures its intended trait and not method noise.

## The deep principle
You discover the true structure of an ability not from one subject's average, but from the CORRELATIONAL STRUCTURE across many subjects and many items. Latent factors, not raw scores, are the real objects of study.

## LLM translation
- **Model-population factor analysis:** treat each MODEL as a subject and each probe-item as a test; factor-analyze the big model×item performance matrix. Is there a single "reasoning-robustness g" that models load on, or several orthogonal factors (surface-robustness, compositional-robustness, source-awareness)? This is the population-level version of the double-dissociation claim, and it directly extends [[DS-07_Variance_Decomposition]] and [[DS-10_Cross_Model_Transfer_Attack]].
- **Mixture IRT for strategy classes:** [[DS-01_Latent_Strategy_Measurement_Model]] — latent classes = different solution strategies, straight from Mislevy & Verhelst ([[P31_Mislevy_Verhelst_1990]]).
- **MTMM validation:** [[EF-04_Convergence_Labels_MTMM]] — prove the probes converge on strategy and are not just re-measuring difficulty.
- **Cross-model fragility factor:** are the item-cliffs shared across models (one latent difficulty-by-surface dimension) or idiosyncratic? ([[DS-10_Cross_Model_Transfer_Attack]].)

## What it proves for the objective
Elevates the claim from "this model does X" to "the latent structure of reasoning-robustness across the model population has K dimensions, and here they are." A field-defining, population-level result. Generates [[DS-19_Model_Population_Factor_Analysis]].

## Papers to be inspired by
- Ye et al. LLM Psychometrics systematic review (arXiv:2505.08245) — the umbrella; note it calls for exactly IRT/validity adoption but mostly on personality/ability, not strategy diagnosis. Your differentiation: strategy, not trait.
- TinyBenchmarks / metabench (IRT for efficient eval) — IRT used for compression, not strategy; contrast.
- Hernández-Orallo "Universal Psychometrics" — the manifesto for measuring machine cognition; cite as framing.
- CogBench (7 cog-psych paradigms across 35 LLMs) — population behavioral phenotyping; you add measurement theory on top.
