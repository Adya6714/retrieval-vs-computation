# BX-02 — Transfer & Concept Tests (developmental psychology's arsenal)

## Origin
Piaget built tasks to test whether a child HAS a concept versus merely parrots it: CONSERVATION (quantity is invariant under irrelevant transformation — pour water into a taller glass, is it still the same amount?), REVERSIBILITY, APPEARANCE-REALITY, CLASS INCLUSION. Barnett & Ceci formalized the NEAR/FAR TRANSFER taxonomy (does a skill survive changes in context, modality, format, distance?). Wimmer & Perner / Baron-Cohen built FALSE-BELIEF tasks with graded manipulations. The unifying idea: real concept possession shows INVARIANCE under irrelevant change and TRANSFER across relevant change; memorized responses do not.

## The deep principle
A concept is defined by what it is INVARIANT to and what it GENERALIZES to. Test possession by systematically varying the irrelevant (should not matter) and the relevant (should transfer) and mapping the boundary. This is precisely our retrieval-vs-computation question in the vocabulary of a century of concept research.

## LLM translation
- **Conservation probes:** our W3 rename IS a conservation task (answer invariant under irrelevant relabeling). Generalize to the full Piagetian set: quantity conservation (change units/scale, answer invariant), reversibility (solve forward and backward — ties [[D06_Direction_Asymmetry]]), class inclusion (does the model respect subset structure of the problem space).
- **Near/far transfer gradient:** operationalize Barnett & Ceci's dimensions as measurable axes (context, modality, format, symbolic distance) — this is exactly [[D03_Continuous_Transfer_Distance]], now with a 60-year theoretical backbone and named dimensions instead of ad hoc buckets.
- **Appearance-reality:** does the model distinguish a problem that LOOKS like a known template but is structurally different (MATH-Perturb's "looks easy, isn't")? A direct import of the appearance-reality distinction.

## What it proves for the objective
Grounds the whole "does it understand" question in the ONLY literature that ever rigorously operationalized concept possession for a mind you cannot open. Turns our probes from clever hacks into instances of a validated measurement tradition. Generates [[DS-15_Conservation_Battery]] and strengthens D03.

## Papers to be inspired by
- **BabyReasoningBench (arXiv:2601.18933)** — generates developmentally-inspired reasoning tasks (false-belief transfer, Sally-Anne) for small "baby" LMs, with children's performance curves as reference. Exactly the porting move; study its task construction.
- Wu et al. Counterfactual ([[P02_Wu_Counterfactual_2024]]) — counterfactual-world tasks = Piagetian "reason outside the memorized world."
- ARC / ConceptARC — concept abstraction, though not framed developmentally.
- The Wellman et al. false-belief meta-analysis and Baron-Cohen Sally-Anne for graded-manipulation design.
