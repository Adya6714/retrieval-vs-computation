# BM-08 — Reading Curriculum (what to read, in what order, and what to extract)

Rule: read with the [[T_Paper_Note]] schema open; a paper is "read" when its note is upgraded to verified-source and its bears_on links are updated. Never read passively.

## Tier 1 — before the end of Phase 1 (these shape your own experiments)
1. **P17 Xie et al., Memorization in Logical Reasoning (2410.23123).** Extract: their fine-tune-and-perturb design, LiMem definition, and the corrupted-answer control you are copying in HP-06. Your differentiation sentence lives or dies on knowing this paper cold.
2. **P31 Mislevy & Verhelst 1990 (mixture IRT).** Extract: the model specification and identifiability conditions; skim the math, master the logic. This is the theory under HP-14.
3. **P35 Yang et al., Emergent Symbolic Mechanisms (2502.20332) + their code repo.** Extract: how heads were identified (causal mediation + permutation tests), exact head lists, task format. Required before HP-15.
4. **P23 Zhang & Nanda, patching best practices.** Extract: the metric and corruption-choice checklist; it is the methodology contract for HP-07.
5. **P18 Ruis et al., Procedural Knowledge (2411.12580).** Extract: procedural-vs-answer-document finding; it dictates DS-04's arms and your exposure story.
6. **P05 MATH-Perturb.** Extract: their simple-vs-hard perturbation split and the "memorized technique blindly applied" observation your DS-02 taxonomy formalizes.

## Tier 2 — before Phase 2 (mechanism literacy)
7. **P21 Feng & Steinhardt, binding IDs.** 8. **P22 lookback + rebinding circuit (2606.08644, full text).** 9. **P24 Stolfo, arithmetic causal mediation.** 10. **Anthropic circuit tracing / attribution graphs (queue).** 11. **Retrieval heads (2404.15574, queue).** Extract from all: where to patch, what counts as evidence of mediation, standard pitfalls.

## Tier 3 — before Phases 3–4 (laws + development)
12. **P09 Razeghi (frequencies).** 13. **P25 Thinking Forward/Backward.** 14. **P20 OLMo 3 report** (read the data-mix and checkpoint sections only). 15. **P30 Grokked Transformers (verify details).** 16. **Dziri Faith & Fate; McCoy Embers (queue).** 17. **P29 MGSM** if D5 activates.

## Method primers (read once, early, ~1 evening each)
- An activation-patching tutorial (e.g., ARENA/TransformerLens walkthrough) before any GPU work.
- The mirt (R) or StepMix vignette before HP-14.
- Campbell & Fiske 1959 (MTMM original) — short, foundational, makes EF-04 obvious.
- Any IRT primer chapter — difficulty/ability separation.

## How to build on papers (the move, every time)
For each paper ask, in order: what did they measure → what could they NOT distinguish → which of our probes/claims resolves that → does their method port to our bank? Write the answer into the paper note's new_angle field. That field is where your next experiments come from.

## Standing monthly searches (from [[P-00_Ingestion_Queue]])
per-instance memorization reasoning · activation patching reasoning robustness · training checkpoints emergence reasoning · benchmark perturbation distance · steering/representation-engineering invariance (pre-AC-2) · mixture IRT LLM (defend DS-01 priority) · symbolic mechanisms LLM (track the P35 line — your AC-1 depends on staying current here).
