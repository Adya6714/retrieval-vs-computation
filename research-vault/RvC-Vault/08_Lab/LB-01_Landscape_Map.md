# LB-01 — Landscape Map: What Exists, Where Novelty Lives

Nine clusters. Status per cluster: saturated / active / underoccupied. Novelty in this field is at intersections, not inside clusters.

**C1. Perturbation & robustness benchmarks (behavioral, group-level).** GSM-Symbolic, GSM-Plus, MATH-Perturb, RUPBench, PutnamGAP, VAR-MATH, functional/multi-instance benchmarks. Status: **saturated** for "models drop under perturbation." Another dataset here is not a contribution.

**C2. Contamination & memorization detection.** Razeghi frequencies, Min-K%, Time-Travel/guided completion, extraction attacks, counterfactual memorization, Xie K&K LiMem. Status: active; per-instance instruments rare; validation against ground truth almost absent.

**C3. CoT faithfulness / process evaluation.** Turpin, Lanham, Tutek unlearning, von Recum interventions, MATCHA. Status: active; within-trace designs dominate; cross-session isolation (your P2) is the wedge.

**C4. Mechanistic interpretability of recall vs computation.** Binding IDs, lookback/rebinding circuits, arithmetic causal mediation, retrieval heads, grokked transformers, circuit tracing. Status: mature methods, **rarely coupled to behavioral instruments on the same items** — that coupling is open.

**C5. Training dynamics / developmental.** Pythia, OLMo 2/3 + Dolma, emergent memorization, grokking, procedural-data pretraining (Ruis; 2505.22308). Status: newly enabled at scale by OLMo 3's checkpoints+corpus; reasoning-robustness developmental curves essentially unwritten.

**C6. Cognitive-science-inspired evaluation ("machine psychology").** CogBench (7 cog-psych paradigms, 10 behavioral metrics, ICML 2024), Binz & Schulz, Hagendorff, developmental-psych proposals, Embers of Autoregression, counterfactual-world evals (Wu). Status: active and growing — but it imports **tasks**, almost never the **measurement theory** behind them.

**C7. Psychometrics for LLMs.** A systematic-review-scale literature now exists (reliability, validity, IRT); IRT applied to benchmark efficiency (TinyBenchmarks/MetaBench), adaptive testing, judge reliability, medical competency modeling. Status: **underoccupied where it matters to you** — IRT-for-LLMs is used to rank models cheaply and stably, not to *diagnose solution strategy*; construct validation (MTMM, criterion validity against exposure ground truth) of a reasoning diagnostic appears unclaimed.

**C8. Transfer, abstraction, analogy.** Emergent-analogies debate, ARC/ConceptARC, counterfactual transfer; human near/far-transfer taxonomies (Barnett & Ceci) essentially unformalized for LLMs. Status: active on tasks, open on **graded distance** (your D3).

**C9. Knowledge organization & editing.** ROME-line editing, consistency (ParaRel), fact geometry. Adjacent; source of one problem statement (conceptual revision vs local patching, LB-02 #7).

## Where your program sits (the moat)
Four intersections, all currently thin:
1. **C2×C5:** per-instance memorization instrument *validated against known exposure* (D1 + D4). Nearest neighbor Xie et al. — one signal, one family, no calibration framing.
2. **C6×C7:** import the *measurement theory*, not just the tasks — MTMM matrices, criterion validity, reliability studies for a reasoning-strategy diagnostic. CogBench phenotypes behavior; nobody certifies an instrument.
3. **C1×C4:** behavioral fragility and causal mechanism on the *same items, same model* (D2+D8 on Llama, closing the loop the field leaves open).
4. **C8×C1:** dose-response transfer curves replacing perturbation buckets (D3), with direction (D6) and language (D5) as points on the same axes.

## The "best in the world" framing (also the practitioner answer)
Do not compete on "another way models fail" (C1 is full). Compete on **measurement science**: the field has thermometers nobody has calibrated. A validated, per-instance, calibrated instrument is what lets a scientist explain models, an engineer monitor training ([[BI-01_Training_Time_Diagnostic]]), and a deployer audit before shipping ([[BI-02_Predeployment_Audit]]). Validation + longitudinal reruns of a frozen bank on every new model release (cheap, compounding, nobody maintains one) = an asset no single paper can scoop.
