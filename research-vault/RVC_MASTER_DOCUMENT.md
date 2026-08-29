# RvC Research Program — Master Document
### Retrieval vs Computation: A Measurement Science for How LLMs Actually Solve Problems

This document merges the two documents that together define the full scope of the research vault: **THE PLAN** (the canonical phased execution roadmap) and the **EVALUATION CATALOG, EXTENDED** (the complete menu of evaluation strategies, including the ones reinvented from neuroscience, psychology, and psychometrics). Read together, they answer three questions: what are we building, why, and exactly how — phase by phase, method by method.

The vault itself (110+ linked notes across 12 folders) is the working memory this document summarizes. Folder-by-folder detail lives there; this document is the single file you hand to anyone — a collaborator, a professor, your future self — who needs the whole shape of the program in one sitting.

---

## PART A — THE ROADMAP

*(Source: THE_PLAN.md — the canonical, frozen execution document. If anything elsewhere in the vault conflicts with this section, this section wins until a dated update says otherwise.)*


This is the single canonical execution document. Everything else in the vault supports it. If a conflict exists between this note and any other note, this note wins until a dated changelog entry says otherwise. Deviations are allowed only through a gate decision (below), recorded in [[01_Program_State]].

**Mission.** Build the first validated, per-instance measurement science for how LLMs solve problems (retrieval vs computation), then use it to establish capability claims, an architecture claim, and training-data laws.

**The claims this program exists to establish** (full detail: [[DS-13_Capability_Architecture_Claims]]):
- CC-1 Two-dimensional capability space: accuracy and surface-invariance are dissociable capabilities with different causal antecedents.
- CC-2 Absence of source monitoring: models cannot report whether an answer came from memory.
- CC-3 Developmental order: invariance emerges after accuracy, with measurable lag.
- CC-4 Compositional boundary: a depth k beyond which retrieval cannot substitute for computation.
- AC-1 Strategy is an architectural variable: per-item occupancy of the emergent symbolic pathway (abstraction→induction→retrieval heads) predicts robustness; fragility is pathway bypass.
- AC-2 Installation: invariance can be installed by repairing stage 1 (steering / band-constrained fine-tune) without collateral damage.
- AC-3 Training law: exposure DIVERSITY (not dose) drives invariance because varied surfaces force abstraction-head formation.

**Hypothesis registry.** H1 labels track exposure (D1). H2 W3 collapse mediated by persistent canonical binding (D2). H3 fragility is a dose-response curve with stable cliffs (D3). H4 invariance emerges late, tracks exposure (D4). H5 answers fixed before injection (D2 arm). H6 strategy = symbolic-pathway occupancy (HP-15). H7 diversity forces abstraction-head formation (DS-04 + AC-3). Status lives in [[01_Program_State]] C.

**Non-negotiable rules.** (1) Raw CSVs are truth; derived tables regenerate by script. (2) External numbers are quarantined until reproduced. (3) Pre-register thresholds, metrics, and kill criteria BEFORE runs. (4) Gates decide escalation; no skipping ahead on excitement. (5) Ship narrow papers; the breadth lives here. (6) Reasoning sessions never invent numbers; Cursor never interprets. (7) Every session writes back to [[01_Program_State]]. (8) Log every dollar to costs.csv.

---

## PHASE 0 — Foundations and free frontier results (Weeks 1–2) · cost ≈ $20–80
Do: [[BM-04_Your_First_Week]] days 1–2 (setup) → **HP-13** intrusion errors → **HP-14** mixture-IRT fit → **HP-04** prereg + MTMM + pooled models → **HP-03** mechanistic ledger + base-vs-instruct analysis → **HP-02** draft reconciliation → **HP-01** GSM gap-fill (only paid item).
Read (full text): P17 Xie K&K; P31 Mislevy & Verhelst (model section); P05 MATH-Perturb; P32; P33. Schedule per [[BM-08_Reading_Curriculum]].
Produces: intrusion-rate table (novel), fitted strategy classes (novel), frozen thresholds, clean denominators, MTMM matrix.
**GATE G0 (end W2):** all Phase-0 HP validation checks passed; strategy_posteriors.csv exists; prereg doc committed. Fail → fix before spending on anything else.

## PHASE 1 — Calibration + mechanism flagship (Weeks 3–8) · cost ≈ $120–400 · needs 1× A100-80GB rental
Do: **HP-06** D1-lite (LoRA seen/unseen + corrupted-answer arm) with **DS-05** source-monitoring questions attached (CC-2) → **HP-07** patching (H2, H5) + **HP-08** Commitment Depth in the same GPU sessions → **HP-05** matched WIS bank (API).
Read: P18 Ruis; P21; P23 (before HP-07); P24; P28; P35 first pass.
Claims progressed: CC-2 decided; H1 calibrated; H2/H5 decided; CC-1 evidence strengthened.
**GATE G1 (D1-lite ROC):** AUC ≥ .75 → labels are calibrated, proceed and freeze thresholds v2. .60–.75 → revise indicators (add intrusion, CD) and refit once. < .60 → labels do not track exposure: pivot program framing to mechanism-first (Paper II leads), record decision.
Ship: **Paper I draft** — the instrument: DS-01 + DS-02 + MTMM + D1-lite calibration (venue: NeurIPS/ICML/COLM track or TMLR).

## PHASE 2 — Architecture claim (Weeks 9–14) · cost ≈ $100–400 (more if 70B replication needed)
Do: **HP-15** pathway occupancy (H6/AC-1) → if supported, **AC-2 repair**: steering and band-constrained contrastive fine-tune (scoop-check steering literature first — queue item) → **DS-03** RSA invariance in the same sessions.
Read: P35 deep + LLMSymbMech code; P22 full; Abstractors (queue); circuit tracing (queue); retrieval heads (queue).
**GATE G2:** H6 verdict. Supported → AC-2 experiments. Mixed/heads-don't-transfer → report transfer failure as the finding; AC-1 reframed as "task-type boundary of the symbolic pathway" (still a claim). 
Ship: **Paper II draft** — mechanism/architecture: D2 + D8 + DS-03 + HP-15 (+ AC-2 if landed).

## PHASE 3 — Laws (Weeks 15–22) · cost ≈ $200–700
Do: **HP-10** distance ladders (H3) + **HP-11** direction + **HP-12** sampling pilot → **DS-06** law fitting with out-of-sample prediction → **DS-09** compositional stress (CC-4) → **DS-04** exposure-variance training arms with **AC-3** head-emergence tracking (H7).
Read: P09; P25; P26; Dziri, McCoy (queue); P30 verified.
**GATE G3:** pre-registered functional forms fit vs fail — either is publishable; H7 verdict.
Ship: **Paper III draft** — behavioral + training laws (venue: *ACL/COLM).

## PHASE 4 — Ecology and development (Weeks 23–30) · cost ≈ $80–250
Do: **HP-09** OLMo checkpoint sweep (H4, CC-3) → **DS-07** variance decomposition → **DS-10** cross-model fragility structure.
Read: P20 + OLMo 3 report; Pythia/memorization (queue); P29.
Ship: **Paper IV draft** — the ecology of fragility.

## PHASE 5 — Year 2 summit
**DS-11** inverse inference (behavior → training-data structure of closed models), **DS-08** adaptive instrument, **D10/BI** audit case study with one real deployment task. Prerequisite: Papers I–II accepted or arXived with calibration intact.

---

**Weekly rhythm (always).** Mon: load [[00_MOC]]+[[01_Program_State]]+current phase section, pick tasks. Midweek: execute via HP prompts. Fri: write-back changelog, update claim/hypothesis statuses, 1–2h literature inbox ([[LB-00_Lab_Architecture_Build_Plan]] Layer 2), log costs. Monthly: rerun standing searches in [[P-00_Ingestion_Queue]]; quarterly: update [[LB-03_Evaluation_Strategy_Catalog]] prices.

**Budget assumptions.** Marketplace GPU (A100-80GB ≈ $1.3–2.5/hr) rented per-phase; OpenRouter for sweeps with model tiering. Year-one envelope ≈ $520–1,830 excluding D1-full and human studies. If actual budget is materially lower, the plan degrades gracefully: Phases 0 is free-ish, Phase 1 is the priority spend, Phase 3's DS-04 is the first thing to defer.

**If only 4 hours exist this week:** do the next unchecked HP in the current phase, alone, fully, with write-back. Never start two HPs at once.

---

## PART B — THE EVALUATION CATALOG, EXTENDED

*(Source: EVAL-CATALOG_Extended.md, in the vault's 11_Borrowed_Methods folder — the complete menu of evaluation strategies, grouped by the human-science discipline each one is ported from. This is where the program's deepest novelty lives: not new tasks, but new *inference logic*, borrowed from the fields that spent a century solving our exact problem — inferring hidden process from observable behavior in a mind you cannot open.)*

### Why this section exists
The three probes described in Part A (surface invariance, plan-execution coupling, exposure proximity) are the founding instrument. But if the only strategies in this program were extensions of that instrument, the novelty would stay combinatorial — a better version of what MATCHA, RRB, and PutnamGAP already do. The catalog below is organized instead around the inferential *machinery* of neuroscience, developmental psychology, memory science, psychophysics, and psychometrics — machinery those fields built specifically to diagnose hidden process from behavior, which the "machine psychology" literature has so far borrowed tasks from but not methods. Each entry below names its source method, its LLM translation, and a novelty rating (* incremental, ** strong, *** potentially field-defining / no known prior as of the last search pass).


The original four (validate / grade / open / train-laws) were the shallow layer. Here is the full menu, each tagged with the human-science method it ports and the vault note that develops it. Novelty rating: * = incremental, ** = strong, *** = potentially field-defining / no known prior.

## From dissociation logic (neuropsychology)
- **DS-14 Double-Dissociation Battery*** — find crossover pairs (M1 breaks A not B; M2 breaks B not A) to prove reasoning = separable capabilities. Behavioral + lesion versions. → [[BX-01_Dissociation_Logic]]

## From concept/transfer tests (developmental psych)
- **DS-15 Conservation Battery** ** — full Piagetian invariance set (quantity, reversibility, class-inclusion, appearance-reality) as LLM probes. → [[BX-02_Transfer_and_Concept_Tests]]
- **D3 Continuous Transfer Distance** ** (existing, now backboned by Barnett & Ceci near/far taxonomy). → [[D03_Continuous_Transfer_Distance]]

## From memory science
- **DS-02 Intrusion-Error Analysis*** (proactive interference). → [[DS-02_Intrusion_Error_Analysis]]
- **DS-05 Machine Source Monitoring*** (source vs content). → [[DS-05_Machine_Source_Monitoring]]
- **DS-16 Memory-Signature Suite*** — savings, priming, recognition-vs-recall gap as three more orthogonal signals. → [[BX-03_Memory_Science_Probes]]

## From adaptation/habituation (perceptual neuroscience)
- **DS-17 Adaptation/Repetition-Suppression Probe*** — reveal invariance classes by what the model stops responding to; representational + in-context versions. → [[BX-04_Adaptation_and_Habituation]]
- **DS-03 Representational Invariance (RSA/CKA)** ** (existing; static cousin of DS-17). → [[DS-03_Representational_Invariance_RSA]]

## From developmental order + lesion
- **DS-18 Order-of-Acquisition Study** ** — is there a fixed emergence order of robustness across families/checkpoints (Piaget's fixed order, ported)? → [[BX-05_Developmental_and_Lesion]]
- **D4 Developmental Checkpoints** ** (existing; now framed as order-of-acquisition). → [[D04_Developmental_Checkpoints]]

## From individual differences / psychometrics
- **DS-01 Latent-Strategy Mixture IRT*** — strategy classes from response patterns. → [[DS-01_Latent_Strategy_Measurement_Model]]
- **DS-19 Model-Population Factor Analysis*** — discover the latent dimensions of reasoning-robustness across many models (is there a "robustness g"?). → [[BX-06_Individual_Differences_Psychometrics]]
- **DS-07 Variance Decomposition** ** and **DS-10 Cross-Model Fragility Factor** ** (existing; population view). 

## From mechanism (interpretability, our own bridge)
- **D2 Causal Patching / D8 Commitment Depth / DS-13 Pathway-Occupancy (architecture claim)** — → [[DS-13_Capability_Architecture_Claims]]

## The generative principle (so you can invent more yourself)
For ANY method from a science of hidden minds, ask: (1) what hidden property did it infer? (2) what observable signature did it exploit? (3) what is the LLM analog of that signature? (4) does it give a per-instance, theory-grounded signal our convergent instrument can use? If yes, it is a candidate probe. The six families above are not exhaustive — psychophysics (signal detection theory, staircase thresholds), animal cognition (delayed match-to-sample, transitive inference), and sociology (network diffusion of behavior) each hold more. See [[BX-07_Further_Frontiers]].

---

## PART C — HOW THE TWO PARTS FIT TOGETHER

Part A is the *when* and *in what order*. Part B is the *with what tools*. Mapped together:

- **Phase 0** (foundations) executes DS-01 (mixture IRT) and DS-02 (intrusion errors) from the memory-science and psychometrics families in Part B — both free, both run on data already collected.
- **Phase 1** (calibration + mechanism) executes D01 (ground-truth validation) and D02/D08 (causal patching, Commitment Depth), and is the natural home for DS-05 (source monitoring, from memory science) and the double-dissociation battery DS-14 (from neuropsychology) once dissociating item pairs exist.
- **Phase 2** (architecture claim) executes DS-13/HP-15 (symbolic pathway occupancy), which is the mechanistic backbone that DS-01's dissociation logic and BX-01's lesion methods both feed into.
- **Phase 3** (laws) executes D03 (distance ladders) backboned by BX-02's near/far transfer taxonomy, DS-09 (compositional stress), and DS-04 (exposure-variance training laws) with DS-17 (adaptation/habituation probes) riding alongside as a mechanistic complement.
- **Phase 4** (ecology) executes D04 (developmental checkpoints) reframed by BX-05 as an order-of-acquisition study, plus DS-07/DS-10/DS-19 (variance decomposition, cross-model transfer, population factor analysis) from the individual-differences family in Part B.
- **Phase 5** (Year 2 summit) executes DS-11 (inverse inference) and DS-08 (adaptive diagnosis), both of which require the earlier phases' calibration to be trustworthy inputs.

**The one-sentence summary of the whole program:** build a validated, per-instance instrument for whether an LLM's answer was retrieved or computed, using not just perturbation tests but the full inferential toolkit that neuroscience and psychology built to study minds they could not open; use that instrument to establish capability claims (what abilities are actually separable), an architecture claim (where and how the separation lives inside the model), and training-data laws (what causes robust reasoning to form); and ship the results as narrow, well-controlled papers rather than one sprawling submission, because that is what the venues these results are aimed at actually reward.

**Where to go next.** If you are picking up this document cold: read Part A's rules and Phase 0 first, then skim Part B's family headers to see the full menu, then open the vault itself (00_MOC.md) and follow the links from whichever specific note you need to act on today. The vault's 10_Beginners_Manual folder has a literal first-week walkthrough if you are starting from zero.
