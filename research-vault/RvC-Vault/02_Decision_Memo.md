# Decision Memo — Which Direction, and Why (v1, 2026-07-07)

## The one-sentence diagnosis
The program's real weakness is not coverage or statistics; it is that "retrieval vs computation" has never been checked against anything that is actually retrieval or actually computation. Three correlated proxies agreeing is consistency, not validity. Every direction below is scored primarily on whether it converts the framework from a *consistent* instrument into a *validated* one.

## Scoring axes
1. **Attacks the core weakness** (behavioral-only, uncalibrated labels)?
2. **Validation vs accumulation** — does it calibrate the instrument against ground truth, or just add another correlated signal?
3. **Cost** (API $, GPU, engineering weeks).
4. **Scoop risk** — how close is the nearest published work (see per-direction notes for the frontier check done 2026-07-07)?
5. **Asset reuse** — does it run on the existing 219-problem bank and pipeline?

## Ranked verdicts

**Tier 1 — do these; they define the flagship paper.**

**D1 Controlled-exposure validation** ([[D01_Controlled_Exposure_Validation]]). Highest scientific ceiling in the portfolio: train/fine-tune models where exposure is known *by construction*, run all three probes, report sensitivity/specificity of the convergence labels against ground truth. This is calibrating the thermometer instead of taking more readings. Nearest neighbor is Xie et al.'s K&K fine-tune-and-perturb LiMem score ([[P17_Xie_KK_Memorization_2024]]) — one signal, one task family, no execution or exposure probes; nobody has validated a multi-probe per-instance diagnostic against exposure ground truth. Cost is the catch: full pretraining ladders are expensive. **Therefore start with D1-lite** ([[HP-06_D1Lite_Finetune_Calibration]]): LoRA-fine-tune Llama-3.1-8B on a random half of the ALGO canonicals, hold out the rest, rerun the pipeline, and check whether triangulation labels separate seen from unseen. One GPU-day, and it produces the single most reviewer-proof sentence available to this program: "our labels recover known exposure with X% sensitivity at Y% specificity." Escalate to frequency-ladder continued-pretraining (0, 10¹…10⁵ occurrences, paraphrase-diversity controlled) only after D1-lite shows signal.

**D2 Causal patching** ([[D02_Causal_Patching]]). Directly repairs the two biggest holes at once: (a) upgrades the mechanistic story from correlational to interventional, and (b) fixes the manipulation mismatch — the current pilot contrasts canonical-vs-W6 while the headline is about W3. Design: canonical↔W3 pairs on Llama-3.1-8B (already in the behavioral pool, closing the behavioral↔mechanistic loop the paper defers), patch residual stream at entity-token positions and at late layers where trajectories split; if restoring canonical entity representations into the renamed run restores the answer, rename fragility is a binding failure, not a computation failure. This plugs into a mature literature (binding IDs, lookback/rebinding circuits, arithmetic causal mediation — [[P21_Feng_Binding_2023]], [[P22_Prakash_Lookback_Rebinding]], [[P24_Stolfo_Arithmetic_CMA_2023]]) that has never been aimed at the rename-fragility phenomenon. Also tests H5: patch at the injection step to ask whether the answer direction is fixed before the injected state arrives — resolving the "two indistinguishable mechanisms" the paper currently leaves open. Cheapest information-per-dollar in the portfolio: one A100-class GPU, no API spend.

**D8 Commitment Depth** ([[D08_Commitment_Depth]]). Formalize the construct the pilot already computed informally: earliest layer at which the gold token enters top-k was the strongest single predictor of final rank (r≈+0.82 in the existing sweep). Name it, define it, replicate on Llama, and it becomes both the mechanistic dependent variable for D2 and the training-time diagnostic for the industry thread. Near-zero marginal cost; ride-along on D2.

**Tier 2 — high value, run in parallel where budget allows.**

**D4 Developmental checkpoints** ([[D04_Developmental_Checkpoints]]). OLMo 3 (Nov 2025) released full intermediate checkpoints at every stage plus the entire Dolma 3 corpus; OLMo 2 has checkpoints across ~4T-token runs at 1B/7B/13B. This is the only direction that makes Probe 3 *non-proxy on a real model*: count template occurrences in the actual training data, then track W3 retention and Commitment Depth across checkpoints. D1 gives controlled ground truth; D4 gives ecological ground truth. Inference-heavy but no training. Main risk: early checkpoints floor on ALGO tasks → use retention ratios and easy instances.

**D3 Continuous transfer distance** ([[D03_Continuous_Transfer_Distance]]). Replace six buckets with a dose-response curve: define lexical/structural/format distance components, generate graded W3 ladders (rename 10%…100% of entities; nonce-ness graded from synonyms → rare words → nonce tokens), and test monotone decay vs cliff, and whether the cliff location is a stable model fingerprint. Robustness-vs-perturbation-budget curves are standard in adversarial vision; the semantic analog for reasoning text is open (RUPBench/MATH-Perturb use discrete categories). This is also the psychometric payoff: near/far transfer becomes a measured gradient, not a metaphor. Moderate API cost; runs on the existing bank.

**Tier 3 — fold in, don't lead with.**

**D6 Direction asymmetry** ([[D06_Direction_Asymmetry]]). Real phenomenon, partially occupied ground: forward/backward planning asymmetry tracking search complexity is published ([[P25_Forward_Backward_Planning_2024]]), and code-execution invertibility is active. The open contribution is the *interaction*: same problem graph, both directions, then rename — if asymmetry collapses under W3, direction-specific surface templates are implicated. Run as a probe extension (P1.5) inside D3's design, not standalone.

**D5 Cross-lingual** ([[D05_Cross_Linguistic]]). Cheap but crowded (MGSM lineage) and tokenizer-confounded. Correct framing: the far end of D3's distance axis, 2–3 languages chosen for tokenizer diversity, not a standalone paper.

**D7 Sampling-consistency probe** ([[D07_Sampling_Consistency_Probe]]). New, cheap, deployable: at T>0, answer-distribution entropy for canonical vs W3 as a fourth orthogonal signal (sharp canonical basin + diffuse W3 basin ⇒ retrieval-like). Pilot on 30 instances before committing; if it correlates too strongly with CSS it adds nothing.

**D9 Robustness fine-tuning transfer** ([[D09_Robustness_Finetune_Transfer]]). Practitioner-relevant (can rename-robustness be trained in, and does it transfer across families?) but it's an intervention study that presupposes the instrument — schedule after D1-lite.

**D10 Structural audit suite** ([[D10_Structural_Audit_Suite]]). The industry translation. Explicitly speculative until D1 calibration exists; do not pitch it as validated before then. Lives in 07_Broader_Impact.

## Program architecture (recommended)
- **Phase 0 (now, ~2 weeks):** HP-01 GSM gap-fill + canonical numbers, HP-02 draft reconciliation, HP-03 mechanistic ledger, HP-04 threshold pre-registration + MTMM analysis, HP-05 matched-WIS bank. Nothing in Tier 1 should ship on top of unhygienic denominators.
- **Phase 1 (flagship):** HP-06 D1-lite → decision gate → D1-full design; HP-07 D2 patching + HP-08 Commitment Depth in parallel (same GPU, same model).
- **Phase 2:** HP-09 OLMo developmental sweep (D4).
- **Phase 3 (behavioral paper):** HP-10 distance ladders (D3) with HP-11 direction probe (D6) and cross-lingual points (D5); HP-12 sampling-consistency pilot (D7) rides along.
- **Publication logic (decide later, per the brief):** natural cleavage is Paper A = instrument validation + causal mechanism (D1+D2+D8+D4) for an ML venue; Paper B = behavioral geometry of fragility (D3+D5+D6+D7) for an *ACL venue. The CAISc submission is the foundation both cite.

## Why this ordering beats the alternatives
- Leading with D3/D5/D6 (more behavioral data) is the accumulation trap: reviewers will still ask "but how do you know your labels mean retrieval?" and more correlational signals cannot answer that.
- Leading with D1-full (training ladders) before D1-lite risks burning the compute budget before knowing whether labels track even the crudest ground truth.
- Skipping Phase 0 puts the next paper on top of a table that just failed audit once (A1); reviewers who reproduce from raws will find it.
- D2 without D8 produces flips without a reusable metric; D8 without D2 produces a metric without causal warrant. Together they are one workstream.

## Addendum (2026-07-07): Deep strategies and the "ship narrow, build broad" doctrine
The DS series ([[DS-00_Overview]]) adds moves that change the kind of science, not just the number of experiments. Two of them cost nothing and should slot into Phase 0 immediately because they also strengthen everything downstream:
- **[[DS-02_Intrusion_Error_Analysis]]** (HP-13): read wrong-answer content as a retrieval fingerprint. $0, existing data.
- **[[DS-01_Latent_Strategy_Measurement_Model]]** (HP-14): replace hand-set convergence labels with a fitted mixture-IRT model; dissolves the threshold-sensitivity weakness and makes D1-lite validate *classes* instead of thresholds. $0, existing data.
Then **[[DS-03_Representational_Invariance_RSA]]** joins the D2 GPU sessions (mechanistic signal that works below the behavioral floor), and **[[DS-04_Exposure_Variance_Laws]]** becomes the flagship causal experiment after D1-lite (variety-not-volume as a data-curation law).

Publication doctrine (see [[BM-06_Publishing_And_Conferences]] and [[DS-12_Grand_Synthesis]]): build the whole program in the vault, but ship NARROW papers. A single unified mega-paper is a weak paper — reviewers reward one load-bearing, well-controlled, surprising claim with objections pre-closed. The natural decomposition is five narrow papers (instrument / mechanism / laws / ecology / inverse); decide boundaries from evidence, not up front.
