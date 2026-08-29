# Retrieval vs Computation — Research Program MOC

Open this note first. It is the map of the whole vault and the current state of the program.

**Core question:** when two LLMs score the same on a benchmark, are they solving problems the same way, or is one computing and the other pattern-matching? The program builds a validated, convergent, multi-signal instrument (behavioral + mechanistic) that distinguishes them per instance.

**Status (2026-07-07):** CAISc 2026 submission complete ("Same Score, Different Strategy"). Three data-hygiene issues resolved against raw CSVs in this pass (see [[01_Program_State]]). Next phase: instrument validation + causal mechanism.

## Read in this order
0. **[[THE_PLAN]] — the single canonical execution document. If you read one note, read this.**
1. [[01_Program_State]] — what is confirmed, hypothesis, open. The running truth ledger.
2. [[02_Decision_Memo]] — ranked comparison of all candidate directions and the recommended program architecture.
3. [[06_Operating_Manual/OM-00_How_To_Use]] — how to hand work to Cursor / other Claude sessions.

## Vault sections
- **03_Evaluation_Framework/** — what the instrument is. One note per probe ([[EF-01_Probe1_Surface_Invariance]], [[EF-02_Probe2_Plan_Execution]], [[EF-03_Probe3_Exposure]]), the labeling logic ([[EF-04_Convergence_Labels_MTMM]]), metric definitions ([[EF-05_Metrics_Glossary]]), and open methodological questions ([[EF-06_Open_Methodological_Questions]]).
- **04_Directions/** — one note per candidate direction, D01–D10. Each links to every paper that bears on it and to the handoff prompt that executes it.
- **05_Papers/** — one note per ingested paper, populated with the standard schema (see [[99_Templates/T_Paper_Note]]). [[P-00_Ingestion_Queue]] tracks papers found but not yet fully ingested.
- **06_Operating_Manual/** — self-contained handoff prompts HP-01…HP-12, each executable cold by another agent.
- **07_Broader_Impact/** — the industry/practitioner thread, tracked live, with validation status flags.
- **11_Borrowed_Methods/** — the deepest source of novelty: the inference methods of neuroscience, developmental psychology, memory science, psychophysics, and psychometrics, each reinvented as an LLM evaluation. Start at [[BX-00_The_Deep_Logic]]; the full menu is [[EVAL-CATALOG_Extended]].
- **10_Beginners_Manual/** — START HERE if you are new: [[BM-00_Start_Here]], a plain-English [[BM-01_Glossary_For_Beginners]], how the lab runs day-to-day ([[BM-02_How_To_Run_The_Lab]]), thinking checklists ([[BM-03_Thinking_Checklists]]), a literal [[BM-04_Your_First_Week]], tooling+costs ([[BM-05_Tooling_And_Costs]]), publishing+conferences ([[BM-06_Publishing_And_Conferences]]), and the deep strategies in plain words ([[BM-07_Deep_Strategies_Explained_Simply]]).
- **09_Deep_Strategies/** — the deeper, more novel moves beyond perturb-and-observe: latent-strategy measurement ([[DS-01_Latent_Strategy_Measurement_Model]]), intrusion errors ([[DS-02_Intrusion_Error_Analysis]]), representation geometry ([[DS-03_Representational_Invariance_RSA]]), exposure-variance laws ([[DS-04_Exposure_Variance_Laws]]), source monitoring ([[DS-05_Machine_Source_Monitoring]]), predictive fragility laws ([[DS-06_Fragility_Laws_Prediction]]), variance decomposition ([[DS-07_Variance_Decomposition]]), adaptive diagnosis ([[DS-08_Adaptive_Diagnosis]]), compositional stress ([[DS-09_Compositional_Stress]]), cross-model transfer ([[DS-10_Cross_Model_Transfer_Attack]]), inverse data attribution ([[DS-11_Data_Attribution_Inverse]]), the [[DS-12_Grand_Synthesis]], and the escalation path from protocol to capability/architecture claims ([[DS-13_Capability_Architecture_Claims]]).
- **08_Lab/** — the lab itself: [[LB-00_Lab_Architecture_Build_Plan]] (how to build and run the research brain), [[LB-01_Landscape_Map]] (what exists, where novelty lives), [[LB-02_Problem_Statement_Engine]] (how to generate new problem statements), [[LB-03_Evaluation_Strategy_Catalog]] (every strategy with requirements and costs).
- **99_Templates/** — templates for every note type. Use them; consistency is what makes the vault queryable.

## Linking conventions (enforce these)
- Every paper note lists `bears_on:` with [[D-links]]; every direction note lists `evidence:` with [[P-links]] back. Both must exist — the graph is bidirectional or it is useless.
- Every handoff prompt lists `addresses:` with the direction or gap it exists for, and every direction note lists `execution:` with its HP link(s). Never delegate a task whose "why" is not written down.
- Program-state changes (a hypothesis confirmed, a number corrected) get a dated entry in [[01_Program_State]] and, if they change priorities, in [[02_Decision_Memo]].
- Claims from external sources carry a status tag: `verified-raw` (checked against our CSVs), `verified-source` (read in the primary paper), `abstract-only` (from abstract/secondary coverage — do not build on it yet), `unverified-external` (from a handed-in document — quarantine).

## Current top priorities (mirror of Decision Memo)
1. Phase 0 data hygiene: [[HP-01_GSM_GapFill_and_Canonical_Numbers]], [[HP-03_Mechanistic_Provenance_Ledger]], [[HP-04_Threshold_Prereg_and_MTMM]], [[HP-05_Difficulty_Matched_WIS_Bank]].
2. Flagship science: [[D02_Causal_Patching]] + [[D08_Commitment_Depth]] (mechanism), [[D01_Controlled_Exposure_Validation]] starting with D1-lite ([[HP-06_D1Lite_Finetune_Calibration]]).
3. Ecological ground truth: [[D04_Developmental_Checkpoints]] on OLMo (exposure measured in the actual corpus).
4. Behavioral expansion: [[D03_Continuous_Transfer_Distance]] with [[D06_Direction_Asymmetry]] and [[D05_Cross_Linguistic]] folded in as distance points.
