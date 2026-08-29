# Program State — Confirmed / Hypothesis / Open

Last updated: 2026-07-07 (raw-data audit of github.com/Adya6714/retrieval-vs-computation, commit at clone time).

## A. Resolved this pass (verified-raw)

**A1. GSM GPT-4o/Llama coverage discrepancy — RESOLVED. The CAISc draft is correct; the EMNLP draft's GSM numbers do not reproduce.**
Direct recount of `results/raw/GSM_P1_behavioral_{gpt4o,llama}.csv` against `data/problems/question_bank_gsm.csv`:
- Bank canonical n=44 (GSM_001–020, GSM_041–064; no 021–040 in bank). Confirmed.
- GPT-4o: 64 canonical rows total → 44 in-bank → 24 are `ERROR:` rows (all of GSM_041–064) → **20 valid (GSM_001–020 only)**. Canonical acc **0.850**, W3 acc **0.300** (n=20).
- Llama: identical structure. **20 valid**, canonical **0.800**, W3 **0.150**.
- GSM_021–040 rows are duplicate reruns of 001–020 (repo's own `COVERAGE_AUDIT_SUMMARY.md` confirms: not a remap to 041–060).
- Consequence: EMNLP draft Table 3 (n=40, .825/.825, and its claim that coverage = GSM_001–020 + 041–060) is unsupported by the committed raws. Retire those numbers everywhere; the honest cell is n=20 partial replication until HP-01 gap-fill runs. See [[HP-01_GSM_GapFill_and_Canonical_Numbers]], [[HP-02_Draft_Reconciliation]].

**A2. Qwen 7B vs 0.5B mechanistic provenance — RESOLVED.**
Four distinct mechanistic artifacts exist; they were never one dataset:
- `mechanistic_sweep_7b_base_rawqa.csv`: **Qwen/Qwen2.5-7B (base)**, 398 rows, includes `target_rank_per_layer` → this is the file behind Appendix F (paper says n=398; matches).
- `mechanistic_sweep_7b.csv` and `mechanistic_sweep_7b_rawprompt.csv`: **Qwen2.5-7B-Instruct**, 268 rows each, no rank column → unused in the drafts; a base-vs-instruct comparison is sitting unanalyzed ([[HP-03_Mechanistic_Provenance_Ledger]]).
- `{ALGO,GSM,BW}_P3_mechanistic.csv`: **Qwen2.5-0.5B-Instruct**, 20+20+35 = 75 rows → exploratory pilots only. Any "0.5B results" referenced in conversation refer to these; they must never be cited as supporting Appendix F claims.

**A3. Triangulation counts — CONFIRMED.** `ALGO_P3_triangulation_v3.csv`: 440 rows; ambiguous 271, mixed 157, retrieval_signal 8, computation_signal 4. Matches both drafts exactly.

**A4. Coverage audit (repo `COVERAGE_AUDIT_SUMMARY.md`):** 35/40 model×probe slices bank-complete. Incomplete: GSM P1 GPT-4o/Llama (20/44) and ALGO P2A-elicited for Claude/Gemini/Llama (61/110).

## B. Confirmed findings (verified-raw or reproduced in both drafts)
- Subtype-specific robustness inversion: SP-adv Claude 0.647→0.000 under W3 vs GPT-4o 0.412→0.265 (n=34, Fisher p=0.0021); CC reverses the pair (n=10, exploratory).
- WIS universal W3 collapse incl. o4-mini 1.00→0.00; exposure–difficulty confound acknowledged (Table 5) — the confound is real and blocks any strong exposure claim until [[HP-05_Difficulty_Matched_WIS_Bank]].
- Injection compliance ≠ final correctness (88–100% acceptance, ~50% post-injection accuracy); plausibility null across 5 models; elicitation null across 5 models.
- BW W5 block-rename sign flip: Claude +23.9pp, Gemini +18.5pp, Llama −22.0pp, GPT-4o/o4-mini 0 (NL-tolerant protocol).
- BW strict-PDDL protocol aborts 84–100% — a measurement finding, kept separate from model claims.
- Proximity→VRI correlation for Claude (r=+0.44) and GPT-4o (r=+0.37) only; survives residualizing on canonical accuracy; o4-mini breaks the link (r=−0.09, one model).

## C. Standing hypotheses (not yet established)
- H1: Convergence labels track actual training exposure (never tested against ground truth → [[D01_Controlled_Exposure_Validation]]).
- H2: W3 collapse in "retrieval-consistent" instances is mediated by a canonical answer/binding representation that persists under rename (correlational only → [[D02_Causal_Patching]]).
- H3: Fragility decays with surface distance as a curve with a family-consistent cliff ([[D03_Continuous_Transfer_Distance]]).
- H4: Surface-invariant reasoning emerges late in training and tracks accumulated template exposure ([[D04_Developmental_Checkpoints]]).
- H5: The injected-state recovery effect reflects an answer direction fixed before the injection point (patching-testable, ties H2 to Probe 2).

## D. Open issues / known weaknesses
- Central claim is behavioral-only; the one mechanistic pilot is single-model, correlational, and — important — contrasts **canonical vs W6**, not the W3 rename that carries the headline. The mechanistic evidence and the headline claim are currently about different manipulations. Fix in [[D02_Causal_Patching]].
- Label thresholds swing strong-label rate 2.7%→57.7%; labels are a design choice until externally calibrated ([[HP-04_Threshold_Prereg_and_MTMM]]).
- Infini-gram over Pile/DCLM is not the corpus of any tested model; promised Llama validation not delivered. OLMo+Dolma makes exposure measurable in the actual corpus ([[D04_Developmental_Checkpoints]]).
- One reasoning-trained model; per-cell Fisher tests instead of a pooled mixed-effects model ([[EF-06_Open_Methodological_Questions]]).
- Compute reality: mechanistic pilot ran on a Colab T4. D2/D6-mech/D8 need ≥1 A100-40GB-class GPU; D1 full needs multi-GPU training budget. **Open question for Adya: what GPU budget and OpenRouter budget are actually available?**

## E. Changelog
- 2026-07-07: A1–A4 resolved via direct repo audit; vault created; decision memo v1.
- 2026-07-07 (later): added Deep Strategies series (DS-01..DS-12), Lab build/landscape/problem-engine/cost notes (LB-00..03), and a full Beginners Manual (BM-00..07). New paper notes P31 (Mislevy & Verhelst mixture IRT), P32 (perturbation-sensitivity), P33 (code memorization MRI), P34 (ReEval transfer). New handoff prompts HP-13 (intrusion errors) and HP-14 (mixture-IRT fit), both zero-cost on existing data. Scoop-checked mixture-IRT-for-strategy and intrusion-content signals: both appear open.
- 2026-07-07 (later still): added DS-13 (capability/architecture claim ladder), P35 (Yang et al. emergent symbolic mechanisms, ICML 2025), HP-15 (pathway-occupancy pilot). New hypotheses H6 (strategy = pathway occupancy) and H7 (diversity forces abstraction-head formation) registered.
- 2026-07-07 (final consolidation): THE_PLAN v1.0 frozen at vault root — single canonical phased roadmap (Phases 0–5, gates G0–G3, claim registry CC-1..4 / AC-1..4, hypothesis registry H1–H7, budget envelope). BM-08 reading curriculum added. OM-00 priority order updated to final sequencing. All prior planning notes remain as supporting detail; THE_PLAN wins conflicts.
- 2026-07-07 (deep-methods pass): added 11_Borrowed_Methods folder mapping six families of hidden-mind inference (dissociation, transfer/concept, memory-signature, adaptation, developmental/lesion, individual-differences) to new LLM evaluation strategies DS-14..DS-19, plus BX-07 frontier list (psychophysics, animal cognition, sociology, linguistics/wug, causal inference) and EVAL-CATALOG_Extended. Verified: 'machine psychology'/'LLM psychometrics' fields (Ye et al. 2505.08245, Hagendorff) borrow personality/ability TASKS but not the inferential MACHINERY — that machinery is the moat. Key inspiration papers logged: BabyReasoningBench 2601.18933, CogBench, Berko wug test, Teuber double dissociation.
