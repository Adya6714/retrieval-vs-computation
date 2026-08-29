# THE PLAN — v1.0 (frozen 2026-07-07)

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
