# BM-06 — Publishing, Conferences, and What "Best Paper" Actually Means

## The venues (where this work goes)
- **ACL, EMNLP, NAACL** — NLP. Behavioral/evaluation work (the instrument, the fragility laws) fits here. ARR (ACL Rolling Review) is the shared submission system.
- **NeurIPS, ICML, ICLR** — general ML. Method + validation + mechanism papers fit; ICLR uses OpenReview with public reviews you can read.
- **COLM** — newer, LLM-focused; a natural home for this whole line.
- **TMLR** — journal-style, rolling, values correctness over novelty-hype; good for the thorough measurement paper.
- **Workshops (BlackboxNLP, GenBench, etc.)** — lower stakes, fast feedback, good for interim pieces. You have used these already.

## What actually wins best-paper / spotlight (the uncomfortable truth)
Not size. Not "we ran the most experiments." The pattern across recent best papers is: ONE sharp, surprising, well-verified claim that changes how people think, stated cleanly, with the obvious objections already closed. GSM-Symbolic is small and reframed a whole conversation. Reviewers reward a load-bearing result others must build on, not a sprawling system.

Concretely, best papers tend to have:
- A crisp claim a smart outsider can restate in one sentence.
- A result that is surprising OR settles a live dispute.
- Airtight controls for the obvious confounds (for us: tokenization, difficulty, exposure).
- A validation the reader trusts (for us: calibration against ground truth — this is our edge).
- Clear honesty about scope and limits.

## The strategic consequence for THIS program (important)
Build BROAD (the vault, all the DS strategies) but SHIP NARROW. A giant unified paper is usually a WEAK paper — reviewers can't find the claim, and every extra claim is another attack surface. So:
- Each paper in [[DS-12_Grand_Synthesis]]'s decomposition is deliberately narrow: the instrument; the mechanism; the laws; the ecology; the inverse.
- The breadth lives in the research program and the vault, not in any single submission.
- Decide paper boundaries from the EVIDENCE you actually get, not up front ([[02_Decision_Memo]] publication-cleavage note).

## What the AI industry actually needs (why this matters beyond a paper)
The field is drowning in leaderboards and starving for MEASUREMENT. Scores go up; nobody can certify what a score means. The three things practitioners genuinely lack, which this program provides:
1. A way to tell whether a capability is real or contamination — per instance, calibrated ([[D01_Controlled_Exposure_Validation]], [[DS-01_Latent_Strategy_Measurement_Model]]).
2. A cheap monitor for memorization creeping in during training ([[BI-01_Training_Time_Diagnostic]], [[D08_Commitment_Depth]]).
3. A pre-deployment audit that predicts failure under harmless real-world variation ([[BI-02_Predeployment_Audit]]).
"Best in the world" here is not the biggest study; it is the first VALIDATED instrument plus the first predictive LAWS. Fields become sciences when they can measure and predict, not when they catalog failures.

## Practical publishing habits
- Read OpenReview reviews of accepted AND rejected papers in your area — the fastest way to learn the bar.
- Keep a claim-audit table per paper: every quantitative sentence -> (file, filter, script). ([[HP-02_Draft_Reconciliation]] is the template.)
- Pre-register key analyses; report negative results honestly (a failed strong prediction is still a finding — see [[DS-06_Fragility_Laws_Prediction]]).
- Put out a clean interim workshop paper early; use the feedback to aim the main submission.
