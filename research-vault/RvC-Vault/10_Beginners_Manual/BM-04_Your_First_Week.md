# BM-04 — Your First Week (do this, in this order)

Goal: by day 7 you have run two real, novel analyses (HP-13 and HP-14) that need no GPU and no new data, and you understand the loop. These two ARE frontier contributions ([[DS-02_Intrusion_Error_Analysis]], [[DS-01_Latent_Strategy_Measurement_Model]]), not toy exercises.

## Day 1 — Set up the brain (2–3 hrs)
- Install Obsidian (free). Open this vault as a folder. Read [[BM-00_Start_Here]] and [[BM-01_Glossary_For_Beginners]].
- Put the vault in a private GitHub repo (free). Learn two commands: `git add -A && git commit -m "..."` and `git push`.
- Install Obsidian community plugins: Dataview, Templater. (Skip Zotero/ingestion for now.)
- Success check: you can click a [[link]] in 00_MOC and land on the right note.

## Day 2 — Get the code and data (2–3 hrs)
- Clone the research repo: github.com/Adya6714/retrieval-vs-computation.
- Install Python bits: `pip install pandas numpy scipy matplotlib jupyter --break-system-packages`.
- Open results/raw/ and results/derived/. Just LOOK. Open one CSV in Jupyter, print the columns, read [[EF-05_Metrics_Glossary]] alongside so the column names mean something.
- Success check: you can load GSM_P1_behavioral_gpt4o.csv and count how many rows have canonical correct.

## Day 3–4 — Run HP-13 (intrusion errors), your first novel result
- Open [[HP-13_Intrusion_Error_Analysis]]. Read it fully.
- Do step 1–2 yourself in Jupyter: for items where canonical is right but W3 is wrong, check whether the W3 answer equals the CANONICAL answer. That fraction is the "canonical-intrusion rate" — direct evidence of memory interference nobody has quantified this way.
- Do the manual audit (step 3) on ~50 traces with the rubric. This is slow and boring and is exactly what real research feels like.
- Write INTRUSION_REPORT.md. Then WRITE BACK: add a line to [[01_Program_State]] changelog.
- Success check: you have a per-model intrusion-rate table with confidence intervals.

## Day 5–6 — Run HP-14 (mixture IRT), your second novel result
- Install one tool: R with `mirt` (or Python `pip install stepmix`). 
- Follow [[HP-14_Mixture_IRT_Fit]]: build the indicator table (include your Day 3–4 intrusion column), fit 1–4 classes, pick by BIC, inspect which class is the "retrieval" class.
- This replaces the shaky hand-set labels with a fitted model — the single deepest upgrade in the program.
- Success check: you can say "BIC prefers K classes; the retrieval-process class has high W3-collapse + high intrusion + high proximity," and you exported strategy_posteriors.csv.

## Day 7 — Reflect and plan
- Run [[BM-03_Thinking_Checklists]] on both results. What's the alternative explanation for your intrusion finding? Write it down.
- Read [[02_Decision_Memo]] again — it will mean much more now.
- Pick your next 2 weeks: the natural next step is [[HP-04_Threshold_Prereg_and_MTMM]] (more free analysis) then planning [[HP-06_D1Lite_Finetune_Calibration]] (your first GPU experiment).

## What you should NOT do in week 1
- Don't rent GPUs. Don't fine-tune anything. Don't build the ingestion pipeline. Don't try to read all 34 papers. Depth on two real analyses beats breadth on zero.
