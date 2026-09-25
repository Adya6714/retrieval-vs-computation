# HP-20 Surface ensembling vs self-consistency (T6)
addresses: [[D16_Surface_Ensembling]] · phase: Track T · needs: nothing for the retrospective arm; T4 for the open-weight arm

PROMPT:
Goal: test whether majority vote over answer-preserving surfaces beats self-consistency at matched call count.
Rules: same as HP-16. Vote rule fixed before running: majority over {canonical, W1, W2, W3, W4}; ties broken by the canonical answer.
Steps:
1. Retrospective arm ($0): from existing rescored P1 CSVs, extract parsed answers per item × model × variant for ALGO (`model_answer`) and GSM (parse `raw_response` with the existing parser). Compute surface-vote accuracy per model. Output `results/derived/T6_surface_vote_retro.csv`.
2. Open-weight arm (T4): for the HP-17 models, run 5 samples at temperature 0.7, top_p 0.95, on canonical only; majority vote. Surface-vote arm uses the T2 greedy raws. Output `results/raw/T6_selfcons_{model_slug}.csv`, `results/derived/T6_ensemble_compare.csv`.
3. Report accuracy with Wilson CIs (GSM) and cluster bootstrap (ALGO).
Output: `docs/trackT/T6_REPORT.md`, tables only.
Validate: only items with all five surfaces present enter the comparison; list excluded items.
