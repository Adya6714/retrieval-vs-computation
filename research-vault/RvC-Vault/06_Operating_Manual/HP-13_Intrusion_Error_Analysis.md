# HP-13 — Intrusion-Error Analysis (zero new data)
addresses: [[DS-02_Intrusion_Error_Analysis]] · phase: can run NOW · needs: repo, pandas, ~2 hrs manual audit

PROMPT:
Goal: turn wrong answers into a per-instance signal by classifying WHAT the model answered when W3 failed.
Steps:
1. From results/raw/ P1 files, for each (model, item) with canonical correct AND W3 incorrect, extract: the model's W3 final answer, the canonical correct answer, and the W3 correct answer.
2. Rule-based labels: canonical_intrusion = (W3 answer == canonical answer within numeric tol); else flag for manual review.
3. Manual audit: sample 100 flagged traces across families/models; label each computational_slip / procedure_intrusion / degenerate using a written 1-paragraph rubric per class (put the rubric in docs/INTRUSION_RUBRIC.md). Two-pass by the same annotator on 20 items to get self-agreement.
4. Compute per-model intrusion-type rates with Wilson CIs; correlate canonical-intrusion rate with (a) proximity, (b) CCI≈0 flag. Export per-item canonical_intrusion (0/1) and type for use as a DS-01 indicator.
Output: results/derived/intrusion_labels.csv, INTRUSION_RUBRIC.md, INTRUSION_REPORT.md with the rate table + two correlation figures.
Validate: every canonical_intrusion=1 row must have W3 answer numerically equal to the canonical answer (spot-check 20); report annotator self-agreement; unparsed answers excluded and counted, never labeled.
