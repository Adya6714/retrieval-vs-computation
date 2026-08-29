# HP-04 — Threshold Pre-Registration, MTMM Matrix, Pooled Models
addresses: [[EF-04_Convergence_Labels_MTMM]], [[EF-06_Open_Methodological_Questions]] #1/#4/#5 · phase: 0 · needs: repo, Python (statsmodels), no API

PROMPT:
Three analysis tasks on existing CSVs; no new data.
1. Threshold pre-registration: write docs/LABEL_THRESHOLDS_PREREG.md freezing the convergence-label rules currently in the CAISc draft (W3-collapse, CCI, proximity cutoffs), each with one-paragraph justification. Recompute the strict/liberal sensitivity sweep (expect ~2.7% vs ~57.7% strong-label rates on ALGO 440) and present it as sensitivity analysis subordinate to the frozen rules. Also reconcile the VRI definition mismatch: paper App. C = mean(W1,W2,W4)−W3 vs scripts css.py = mean(W2,W4)−W3; pick one (recommend the paper's), patch the script, regenerate derived tables, and report any numbers that move.
2. MTMM matrix: per model, on continuous per-instance signals (per-item R_W3 or CSS, CCI, template+instance proximity), compute the multitrait–multimethod correlation matrix (Campbell & Fiske); report convergent (same-trait) vs discriminant (cross-trait) blocks with bootstrap CIs. Deliver as a figure + table replacing raw label counts as headline validity evidence.
3. Pooled models: (a) mixed-effects logistic, correct ~ variant*model*family + (1+variant|problem_id), on the P1 long table; report variant×model contrasts vs the per-cell Fisher results. (b) 2PL IRT with items as GSM/ALGO instances and an exposure covariate on item difficulty — the statistical attack on the WIS exposure–difficulty confound.
Output: prereg doc, analysis notebook, 2 figures, results markdown with verified-raw tags.
Validate: n's must reconcile with COVERAGE_AUDIT_SUMMARY.md; every reported coefficient carries a CI; convergence of the mixed model must be checked (report optimizer + any singular fits honestly).
