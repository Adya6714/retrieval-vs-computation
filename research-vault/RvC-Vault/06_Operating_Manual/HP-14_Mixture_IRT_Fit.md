# HP-14 — Latent-Strategy Mixture IRT Fit (zero new data)
addresses: [[DS-01_Latent_Strategy_Measurement_Model]] · phase: after HP-04 + HP-13 · needs: R (mirt/sirt) or Python (StepMix/py-irt), existing CSVs

PROMPT:
Goal: replace threshold labels with a fitted latent-class strategy model.
Steps:
1. Build a per-(model,item) table of indicators: per-item W3 retention (0/1 correct under W3 | canonical correct), CSS, CCI≈0 flag, proximity bin, canonical_intrusion (from HP-13). Binarize/ordinalize as each package needs; document coding in docs/IRT_CODING.md.
2. Fit latent-class / mixture models with 1,2,3,4 classes (start dichotomous mixture Rasch; then allow class-specific difficulties). Use family as a covariate or fit multilevel to respect item nesting (note the local-independence caveat in [[DS-01_Latent_Strategy_Measurement_Model]]).
3. Model selection by BIC (and AIC, report both); inspect class profiles — which class shows W3-collapse + CCI≈0 + high proximity + high intrusion (the retrieval-process class).
4. Sanity anchors: check that the 26 universal-collapse items and GSM Claude's 35 computation-leaning items land in the expected classes. Export per-response posterior class probabilities.
5. Validation hook: hand posteriors to HP-06 so D1-lite SEEN/UNSEEN can be tested against CLASS membership (not thresholds) — the real calibration.
Output: fitted model objects, class-profile table + figure, results/derived/strategy_posteriors.csv, IRT_REPORT.md stating chosen class count with BIC justification.
Validate: report convergence and any label-switching handling; if 4-class fit is unstable at current n, say so and default to the best stable K; no causal language — classes are latent, validated only once HP-06 aligns them to ground truth.
