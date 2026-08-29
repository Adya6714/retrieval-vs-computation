# DS-01 — Latent-Strategy Measurement Model (mixture IRT)
family: deeper measurement · cost: $0 · execution: [[HP-14_Mixture_IRT_Fit]]

**Idea.** Replace threshold-based convergence labels with a fitted latent-class measurement model. Psychometrics built exactly this: Mislevy & Verhelst 1990 ("Modeling item responses when different subjects employ different solution strategies") and Rost's 1990 mixture Rasch model, where latent classes of respondents have different item parameters and class membership is estimated from response patterns ([[P31_Mislevy_Verhelst_1990]]). Port: the "respondents" are (model, item) response bundles; the indicators are the continuous probe signals (per-item R_W3, CSS, CCI, proximity, intrusion rate from DS-02, later Commitment Depth and RSA-invariance). Fit a 2–4 class mixture; each response gets a posterior probability of retrieval-process vs computation-process class.

**Why deeper.** The threshold-sensitivity weakness (2.7% vs 57.7% strong labels) dissolves: class boundaries come from likelihood + BIC model selection, not hand rules. Labels become posterior probabilities with uncertainty attached. And D1-lite ground truth then validates *classes*, not thresholds — if the fitted retrieval class aligns with seen items, the instrument is calibrated in the full psychometric sense.

**First experiment.** Long-format table of per-(model,item) signals from existing CSVs → fit 1/2/3/4-class models (mirt or sirt in R; StepMix or py-irt in Python) → compare BIC → inspect class profiles → check class assignments against the 26 universal-collapse items and GSM Claude's 35 computation-leaning items as sanity anchors.

**Scoop status (checked 2026-07-07).** Mixture IRT applications found are all human educational testing; LLM-side IRT is used for benchmark efficiency/ranking, not strategy diagnosis. Open.

**Risks.** Local independence violations (items nested in families — use family covariates or multilevel mixture); class label-switching (anchor with clear indicator items); small n for stable 4-class fits (start with 2).
