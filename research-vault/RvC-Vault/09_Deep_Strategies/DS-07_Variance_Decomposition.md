# DS-07 — Variance Decomposition Across Model Populations
family: deeper theory · cost: analysis-only on existing + base/instruct sweeps

**Idea.** Twin-study logic. Fit a variance-components model to per-(item, model) fragility: how much variance is ITEM-driven (shared across all models) vs MODEL-driven (a defect of particular models) vs item×model interaction? Seed data: the 26 universal-collapse items, GSM/ALGO cross-model tables, and the unanalyzed base-vs-instruct 7B sweeps.

**Why deeper.** If item variance dominates, the conclusion is contrarian and important: fragility is an ECOLOGICAL property of the problem distribution in human text (some problems are just underspecified-by-surface-form for everyone), not a per-model flaw. That reframes the entire field's "model X is brittle" language into "problem class Y is brittle across models," and tells practitioners to audit their task distribution, not shop for a tougher model. If model variance dominates, the opposite: model choice is the lever. Either answer is publishable and neither is currently established.

**Method.** Mixed-effects logistic with crossed random effects (1|item)+(1|model)+(1|item:model); report variance partition coefficients with bootstrap CIs. Extends [[EF-06_Open_Methodological_Questions]] #5.
