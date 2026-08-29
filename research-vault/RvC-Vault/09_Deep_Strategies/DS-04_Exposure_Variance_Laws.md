# DS-04 — Exposure-Variance Laws (variability of practice)
family: deeper causation · cost: ≈ $150–600 (D1-mid infra) · gated on HP-06

**Idea.** Everyone manipulates exposure COUNT. Learning science says the causal variable for transfer is exposure VARIANCE: variability of practice, interleaving, and "desirable difficulties" build transferable skill in humans; identical repetition builds brittle recall. Experiment: continued-pretrain an open 1B model with template exposure at FIXED total count but varied surface diversity — arm A: 1 surface form × 1000 occurrences; arm B: 10 forms × 100; arm C: 100 forms × 10; arm D: interleaved vs blocked scheduling of the same documents. Measure W3 retention, intrusion rate (DS-02), RIS (DS-03), Commitment Depth.

**Why deeper.** A positive result is a causal law of data curation — invariance is bought with diversity, not dose — with immediate practitioner value ("diversify surface forms; naive dedup that collapses paraphrases may be harming robustness"). It also reframes DS-07's ecological question: universally fragile items may be exactly those that appear in human text with LOW surface variance.

**Positioning.** Ruis et al. showed procedural documents drive reasoning ([[P18_Ruis_Procedural_2024]]); dedup literature shows repetition effects on memorization; the variance-at-fixed-dose experiment with a validated fragility instrument as the outcome appears unrun.

**Risks.** Confounding diversity with effective dataset size (hold token count identical); paraphrase quality drift across arms (generate all forms from one grammar, verify with the family verifier); small-model floor (use easy instances).
