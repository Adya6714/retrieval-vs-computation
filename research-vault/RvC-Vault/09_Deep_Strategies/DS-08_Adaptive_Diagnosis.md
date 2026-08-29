# DS-08 — Adaptive Diagnosis (CAT for strategy)
family: deeper theory / instrument · cost: low, requires DS-01

**Idea.** Once DS-01 gives a generative measurement model, the instrument can choose its OWN next perturbation per item — the variant with maximum expected information about THIS item's latent strategy class (computerized adaptive testing, repurposed from ranking test-takers to diagnosing a single response). Stop when class posterior passes a confidence threshold.

**Why deeper.** (1) Makes the eventual audit product ([[D10_Structural_Audit_Suite]]) an order of magnitude cheaper — a handful of adaptively chosen probes instead of a fixed 7-variant battery. (2) It is itself a scientific claim: the perturbations that are most diagnostic reveal which surface features the model's competence actually hangs on.

**Prerequisite.** DS-01 fitted and validated. Pre-register the information criterion (expected posterior entropy reduction) and stopping rule.
