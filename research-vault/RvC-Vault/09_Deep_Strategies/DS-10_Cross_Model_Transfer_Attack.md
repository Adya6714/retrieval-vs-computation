# DS-10 — Cross-Model Fragility Transfer (are the cliffs shared?)
family: deeper theory · cost: low, existing data + one small sweep

**Idea.** Do the specific items/variants that break model A also break model B? Build the item×model fragility matrix and test its rank structure: is there a dominant shared "difficulty-by-surface" dimension (one latent fragility factor most models load on) or are cliffs idiosyncratic? Adversarial-transfer work shows perturbations transfer across models ([[P34_ReEval_Transfer_2023]]); nobody has asked whether STRATEGY-fragility (not adversarial loss) shares a latent structure across a model population.

**Why deeper.** A shared fragility factor would imply a common cause — shared pretraining-corpus surface statistics — linking DS-07's ecological hypothesis and DS-04's variance hypothesis into one story: models inherit the same brittleness from the same text. That is a claim about how the AI ecosystem works, not one model.

**Method.** Low-rank factorization of the fragility matrix; report shared-variance fraction; correlate item loadings with proximity and DS-02 intrusion.
