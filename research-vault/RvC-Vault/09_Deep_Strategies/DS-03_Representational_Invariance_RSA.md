# DS-03 — Representational Invariance Geometry (RSA/CKA import)
family: deeper measurement · cost: 1–2 GPU-days ≈ $50–150 · joins the [[HP-07_D2_Patching_Llama]] GPU sessions

**Idea.** Score representations, not just answers. For each item's variant ladder (canonical, W1…W6, later D3 rungs), extract residual-stream states from an open model at several layers; compute representational similarity (RSA from systems neuroscience; CKA from deep learning) between canonical and each variant. Define **Representational Invariance Score** RIS(item, layer) = similarity(canonical, W3) normalized by within-canonical variability.

**Why deeper.** (1) Works even when answers are wrong — fragility becomes measurable below the behavioral floor. (2) Gives a continuous mechanistic dose-response curve when run over D3 ladders. (3) Layer profile is informative: computation-consistent items should show early divergence (different tokens) converging by mid layers (same problem); retrieval-consistent items should never converge or diverge again late.

**Validation chain.** RIS should predict behavioral retention out-of-sample; separate D1-lite seen/unseen; and localize the layers where D2 patching restores answers. If all three hold, RIS becomes the cheapest mechanistic column in the MTMM matrix (no patching needed, one forward pass per variant).

**Scoop status.** RSA is heavily used for LLM–brain alignment and CKA for model comparison; aimed at perturbation ladders as a per-item invariance instrument tied to a behavioral suite — not found. Verify once more before drafting.

**Risks.** Similarity metrics are sensitive to token alignment (use mean-pooled problem-span states + last-token states, report both); layer choice cherry-picking (pre-register the layer bands).
