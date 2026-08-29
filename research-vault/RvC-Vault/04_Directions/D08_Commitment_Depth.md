# D08 — Commitment Depth: Formalizing the Mechanistic Metric
status: Tier 1 ride-along with D02 · execution: [[HP-08_D8_Commitment_Depth]]

**Construct.** CD_k(item) = earliest layer at which the gold answer token enters the top-k of the unembedding projection of the residual stream, read at the answer position. The existing Qwen-2.5-7B sweep already computed the k=100 version informally and found it the strongest single predictor of final rank (r≈+0.82) — this direction names it, defines it, and validates it. (Provenance note: this construct was suggested by an external "Triangulation Framework" document whose *numbers* conflicted with our data; per program policy the idea is extracted, the numbers quarantined.)

**Validation plan.** (1) Replicate on Llama-3.1-8B (same model as behavioral pool + D2). (2) Convergent validity: CD should differ between D1-lite seen vs unseen items (early commitment on memorized items). (3) Causal warrant: D2 patching at layer CD±2 should have outsized effect vs other layers. (4) Robustness: k ∈ {10, 50, 100}; answer-position vs final-position readout.

**Payoff.** One number per (item, model) that summarizes "how early the answer was fixed" — the mechanistic column of the MTMM matrix, the dependent variable for D4's developmental traces, and the candidate training-time diagnostic in [[BI-01_Training_Time_Diagnostic]]. Speculative-flag discipline: it is a *candidate* diagnostic until D1-lite ground truth confirms it separates memorized from computed items.
