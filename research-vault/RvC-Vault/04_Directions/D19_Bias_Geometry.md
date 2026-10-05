# D19 Bias geometry (attribute direction vs W3a binding)

status: proposed · Track B, Amendment A2 · execution: pending handoff · not run · hypothesis: H18 · serves: AC-5, [[DS-03_Representational_Invariance_RSA]]
amendment: [[THE_PLAN_AMENDMENT_A2]] · related: [[D18_Protected_Attribute_Surface]], [[HP-23_W3a_Nonce_Bank]], [[D17_Commitment_Depth_T4]], T3 ([[D12_Precision_Compression_Invariance]]), [[D15_RLVR_Surface_Diversity]]

**Claim tested.** The residual-stream direction separating W8 variants overlaps the canonical-vs-W3a direction above a permutation null (cosine and CKA) in the same layer band identified by T3/T7 (H18).

**Question.** Do bias and rename fragility share a localisable representational substrate?

**Hypothesis.** H18 (see [[THE_PLAN_AMENDMENT_A2]]).

**Primary contrast.** Cosine and CKA overlap between (i) the mean residual-stream difference across W8 attribute groups and (ii) the canonical-vs-W3a difference, in the pre-registered T3/T7 layer band, tested against a permutation null over item labels.

**Protocol steps.**
1. Requires D18 W8 bank, HP-23 W3a variants, and models that clear Acc_can ≥ .30 on the selected items.
2. Models: 1.5B to 3B open-weight (same family as Track T where possible), FP16 on open-weight GPU (A100-class when available; 1.5B-3B FP16 is T4-feasible).
3. Extract residual-stream states at the T3/T7 band for canonical, W3a, and W8 group variants (mean-pool problem-span and last-token; report both).
4. Define attribute direction as mean(W8_group_A) - mean(W8_group_B); binding direction as mean(canonical) - mean(W3a).
5. Score cosine and CKA; permutation null over item identity (N ≥ 1000, seed 42).
6. Pre-register band and metrics before the first forward pass of this direction.

**Compute.** Open-weight GPU (A100-class when available; 1.5B-3B FP16 is T4-feasible) or API. No run scheduled; costs estimated at preregistration.

**Kill criterion.** If overlap is not above the permutation null, record "bias and rename fragility are separate mechanisms" as the finding and drop the joint-repair claim; attribute-only repair continues as a separate arm under D20.

**Why this tier.** Mechanistic link from CC-5 behaviour to AC-5; feeds Paper II and the "Same Problem, Different Person" short paper with D18.
