# D20 Debias by repair (steering and demographic-diverse RLVR)

status: proposed · Track B, Amendment A2 · execution: pending handoff · not run · hypotheses: H19, H20 · serves: AC-5, AC-2, AC-3
amendment: [[THE_PLAN_AMENDMENT_A2]] · related: [[D18_Protected_Attribute_Surface]], [[D19_Bias_Geometry]], [[D15_RLVR_Surface_Diversity]], [[D17_Commitment_Depth_T4]], T3 ([[D12_Precision_Compression_Invariance]])

THE_PLAN gates AC-2 repair behind G2 (H6 verdict). Both steering arms below are proposed under G2, not G1. THE_PLAN wins any conflict.

**Claim tested (proposed, not run).** Steering against a representational direction reduces W8 gap and/or W3a cost with canonical accuracy loss ≤ 2 points (H19, arm-dependent). Separately, RLVR with demographic-diverse surfaces reduces the W8 gap on held-out attribute groups more than canonical-only RLVR at matched steps and equal Acc_can (H20, extends H7b).

**Question.** Can one repair reduce bias, rename fragility, or both?

**Hypothesis.** H19 (joint arm), H19-style attribute-only steering (attribute-only arm), H20 (RLVR).

**Steering arms (both gated behind G2).**
- **Joint arm (H19):** Project or steer against the shared W8/W3a direction in the band identified by D19. Prerequisite: D19 verdict supported (shared substrate). Primary contrast: Δ W8 group-vs-group gap and Δ W3a retention vs sham; Acc_can drop ≤ 2 points on neutral-name baseline.
- **Attribute-only arm:** Steer against the W8 attribute direction only (no joint W3a objective required). Prerequisite: D18 non-null (W8 gap above T1 floor). Same accuracy cap and held-out group rules.

**RLVR arm (H20).** Demographic-diverse vs canonical-only at matched steps; gated behind G2 like D15. Prerequisite: same envelope as [[D15_RLVR_Surface_Diversity]].

**Protocol steps.**
1. No steering until G2 has passed (THE_PLAN / AC-2).
2. Joint arm only if D19 supports shared substrate; attribute-only arm if D18 is non-null regardless of D19 joint verdict.
3. Extract steering vectors from a held-out item set; evaluate on disjoint items and attribute groups not used in extraction.
4. Band constrained to the T3/T7 (or D19) band; sham control: random direction in the same subspace.
5. RLVR arm: gated behind G2 like D15; plumbing dry run on Qwen2.5-0.5B allowed with no reported result.
6. Pre-register arm choice, Acc_can loss bound (2 points), and held-out group list before any steering run.

**Compute.** Steering: open-weight GPU (A100-class when available) or API. RLVR: gated behind G2 (A100-class), same envelope as D15. No run scheduled; costs estimated at preregistration.

**Kill criterion.** If canonical accuracy drops > 2 points, or the gap closes only on attribute groups seen during steering-vector extraction, report as failed installation for that arm.

**Why this tier.** Turns diagnosis (D18/D19) into intervention; H20 is the fairness form of AC-3 / H7b.
