# D06 — Procedural Direction Asymmetry
status: Tier 3 (probe extension P1.5) · execution: [[HP-11_D6_Direction_Probe]]

**Claim tested.** Models solve the same underlying problem graph unequally in forward vs backward direction, and the asymmetry interacts with entity rename — if renaming collapses the *asymmetry*, direction-specific surface templates (not planning capacity) are implicated.

**Positioning honesty.** The base phenomenon is published: forward/backward planning asymmetry tracking search complexity ([[P25_Forward_Backward_Planning_2024]]); code-execution invertibility as a memorization filter is active ([[P26_Code_Invertibility_2026]], abstract-only); reversal curse covers factual recall ([[P28_Berglund_Reversal_2023]]). The open contribution is the *rename × direction interaction on matched problem graphs with exposure control* — an extension of existing W5, exactly as the program brief scopes it. Do not pitch as a new phenomenon.

**Design.** BW assembly vs disassembly on identical state graphs (Fast Downward gives optimal plans both ways — pipeline asset reuse); SP source↔destination swap already exists as W5; add encode/decode pairs for a small string-rewriting family. Condition grid: {forward, backward} × {canonical names, W3 rename}. Control search-complexity asymmetry per Forward/Backward-2024 by matching BFS step counts across directions, so residual asymmetry is not just search geometry.

**Cost.** Cheap (API-only, existing verifiers). Runs inside D3's sweep.

## Update 2026-09-25: cheapest discriminating test for F3
- Paper I F3 (Claude .172 → .873 under W5 on BW) has two accounts. Search direction (S) predicts that instructing backward reasoning on the canonical item recovers most of the W5 gain. Instance constraint (C) predicts it does not, and that the gain tracks per-item constraint asymmetry (applicable actions at initial vs goal).
- Execution: [[HP-22_Backward_Prompt_Discriminator]]. Under 100 calls per model; reuses existing canonical and W5 raws.
- Positioning stays as written above: the asymmetry is published (arXiv:2411.01790). Paper I must cite it. Our contribution is the discriminating test and, later, the rename × direction interaction.
- The rename cell in [[HP-11_D6_Direction_Probe]] must use W3a nonce, not the current W3 (isomorph). See [[EF-07_W3_Construct_Audit]].
