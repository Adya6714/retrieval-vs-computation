# D06 — Procedural Direction Asymmetry
status: Tier 3 (probe extension P1.5) · execution: [[HP-11_D6_Direction_Probe]]

**Claim tested.** Models solve the same underlying problem graph unequally in forward vs backward direction, and the asymmetry interacts with entity rename — if renaming collapses the *asymmetry*, direction-specific surface templates (not planning capacity) are implicated.

**Positioning honesty.** The base phenomenon is published: forward/backward planning asymmetry tracking search complexity ([[P25_Forward_Backward_Planning_2024]]); code-execution invertibility as a memorization filter is active ([[P26_Code_Invertibility_2026]], abstract-only); reversal curse covers factual recall ([[P28_Berglund_Reversal_2023]]). The open contribution is the *rename × direction interaction on matched problem graphs with exposure control* — an extension of existing W5, exactly as the program brief scopes it. Do not pitch as a new phenomenon.

**Design.** BW assembly vs disassembly on identical state graphs (Fast Downward gives optimal plans both ways — pipeline asset reuse); SP source↔destination swap already exists as W5; add encode/decode pairs for a small string-rewriting family. Condition grid: {forward, backward} × {canonical names, W3 rename}. Control search-complexity asymmetry per Forward/Backward-2024 by matching BFS step counts across directions, so residual asymmetry is not just search geometry.

**Cost.** Cheap (API-only, existing verifiers). Runs inside D3's sweep.
