# D04 — Developmental Study on Open Checkpoints
status: Tier 2 · execution: [[HP-09_D4_OLMo_Checkpoint_Sweep]]

**Claim tested.** H4: surface-invariant reasoning emerges over training and tracks accumulated exposure — measured in the *actual* corpus, not a proxy.

**Why now.** OLMo 3 (Nov 2025) shipped intermediate checkpoints across pretrain/midtrain/long-context/post-train stages for 7B and 32B, plus the full Dolma 3 corpus ([[P20_OLMo3_2025]], verified-source via release coverage). OLMo 2 provides checkpoints along ~4T-token runs at 1B/7B/13B; Pythia adds a 154-checkpoint ladder at smaller scales. Consequence: Probe 3 stops being a proxy for one model family — count template/procedural occurrences in Dolma directly (index or targeted grep at scale), then correlate accumulated exposure with per-item W3 retention across checkpoints.

**Design.** X-axis: checkpoint (tokens seen). Y-axes per item: canonical accuracy, R_W3, Commitment Depth ([[D08_Commitment_Depth]]) — the last gives a mechanistic developmental trace, which to our knowledge has not been published for reasoning robustness. Questions: does invariance emerge gradually or in a phase change; does it precede or follow canonical accuracy; does midtraining's math/code mix (documented in the OLMo 3 report) coincide with retention jumps?

**Bridge role.** D1 = controlled ground truth; D4 = ecological ground truth. Together they make the exposure story airtight from both ends.

**Risks.** Early checkpoints floor on ALGO → include easy instances; report retention conditional on canonical-correct. Base models need scaffolding → fixed few-shot harness held constant across checkpoints (deviating from the paper's zero-shot scope is fine here; different question). Dolma counting cost → start with template-string and solution-string counts; escalate to fuzzy/procedural matching only if flat.

evidence: [[P20_OLMo3_2025]] [[P09_Razeghi_2022]] [[P18_Ruis_Procedural_2024]]
