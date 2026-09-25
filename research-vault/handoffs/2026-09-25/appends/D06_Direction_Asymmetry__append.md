<!-- APPEND to research-vault/RvC-Vault/04_Directions/D06_Direction_Asymmetry.md -->

## Update 2026-09-25: cheapest discriminating test for F3
- Paper I F3 (Claude .172 → .873 under W5 on BW) has two accounts. Search direction (S) predicts that instructing backward reasoning on the canonical item recovers most of the W5 gain. Instance constraint (C) predicts it does not, and that the gain tracks per-item constraint asymmetry (applicable actions at initial vs goal).
- Execution: [[HP-22_Backward_Prompt_Discriminator]]. Under 100 calls per model; reuses existing canonical and W5 raws.
- Positioning stays as written above: the asymmetry is published (arXiv:2411.01790). Paper I must cite it. Our contribution is the discriminating test and, later, the rename × direction interaction.
- The rename cell in [[HP-11_D6_Direction_Probe]] must use W3a nonce, not the current W3 (isomorph). See [[EF-07_W3_Construct_Audit]].
