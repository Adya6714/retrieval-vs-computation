# D15 RLVR with surface-diverse verifiable environments

status: Phase 3, gated behind G2 · execution: [[HP-21_RLVR_Dry_Run]] (plumbing only until G2) · hypothesis: H7b (RL form of H7 / AC-3)

**Claim tested.** With matched optimisation steps, RL from verifiable rewards on surface-diverse prompts yields higher held-out invariance than RL on canonical-only prompts, at equal canonical accuracy. This is AC-3 (diversity, not dose) in the regime frontier labs actually train in.

**Programme asset.** The verifiers are RL reward functions already: exact DP solver, PDDL simulator, numeric match, and the generator persists entity mappings so renamed answers verify. Surface-diverse verifiable environments are scarce in public RLVR work.

**Design.**
- Policy: Qwen2.5-0.5B-Instruct (dry run on T4), Qwen2.5-1.5B-Instruct (real run, 1× A100-day).
- Task: coin change from a procedural generator (fresh instances each step, so no item is memorised).
- Arm A: canonical surface only. Arm B: prompts sampled uniformly from canonical, W1, W2, W4, W6 renderers. Same steps, same group size, same KL.
- Arm C (optional): Arm A + consistency reward (same answer on a paired canonical/W3b prompt), gated by correctness to prevent constant-answer collapse.
- Held-out evaluation: W3a nonce and W3b isomorph renderers never used in training; transfer to WIS (rare subtype) and shortest_path.

**Metrics.** Acc_can, R_W3a, R_W3b, WIS transfer accuracy, per arm, over training steps (a small developmental curve, links to CC-3).

**Prediction.** Equal Acc_can; Arm B higher R_W3a/R_W3b; Arm C highest R but risk of reward hacking (log answer entropy).

**Kill criterion.** If Arm B Acc_can falls more than 5 points below Arm A at the matched step, dose was not matched in effective terms; rebalance and rerun once, then report.

**Cost.** Dry run: T4, $0. Real run: roughly one A100-day.
