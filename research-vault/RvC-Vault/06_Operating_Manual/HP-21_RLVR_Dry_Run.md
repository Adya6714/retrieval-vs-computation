# HP-21 RLVR surface-diversity dry run (plumbing only)
addresses: [[D15_RLVR_Surface_Diversity]] · phase: 3 (gated behind G2; this HP produces no reportable result) · needs: Colab T4, TRL

PROMPT:
Goal: make the RLVR arms runnable end to end so the real run is one command once G2 passes and an A100 exists.
Steps:
1. `scripts/trackT/coin_change_env.py`: procedural coin-change generator (denominations, target ranges matched to the bank's difficulty_params) with renderers canonical, W1, W2, W4, W6 and held-out W3a/W3b renderers. Every instance is solved by the existing exact solver; reward = 1 if the verifier accepts the parsed answer, else 0.
2. `scripts/trackT/grpo_arms.py` using TRL GRPOTrainer: `--arm {A,B,C}`, `--steps`, `--model`. Arm A canonical-only; Arm B uniform over the five training renderers; Arm C = A + correctness-gated consistency reward.
3. Dry run: Qwen/Qwen2.5-0.5B-Instruct, LoRA r=16, 20 steps per arm, group size 4. Confirm rewards are non-zero, logging works, evaluation on held-out renderers runs.
Output: code, a `docs/trackT/HP21_DRYRUN.md` with runtime, memory, and a note that no numbers from this run may be cited.
Validate: the environment never emits a W3a/W3b rendering during training (unit test).
