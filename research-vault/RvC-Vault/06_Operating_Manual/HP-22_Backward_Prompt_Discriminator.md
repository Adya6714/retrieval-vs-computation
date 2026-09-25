# HP-22 Backward-prompting discriminator for F3 (T5)
addresses: [[D06_Direction_Asymmetry]] · phase: Track T · needs: < 100 API calls per model (free-tier Gemini Flash is sufficient) or none if skipped

PROMPT:
Goal: discriminate the two accounts of F3 (initial↔goal exchange makes Blocksworld easier).
Account S (search direction): backward chaining from a specified goal is easier than forward search. Prediction: telling the model to plan backward on the CANONICAL item recovers most of the W5 gain.
Account C (instance constraint): goal states in released instances are more constrained than initial states. Prediction: backward prompting on the canonical item does not help; the gain tracks a per-item constraint asymmetry.
Steps:
1. Items: the 50 standard BW canonical items (exclude obfuscated).
2. Covariates computed by script (no model calls): number of applicable actions in the initial state vs the goal state (branching proxy); number of goal atoms; Fast Downward optimal plan length. Write `results/derived/T5_bw_constraint_covariates.csv`.
3. Condition B prompt: canonical item + one fixed instruction: "First reason backward from the goal state to the initial state. Then output the plan in forward order." Everything else identical to the P1 template.
4. Run condition B for every model you can reach at T=0. Conditions A (canonical) and C (W5) are reused from existing raws.
5. Verify with the existing PDDL simulator.
Output: `results/raw/T5_backward_prompt.csv`, `results/derived/T5_ABC_accuracy.csv` (per model: Acc_A, Acc_B, Acc_C, Wilson CIs), `docs/trackT/T5_REPORT.md` tables only.
Validate: condition B output parses as a forward plan; parsing failures counted and listed.
