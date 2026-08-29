# HP-15 — Symbolic-Pathway Occupancy on Canonical/W3 Pairs
addresses: [[DS-13_Capability_Architecture_Claims]] AC-1 (H6) · phase: after HP-07 infra exists · needs: 1× A100-80GB (8B) or multi-GPU (70B), the LLMSymbMech repo, our repo

PROMPT:
Goal: test whether per-item strategy corresponds to causal engagement of the emergent symbolic pathway (symbol abstraction heads → symbolic induction heads → retrieval heads) identified by Yang et al. ICML 2025 (arXiv:2502.20332, code github.com/yukang123/LLMSymbMech).
Steps:
1. Reproduce their head-identification CMA on Llama-3.1-8B using their released pipeline and identity-rule tasks. If significant head sets emerge, record them; if not, document the null and fall back to their published 70B head lists with 70B inference (note cost).
2. Adapt the CMA harness to our items: for each (item) in the HP-07 pair set (canonical correct ∧ W3 wrong = dissociating; canonical correct ∧ W3 correct = control), compute per-item causal mediation scores for the three head classes, on canonical and on W3 runs. Define pathway occupancy O(item) = normalized mediation of abstraction+induction heads at the answer-relevant positions.
3. Analyses: (a) O(canonical) for dissociating vs control items (prediction: lower for dissociating); (b) ΔO = O(canonical)−O(W3) by group; (c) correlation of O with DS-01 class posterior, HP-13 intrusion flag, and Commitment Depth; (d) head-ablation spot check on 10 items per group: ablating abstraction heads should hurt control items' W3 performance more than dissociating items' (which never used them).
4. Report head-transfer honestly: our tasks are math/planning, theirs identity rules; if the head sets do not transfer, that is the finding — do not force it.
Output: occupancy CSV per item, 3 figures (occupancy by group, ΔO distribution, correlation matrix), H6_REPORT.md with an explicit verdict on H6 and whether AC-2 (repair) is warranted.
Validate: reproduce at least one headline number from their repo before adapting (pipeline sanity); permutation tests for head significance as in their method; all occupancy claims paired with control-item baselines; GPU hours and cost logged to costs.csv.
