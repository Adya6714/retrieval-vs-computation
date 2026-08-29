# HP-08 — Commitment Depth: Definition, Replication, Validation
addresses: [[D08_Commitment_Depth]] · phase: 1 ride-along · needs: same GPU session as HP-07

PROMPT:
Define CD_k(item, model) = earliest layer where the gold answer's first token enters the top-k of the unembedding projection (logit lens) of the residual stream at the answer position; +∞ if never.
Steps:
1. Reproduce on existing data: recompute from mechanistic_sweep_7b_base_rawqa.csv (target_rank_per_layer column) at k∈{10,50,100}; confirm the informal r≈+0.82 with final rank; report per-k stability.
2. Replicate on Llama-3.1-8B over the HP-07 item set (canonical + W3 runs): distributions for dissociating vs control items; prediction — retrieval-like items commit earlier on canonical and lose commitment entirely on W3.
3. Robustness: answer-position vs final-token readout; tuned-lens vs logit-lens if time permits (report divergence).
4. Hand the per-item CD values to HP-04's MTMM matrix (mechanistic method column) and to HP-06 (seen vs unseen comparison).
Output: cd.py (reusable), CD_REPORT.md with distributions + correlations, per-item CSV.
Validate: k-sensitivity must be reported; any claim of "early commitment" needs the control-item baseline in the same figure.
