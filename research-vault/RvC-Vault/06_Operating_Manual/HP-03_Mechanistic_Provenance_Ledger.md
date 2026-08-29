# HP-03 — Mechanistic Provenance Ledger + Base-vs-Instruct Analysis
addresses: [[01_Program_State]] A2, D open weakness · phase: 0 · needs: repo, pandas; no API/GPU

PROMPT:
Four mechanistic artifact sets exist in results/raw/: (1) mechanistic_sweep_7b_base_rawqa.csv — Qwen/Qwen2.5-7B base, 398 rows, has target_rank_per_layer → backs paper Appendix F. (2) mechanistic_sweep_7b.csv and (3) mechanistic_sweep_7b_rawprompt.csv — Qwen2.5-7B-Instruct, 268 rows each, currently unused. (4) {ALGO,GSM,BW}_P3_mechanistic.csv — Qwen2.5-0.5B-Instruct, 75 rows total, exploratory.
Steps:
1. Write docs/MECHANISTIC_PROVENANCE.md: for each file — model string, rows, columns, prompt format, which draft claims (if any) it supports, and a DO-NOT-CITE flag for (4).
2. Analysis (new): compare (1) vs (2)/(3) on shared items — layerwise gold-token rank trajectories, the late-layer split, and earliest-entry layer (Commitment Depth, k=100). Question: does instruction tuning move commitment earlier, and does the canonical-vs-W6 gap persist post-instruct?
3. Produce one figure (rank vs layer, base vs instruct, canonical vs W6) + a results table; append findings to the ledger with verified-raw tags.
Output: docs/MECHANISTIC_PROVENANCE.md + notebook/script + figure.
Validate: row counts must match (398/268/268/75); if column schemas differ, document the mapping used; no claims beyond the two models actually present.
