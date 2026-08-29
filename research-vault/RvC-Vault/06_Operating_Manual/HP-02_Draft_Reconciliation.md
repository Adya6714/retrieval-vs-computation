# HP-02 — Reconcile Both Drafts to Raw Data
addresses: [[01_Program_State]] A1 · phase: 0 · needs: repo + both LaTeX sources

PROMPT:
Audit finding (verified 2026-07-07): the EMNLP-format draft's GSM Table 3 row for GPT-4o and Llama (n=40, canonical .825/.825, coverage described as GSM_001–020 + 041–060) does not reproduce from committed raws; the correct current numbers are n=20 (GSM_001–020), canonical .850 (gpt4o) / .800 (llama), W3 .300/.150. The CAISc draft is correct. GSM_021–040 are duplicate reruns, not a remap.
Steps:
1. In the EMNLP-format LaTeX: replace the GSM row values and the coverage description; add the partial-coverage footnote used by the CAISc draft (or the n=44 numbers if HP-01 has landed — check GSM_GAPFILL_REPORT.md first).
2. Grep both drafts for every occurrence of ".825", "n=40", "041–060" in GSM contexts; fix all.
3. Recheck both drafts' Appendix F mechanistic description states: Qwen2.5-7B **base**, n=398 rows, file mechanistic_sweep_7b_base_rawqa.csv; no claim may cite the 0.5B files (75 exploratory rows) or the 7B-Instruct files (268 rows) — see [[HP-03_Mechanistic_Provenance_Ledger]].
Output: patched LaTeX + a diff summary listing every changed number with old→new and the raw-CSV justification.
Validate: every quantitative claim touched must trace to a named CSV + row filter; if any cannot, flag it in the diff summary rather than guessing.
