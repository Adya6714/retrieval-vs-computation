# HP-01 — GSM Gap-Fill and Canonical Number Freeze
addresses: [[01_Program_State]] A1/A4 · phase: 0 · needs: OpenRouter/API keys, repo

PROMPT:
You are completing coverage for the retrieval-vs-computation repo (github.com/Adya6714/retrieval-vs-computation). Verified state: `results/raw/GSM_P1_behavioral_{gpt4o,llama}.csv` contain valid canonical+variant rows only for GSM_001–020; all GSM_041–064 rows are `ERROR:` strings (24 per model); GSM_021–040 rows are duplicate reruns of 001–020 and are off-bank (bank = data/problems/question_bank_gsm.csv, canonical n=44: 001–020, 041–064).
Steps:
1. Locate the P1 sweep script used for GSM (check scripts/ and docs/evaluation/MASTER_EVALUATION_PIPELINES.md for the runner and its resume flag). Re-run GSM_041–064, all variants W1–W6 + canonical, for gpt-4o and llama-3.1-8b-instruct via the same provider settings (T=0, zero-shot CoT, same max_tokens).
2. Append to the raw CSVs without touching existing rows; rerun the derived-table build (probe1 per-model-variant + CSS/VRI scripts).
3. Recompute the Table 3 GSM row for both models at n=44. Also recompute at n=20 (001–020) to show continuity with the CAISc numbers (expected: 0.850 gpt4o / 0.800 llama canonical; 0.300 / 0.150 W3).
4. Optional add-on if budget allows: complete ALGO P2A-elicited for claude/gemini/llama from 61/110 to 110/110.
Output: updated CSVs; a markdown report GSM_GAPFILL_REPORT.md with before/after coverage counts, new accuracies with Wilson CIs, and any rows that still ERROR (list IDs).
Validate: in-bank canonical valid count per model must be 44 (or report which IDs still fail); no duplicate (problem_id, variant_type, model) rows; derived tables regenerate deterministically.
