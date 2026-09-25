# HP-23 Build W3a true nonce rename; relabel current W3 as W3b (T0)
addresses: [[EF-07_W3_Construct_Audit]] · phase: Track T (required Paper I fix) · needs: no GPU; small API budget or T4 for runs

PROMPT:
Goal: make the bank match the paper, and add the binding-vs-isomorph contrast.
Steps:
1. Relabel: in all three banks, add column `w3_kind` with value `isomorph` for every existing W3 row. Do not change `variant_type` values yet; add `variant_subtype` = `W3b`. Update the schema doc in `probes/common/io.py`.
2. Fix the substring bug: find every W3 row where replacement hit a substring (start with the 12 BW rows containing "You are Alice robot arm"); regenerate with whole-word, case-aware replacement; record before/after in `C3_corrections_changelog.md`.
3. Generate W3a for the canonical-correct subset (items with Acc_can ≥ .30 for at least two models): keep the canonical frame, verbs and numbers; replace every entity name with a nonce from a held-out list (pronounceable, not in any tokenizer's top-50k as a whole word, token length matched to the original within ±1 token under the Llama-3 and Qwen2.5 tokenizers). Persist the mapping in `notes` as JSON; verifier consumes it.
4. Gold-in-gold-out: every W3a gold goes through its verifier before any model call; failures excluded with reason.
5. Runs: the 5 API models if credits exist; otherwise the HP-17 open models on T4.
Output: updated banks, `results/raw/T0_W3a_*.csv`, `results/derived/T0_W3a_vs_W3b.csv`, `docs/trackT/T0_REPORT.md` tables only.
Validate: unit test asserting no W3a row shares a whole-word entity token with its canonical; unit test asserting the Alice pattern no longer occurs.
