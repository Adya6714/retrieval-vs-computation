# D13 Continued-pretraining transfer (base vs Coder vs Math)

status: Track T, T2 · execution: [[HP-17_Qwen_Family_Contrast]] · hypothesis: H10 · serves: CC-1; natural-experiment precursor to AC-3

**Claim tested.** Domain continued pretraining changes *which surface the procedure is keyed to*, not just accuracy. Specifically: code-specialised pretraining removes or inverts the W4 formal-notation penalty (F2) without changing W3 cost.

**Why this contrast is clean.** Qwen2.5-1.5B-Instruct, Qwen2.5-Coder-1.5B-Instruct and Qwen2.5-Math-1.5B-Instruct share architecture, tokenizer and origin, and differ mainly in continued-pretraining data. Differences in the surface profile are attributable to that data far more cleanly than any cross-vendor comparison in Paper I.

**Relevance beyond the programme.** Anyone converting a general or multilingual model into a code or math model can use the battery to check whether the abstract procedure survived or only canonical-surface accuracy did.

**Design.** GSM + coin_change + shortest_path (where floor cleared); all variants; greedy; one flip-stable configuration from D11; identical prompt template across the three models.

**Predictions.**
- Math: highest Acc_can on GSM; W4 penalty smaller than Instruct.
- Coder: W4 penalty smallest (possibly inverted) on ALGO; W3 cost unchanged vs Instruct.
- If Coder also reduces W3 cost, the W3 effect is partly about formal abstraction, which bears on H2.

**Primary contrast (pre-register).** Interaction model × variant on correctness, logistic mixed model with (1|item); headline term Coder × W4.

**Cost.** T4 hours.
