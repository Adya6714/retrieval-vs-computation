# G. Intrusion errors

Among W3 **errors**, count cases where the model’s answer equals the **canonical** gold (the pre-rename answer), and does **not** equal the W3 gold.

GSM W3 preserves the numeric gold (name substitution only), so matching the canonical number while being a W3 error is almost impossible unless the verifier and the extractor disagree. ALGO W3 often relabels nodes/items (0,1,2 → Hub A,B,C / Item A,B,C); producing the numeric/canonical identifiers on a renamed instance is the intrusion.

| family | model | W3 errors | intrusions | rate | Wilson 95% CI |
|---|---|---:|---:|---:|---|
| ALGO | Claude | 100 | 2 | 0.020 | [0.006, 0.070] |
| ALGO | GPT-4o | 91 | 2 | 0.022 | [0.006, 0.077] |
| ALGO | Gemini | 82 | 4 | 0.049 | [0.019, 0.119] |
| ALGO | Llama | 108 | 0 | 0.000 | [0.000, 0.034] |
| ALGO | o4-mini | 43 | 5 | 0.116 | [0.051, 0.245] |
| BW | Claude | 56 | 0 | 0.000 | [0.000, 0.064] |
| BW | GPT-4o | 54 | 0 | 0.000 | [0.000, 0.066] |
| BW | Gemini | 58 | 0 | 0.000 | [0.000, 0.062] |
| BW | Llama | 58 | 0 | 0.000 | [0.000, 0.062] |
| BW | o4-mini | 53 | 0 | 0.000 | [0.000, 0.068] |
| GSM | Claude | 11 | 0 | 0.000 | [0.000, 0.259] |
| GSM | GPT-4o | 14 | 0 | 0.000 | [0.000, 0.215] |
| GSM | Gemini | 21 | 0 | 0.000 | [0.000, 0.155] |
| GSM | Llama | 17 | 0 | 0.000 | [0.000, 0.184] |
| GSM | o4-mini | 7 | 0 | 0.000 | [0.000, 0.354] |

**ALGO pooled:** 13/424 W3 errors are canonical-answer intrusions.
**GSM pooled:** 0 intrusions (expected near zero because W3 gold = canonical gold).

Five example rows per model are in `intrusion_examples.csv` (`W3_model_answer` truncated to 500 chars; `match_span` is the Path:/Selected: line that actually matched). Rows with `intrusion=False` are fillers when a model had fewer than 5 true hits. The matching line is often at the **end** of a long CoT, so the truncated `W3_model_answer` alone can look like a false positive.

**Flags:** GSM bank filter applied (GPT-4o/Llama n_valid=20). BW uses `BW_`/`MBW_` IDs from the three P1 files; GSM rows that leaked into `BW_P1_behavioral.csv` are dropped. Answer equality is structured (path / coin multiset / selected set / last number / action list), not raw-string equality — Claude W3 errors never matched canonical GT as a raw string.
