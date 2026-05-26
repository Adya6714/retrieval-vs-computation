# Paper audit bundle

Generated artifacts for `rvc_paper_v2.tex` revision (paste when the `.tex` file is available).

## Scripts (`scripts/audit/`)

| Script | Purpose |
|--------|---------|
| `regenerate_paper_tables.py` | Table 1, Table 3, Table 5 (3-model) |
| `five_model_var.py` | 5-model per-problem VAR + `*_VAR_5model.csv` |
| `o4mini_partial.py` | Honest o4-mini GSM partial metrics |
| `robustness_scatter.py` | Canonical vs W3 retention scatter |
| `appendix_c_5model.py` | Appendix C CSV + `.tex` snippet |
| `contamination_vri.py` | §5.6 contamination–VRI correlations |

## Key outputs

- `table1_gsm.csv`, `table1_algo_adversarial.csv`
- `appendix_c_full_var_5model.csv`, `appendix_c_full_var_5model.tex`
- `contamination_vri_algo_adversarial.csv` — Claude r=.44, GPT-4o r=.36 (n=61)
- `o4mini_gsm_canonical_w3_comparison.csv`
- `robustness_scatter.pdf` / `.png` (in `results/paper/`)
- `../robustness_scatter_data.csv`

## Deferred (needs `.tex`)

Steps 1–6 from paper revision: Table 1 paste, §5.6 text, cut mechanistic §6, o4-mini §5.X, abstract reframe, Llama scoping.
