# Same Score, Different Strategy — NeurIPS 2026

Evaluations & Datasets Track. Source: `main.tex`, `appendix.tex`, `tables/`, `figures/`.

CAISc 2026 archive (style + prior tex): [`venue/caisc2026/`](venue/caisc2026/).  
NeurIPS style kit: [`venue/neurips2026/`](venue/neurips2026/).

## Build

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Submission: `\usepackage[eandd]{neurips_2026}`  
Camera-ready: `\usepackage[eandd, final]{neurips_2026}`  
Preprint: `\usepackage[preprint]{neurips_2026}`

Mandatory checklist: `neurips_checklist.tex` (input at end of appendix).

## Frozen numbers

All Table-7 / proximity / triangulation headline numbers should match
`../rebuild/NUMBERS.csv`. Errata: [`../docs/paper/PAPER_ERRATA.md`](../docs/paper/PAPER_ERRATA.md).

## Regenerate figures / Table 7

| Script | Output |
|--------|--------|
| `figures/scripts/gen_figures.py` | Main paper PDFs (`fig_robustness`, `fig_cci`, …) |
| `figures/scripts/gen_more_figures.py` | Appendix panels |
| `figures/scripts/gen_rebuild_figures.py` | Intrusion, complete-case triangulation, mixed failure, rule crosstab, coverage holes, crossover |
| `figures/scripts/regen_table7_from_numbers.py` | `tables/table7_pervariant.tex` from `rebuild/NUMBERS.csv` |

```bash
python paper/figures/scripts/gen_rebuild_figures.py
python paper/figures/scripts/regen_table7_from_numbers.py
```
