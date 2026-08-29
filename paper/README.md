# Same Score, Different Strategy — CAISc 2026

Accepted paper package. Source: `main.tex`, `appendix.tex`, `tables/`, `figures/`.

## Build

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Submission build (default): line numbers + footer via `\usepackage{caisc_2026}`.  
Camera-ready: `\usepackage[final]{caisc_2026}` in `main.tex`.

## Regenerate figures

| Script | Output |
|--------|--------|
| `figures/scripts/gen_figures.py` | Main paper PDFs (`fig_robustness`, `fig_decay`, `fig_cci`, `fig_paradox`, …) |
| `figures/scripts/gen_more_figures.py` | Appendix panels (`fig_bw_inversion`, `fig_subtype_grid`, …) |
| `figures/scripts/gen_new_figures.py` | Supplementary panels |
| `figures/scripts/gen_corr_figure.py` | `fig_corr_matrix.pdf` |
| `figures/scripts/probe/` | Probe-level diagnostic plots → `results/figures/probe/` |
| `figures/scripts/legacy/` | Earlier figure drafts (reference only) |

Run from repo root with `PYTHONPATH=.`:

```bash
python paper/figures/scripts/gen_figures.py
python paper/figures/scripts/gen_more_figures.py
```

Family-specific figure generators live under `scripts/*_FIG_generate.py` and write to `results/figures/`.
