# Results

**Flow:** `data/problems/*.csv` → run scripts → `raw/` → metric scripts → `derived/` → table/fig scripts → `paper/` + `figures/`

Only **`raw/`** is append-only (`--resume`). Recompute everything else after `raw/` changes.

Paths: `probes/common/results_paths.py`

---

## Artifact registry (tagged index)

**Canonical list:** [`ARTIFACT_REGISTRY.csv`](ARTIFACT_REGISTRY.csv) — every raw, derived, paper, and figure file that records question-bank evaluation output.

Update manually when adding new outputs, or regenerate with a small inventory script over `raw/`, `derived/`, `paper/`, and `figures/`.

### Question banks (inputs)

| Bank | Path |
|------|------|
| Blocksworld / planning | `data/problems/question_bank_bw.csv` |
| GSM arithmetic | `data/problems/question_bank_gsm.csv` |
| Algorithmic | `data/problems/question_bank_algo.csv` |

### Tags (`tag` column)

| Tag | Meaning |
|-----|---------|
| `bank_run` | Direct model run on a question bank (sweep, CCI, TEP, contamination, mechanistic, plans) → lives in `raw/` |
| `bank_run_aux` | Sidecar from a bank run (e.g. review queue) |
| `bank_run_meta` | Resume/progress state for sweeps (`sweep_progress.json`) |
| `derived_from_raw` | Aggregated or merged from `raw/` (metrics, triangulation, regressions) → `derived/` |
| `presentation` | Tables or plots for the paper → `paper/`, `results/figures/`, `paper/figures/scripts/probe/` |
| `presentation_aux` | Audit notes, not primary results |

### Registry columns

`path`, `layer` (`raw` \| `derived` \| `paper` \| `figures` \| `figures_root` \| `analysis_figures` \| `meta`), `family` (`BW` \| `GSM` \| `ALGO` \| `cross`), `probe`, `artifact_type`, `question_bank`, `producer_script`, `tag`, `present`, `format`, `data_rows`, `bytes`, `mtime_utc`.

**Not indexed:** `*.bak` and other ephemeral backups.

---

## Folders

| Folder | Role | You edit this? |
|--------|------|----------------|
| `raw/` | Per-run model outputs | No — produced by sweep scripts |
| `derived/` | Aggregated metrics (long format) | No — run `*_SCR_compute_metrics.py` |
| `paper/` | Wide tables for the manuscript | **Yes** — edit `make_table1.py` or CSV columns |
| `results/figures/` | PNG/PDF probe diagnostic plots | **Yes** — edit `*_FIG_generate.py` or `paper/figures/scripts/` |
---

## Pipeline: what creates what

### Probe 1 — behavioral (accuracy on variants)

| Script | Reads | Writes (`raw/`) |
|--------|-------|-------------------|
| `ALGO_P1_SCR_run_behavioral_sweep.py` | `question_bank_algo.csv` | `ALGO_P1_behavioral_{claude,gpt4o,llama}.csv`, `ALGO_P1_review_queue.csv` |
| `BW_P1_SCR_run_behavioral_sweep.py` | `question_bank_bw.csv` | `BW_P1_behavioral.csv` (all models) |
| `BW_P1_SCR_run_behavioral_sweep.py --family arithmetic_reasoning` | `question_bank_gsm.csv` | same pattern → use GSM bank path flag / GSM sweeps* |
| `colab/llama_greedy_behavioural.ipynb` | banks (canonical + W3) | `raw/llama_greedy_p1.csv` (local Llama-3.1-8B-Instruct, NF4, not OpenRouter) |

\*GSM P1 files: `GSM_P1_behavioral_{claude,gpt4o,llama}.csv` (same sweep script family filter).

| Then | Script | Writes (`derived/`) |
|------|--------|---------------------|
| ↓ | `ALGO_P1_SCR_compute_metrics.py` | `ALGO_P1_metrics.csv` |
| ↓ | `GSM_P1_SCR_compute_metrics.py` | `GSM_P1_metrics.csv` |
| ↓ | `BW_P1_SCR_compute_metrics.py` | `BW_P1_metrics.csv` |

### Probe 2 — plan execution

| Script | Reads | Writes |
|--------|-------|--------|
| `ALGO_P2_SCR_run_phase1.py` | bank + `--output` per model | `raw/ALGO_P2_phase1_{claude,gpt4o,llama}.csv` |
| `ALGO_P2_SCR_run_phase2.py` | bank + phase1 | `raw/ALGO_P2_phase2_normal.csv`, `ALGO_P2_phase2_injected.csv` |
| `GSM_P2_SCR_run_probe2.py` | `question_bank_gsm.csv` | `raw/GSM_P2_cci.csv` |
| `BW_P2_SCR_extract_phase1_plans.py` | `BW_P1_behavioral.csv` | `raw/BW_P2_plans.csv` |
| `BW_P2_SCR_run_cci.py` | bank + plans | `raw/BW_P2_cci.csv` |
| `BW_P2_SCR_run_tep.py` | bank + plans | `raw/BW_P2_tep.csv` |
| `ALGO_P2_SCR_compute_metrics.py` | P2 raw | `derived/ALGO_P2_metrics.csv` |
| `GSM_P2_SCR_compute_metrics.py` | `GSM_P2_cci.csv` | `derived/GSM_P2_metrics.csv` |

### Probe 3 — contamination & mechanistic

| Script | Writes (`raw/` or `derived/`) |
|--------|-------------------------------|
| `BW_P3_SCR_run_contamination_triage.py` | `raw/BW_P3_contamination.csv` |
| `GSM` / `ALGO` triage (family flag on BW script or dedicated) | `raw/GSM_P3_contamination.csv`, `raw/ALGO_P3_contamination.csv` |
| `run_mechanistic_sweep.py` | `raw/*_P3_mechanistic.csv` |
| `colab/mechanistic_frequency_controlled.ipynb` | `raw/mechanistic_frequency_controlled.csv` (+ `_manifest.json`); `derived/mechanistic_frequency_controlled_summary.csv` |
| `ALGO_P3_SCR_triangulation.py` | `derived/ALGO_P3_triangulation.csv` |
| `BW_P3_SCR_run_triangulation.py` | `derived/BW_P3_triangulation_{claude,gpt4o,llama}.csv` |

### Paper & figures (presentation layer)

| Script | Reads | Writes |
|--------|-------|--------|
| `consolidate/make_table1.py` | `raw/` P1+P2+P3, some `derived/` | `paper/TABLE1_cross_family.csv` |
| `consolidate/run_css_regressions.py` | `raw/` sweeps + contamination | `paper/cross_family_regression.csv` |
| `consolidate/run_paper_consolidation*.py` | `derived/` metrics | `paper/PAPER_TABLE1_consolidated_v2.csv` (optional) |
| `ALGO_P1_FIG_generate.py` | `derived/ALGO_P1_metrics.csv` (+ bank) | `figures/ALGO_P1_FIG_*.png/pdf` |
| `GSM_P1_FIG_generate.py`, `GSM_P2_FIG_generate.py`, `GSM_P3_FIG_generate.py` | `derived/` or `raw/` | `figures/GSM_*` |
| `ALGO_P2_FIG_generate.py`, `ALGO_P3_FIG_generate.py` | `derived/` / `paper/` | `figures/ALGO_*` |
| `BW_P2_SCR_generate_figures.py`, `BW_P3_FIG_probe1_triage_plot.py` | `raw/BW_P2_*`, contamination | `figures/BW_*` |
| `paper/figures/scripts/*.py` | various `raw/` + `derived/` | `paper/figures/` or `results/figures/` |

---

## `derived/` — metrics (long tables)

Each row: `model`, `metric_name` (or `metric`), `metric_value`, often `ci_lower` / `ci_upper`, sometimes `subtype` / `variant_type`.

### `ALGO_P1_metrics.csv` ← `ALGO_P1_SCR_compute_metrics.py`

| Metric | Meaning |
|--------|---------|
| `VAR` | Variant accuracy rate (by variant, subtype, model) |
| `VRI_structural`, `VRI_vocabulary`, `VRI_gap` | Structural vs vocabulary robustness (W1/W2/W4 vs W3) |
| `DTS_ALGO` | Difficulty transfer (standard vs adversarial instances) |
| `GSS` | Greedy-susceptibility gap (standard − adversarial accuracy) |
| `Formalism_Gap` | W4 vs prose gap |
| `CFS`, `HDR`, `VWC` | Contamination–stability diagnostics |

### `GSM_P1_metrics.csv` ← `GSM_P1_SCR_compute_metrics.py`

| Metric | Meaning |
|--------|---------|
| `VAR` | Accuracy by variant / contamination pole |
| `CSS` | Cross-variant stability (per problem) |
| `RAR_W5_accuracy` | W5-specific accuracy |
| `VRI_*`, `W6_gap` | Robustness gaps |
| `CFS` | Contamination fragility |

### `BW_P1_metrics.csv` ← `BW_P1_SCR_compute_metrics.py`

| Metric | Meaning |
|--------|---------|
| `VAR` | Accuracy by variant (blocksworld vs MBW) |
| `PDAS` | Procedural difficulty amplification (W5 − canonical) |

### `ALGO_P2_metrics.csv` ← `ALGO_P2_SCR_compute_metrics.py`

| Metric | Meaning |
|--------|---------|
| `ADC_*` | Algorithm description correctness |
| `CCI_*`, `TEP_*` | Plan–execution coupling |
| `CPP`, `FDI_*`, `SC`, `RDI_*`, `RTDA_*` | Phase-2 execution diagnostics |

### `GSM_P2_metrics.csv` ← `GSM_P2_SCR_compute_metrics.py`

| Metric | Meaning |
|--------|---------|
| `mean_cci_excluding_valid_divergence` | Mean CCI |
| `mean_tep` | Mean trace-edit penalty |
| `valid_divergence_rate` | Rate of valid divergences |
| `mean_cci_by_contamination_*` | CCI vs contamination pole |

### `ALGO_P3_triangulation.csv` ← `ALGO_P3_SCR_triangulation.py`

Per-problem merge of P1 behavioral + P2 + contamination labels (for analysis/regression).

---

## `paper/` vs `derived/`

| | `derived/` | `paper/` |
|---|------------|----------|
| **Shape** | Long (many rows per metric) | Wide (one row per family/subtype) |
| **Purpose** | Analysis, figure scripts, debugging | Main text tables |
| **Built by** | `*_compute_metrics.py`, triangulation | `make_table1.py`, consolidation scripts |

**Main table:** `paper/TABLE1_cross_family.csv` — columns like `VAR(canonical)`, `VAR(W3)`, `VRI_gap`, `PDAS/GSS`, `CCI_mean`, `TEP_mean`, `Contamination-CSS slope`.

`make_table1.py` recomputes many cells **directly from `raw/` sweeps** (not only from `derived/`), then joins P2 CCI/TEP and contamination. So changing a metric definition may require editing **both** the compute script and `make_table1.py`.

**Other `paper/` files:** `cross_family_regression.csv`, `PROBE2_consolidated.csv`, audit notes — safe to regenerate; edit scripts if you change formulas.

---

## Changing presentation

| Goal | Edit |
|------|------|
| Table columns / formatting | `scripts/consolidate/make_table1.py` → rerun `make_table1.py` |
| Regression summary table | `scripts/consolidate/run_css_regressions.py` |
| Plot style / which metric | `scripts/{ALGO,GSM,BW}_P*_FIG_generate.py` |
| Shared plot helpers | `paper/figures/scripts/probe/_common.py`, `paper/figures/scripts/gen_*.py` |
| Metric definitions (numbers) | `scripts/*_SCR_compute_metrics.py` → rerun → rerun fig/table scripts |

---

## Regenerate cheat sheet

```bash
# After updating raw/ (or adjudication queue)
python scripts/consolidate/apply_queue_adjudication.py   # ALGO only: patch raw + P1 metrics

python scripts/ALGO_P1_SCR_compute_metrics.py
python scripts/GSM_P1_SCR_compute_metrics.py
python scripts/BW_P1_SCR_compute_metrics.py
python scripts/ALGO_P2_SCR_compute_metrics.py
python scripts/GSM_P2_SCR_compute_metrics.py

python scripts/consolidate/make_table1.py
python scripts/consolidate/run_css_regressions.py

python scripts/ALGO_P1_FIG_generate.py   # + other *_FIG_generate.py as needed
```

---

## Naming

`{FAM}_P{probe}_{artifact}[_{model}].csv` — e.g. `ALGO_P1_behavioral_claude.csv`, `BW_P1_behavioral.csv`.

**Resume key:** `(problem_id, variant_type, model)` — skip if already in output file.
