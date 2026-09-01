# Results

**Flow:** `data/problems/*.csv` → run scripts → `raw/` → metric scripts → `derived/` → table/fig scripts → `paper/` + `figures/`

Only **`raw/`** is append-only (`--resume`). Recompute everything else after `raw/` changes.

Instrument exclusions (`derived/variant_exclusions.csv`, reason `variant_not_transformed`): all ALGO and BW W6 rows (not a fresh-instance control as generated); BW W5 `MBW_496`–`MBW_500` (identical to canonical). GSM W6 stays (23/24 transformed). Filter via `probes.common.exclusions.filter_excluded`. Do not regenerate W6.

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
| `ALGO_P2_SCR_run_phase1.py` | bank + `--output` per model | `raw/ALGO_P2_phase1_{claude,gpt4o,llama}_new.csv` (110 rows; authoritative). The unsuffixed `ALGO_P2_phase1_{gpt4o,llama}.csv` files are 20-row pilots and are not inputs to analysis. Gemini: `raw/ALGO_P2_phase1_gemini.csv`. |
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

## Probe 2 — which Phase 1 file is authoritative

`results/raw/ALGO_P2_phase1_{gpt4o,llama}.csv` are **20-row pilots** (20 problems × 1 model).
`results/raw/ALGO_P2_phase1_{claude,gpt4o,llama}_new.csv` are the **110-row full elicitations**.
`results/raw/ALGO_P2_phase1_gemini.csv` is already 110 rows and has no `_new` sibling.

Downstream code must read only the 110-row set via `probes.common.results_paths.algo_p2_phase1_files()`.
Do not concatenate the pilots with the `_new` files (a keep=last overlay hid the duplication but made the pilots look live).

Paper-ready coverage: `results/derived/P2_coverage.csv`.
BW CCI/TEP null diagnosis: `results/derived/P2_bw_cci_null_diagnosis.csv`, `P2_bw_tep_null_diagnosis.csv`.

**GSM `session_b_correct` is a disjunction**, not Phase 2A accuracy: `verify(phase2a_values[-1]) OR verify(phase1_final)`. It never reads `phase2b_values`. Overlay: `derived/GSM_P2_session_correct.csv` (`either_session_correct`, `phase1_correct`; `phase2a_correct` / `phase2b_correct` blank — those value lists were never persisted). Table 4 Acc_P2A cannot be recovered without a re-run. Do not rewrite `raw/` GSM P2 files.

---

## Probe 3 — no template-vs-instance split; n-gram windows are not comparable across families

The published template-vs-instance claim is **withdrawn**. It is not a measurement of templates versus instances.

**ALGO** `template_contamination_score` in `raw/ALGO_P3_contamination.csv` rescores the **same full `problem_text`** with `family="gsm"` (max n=8 instead of n=13). `instance_contamination_score` is the **gold answer string**, not instance parameters or graph text. These columns are miswired. Do not plot them against each other as template vs instance. Flagged figures: `scripts/ALGO_P1_FIG_generate.py` (`plot_contamination_scatter`), `paper/figures/scripts/gen_figures.py` (`contam_vri_pearson`), rebuild `P3.1.*.template_contamination_score_vs_VRI` vs `P3.1.*.instance_contamination_score_vs_VRI`, `audit_2026_08/new/run_new_analyses.py` instance-vs-template Pearson, `paper/figures/scripts/legacy/fig2_contam_gradient.py`. See `results/derived/H3_template_instance_flags.csv`.

**GSM** has only `contamination_score` on the full problem text (`raw/GSM_P3_contamination.csv`). There is no template/instance split.

**BW** `template_contamination_score` in `raw/BW_P3_contamination.csv` is 0.000 on all 65 rows (original InfiniGram query was a keyword dump). Grammatical stems now live in `scripts/BW_P3_SCR_run_contamination_triage.py`; `raw/` is unchanged. Until a re-query lands, drop BW `template_contamination_score` from every correlation (zero variance). Instance fallback `blocksworld num_blocks N` is still a keyword dump.

**Cross-family contamination comparisons are invalid as computed.** GSM uses max n=8 (`family=arithmetic_reasoning`); ALGO/BW `contamination_score` uses max n=13. `max_ngram_count` is the **raw** Infini-gram count at the longest n with count>0, not a normalized rate. Within-family comparisons stay valid. Construction audit: `results/derived/P3_infinigram_query_audit.csv`.

Triangulation rule comparison (appendix printed vs executed, plus P1-rescored overlay): `results/derived/P3_triangulation_rule_comparison.csv`.

---

## Clone audit (canonical banks)

`derived/bank_clone_audit.csv`: near-duplicate families from token Jaccard ≥ 0.85 or SequenceMatcher ≥ 0.90 **and** identical gold, within each bank.

| Family | canonical n | clone families | problems in clones | effective n |
|--------|-------------|----------------|--------------------|-------------|
| ALGO | 110 | 14 | 73 | **51** |
| BW | 65 | 0 | 0 | 65 |
| GSM | 44 | 0 | 0 | 44 |

ALGO effective n is materially below 110. Every ALGO accuracy (Table 7 slices, Probe 1 figures, rebuild P1.1.ALGO.*) is still computed on 110 IDs but is **not 110 independent items**. WIS_017–020 sit in `ALGO_CLONE_013` (n=12) with other chain-overlap clones sharing `Selected: {4, 5}`.

---

## `derived/` — metrics (long tables)

Canonical Probe 1 accuracies after verifier repair: `derived/P1_rescore_summary.csv` and `derived/*_rescored.csv` (denominators are `included=True` only).

The older `{ALGO,BW,GSM}_P1_metrics.csv` files skipped Gemini and o4-mini by default and reused the name CSS for mean variant accuracy. They live in `results/deprecated/` with a README.

Each row of a family metrics file: `model`, `metric_name` (or `metric`), `metric_value`, often `ci_lower` / `ci_upper`, sometimes `subtype` / `variant_type`.

### `deprecated/ALGO_P1_metrics.csv` ← `ALGO_P1_SCR_compute_metrics.py` (quarantined)

| Metric | Meaning |
|--------|---------|
| `VAR` | Variant accuracy rate (by variant, subtype, model) |
| `VRI_structural`, `VRI_vocabulary`, `VRI_gap` | Structural vs vocabulary robustness (W1/W2/W4 vs W3) |
| `DTS_ALGO` | Difficulty transfer (standard vs adversarial instances) |
| `GSS` | Greedy-susceptibility gap (standard − adversarial accuracy) |
| `Formalism_Gap` | W4 vs prose gap |
| `CFS`, `HDR`, `VWC` | Contamination–stability diagnostics |

### `deprecated/GSM_P1_metrics.csv` ← `GSM_P1_SCR_compute_metrics.py` (quarantined)

| Metric | Meaning |
|--------|---------|
| `VAR` | Accuracy by variant / contamination pole |
| `CSS` | Cross-variant stability (per problem) |
| `RAR_W5_accuracy` | W5-specific accuracy |
| `VRI_*`, `W6_gap` | Robustness gaps |
| `CFS` | Contamination fragility |

### `deprecated/BW_P1_metrics.csv` ← `BW_P1_SCR_compute_metrics.py` (quarantined)

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
