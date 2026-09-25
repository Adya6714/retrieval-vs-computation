# Colab notebooks

Notebooks only live here. Downloads from a Colab run go into `results/raw/` or
`results/derived/`, **never** the repo root. Mis-landed root dumps are archived
at `results/raw/colab_inbox/` (kept for provenance; re-land under the paths below if you need them as canonical raw).

| Notebook | Put the download here |
|----------|------------------------|
| `llama_greedy_behavioural.ipynb` → `colab_out/llama_greedy_p1.csv` | `results/raw/llama_greedy_p1.csv` |
| H6 cell in the same notebook → `colab_out/llama_greedy_p1_gsm_canonical_768.csv` | `results/raw/llama_greedy_p1_gsm_canonical_768.csv` (**new file; never overwrite `llama_greedy_p1.csv`**) |
| `mechanistic_frequency_controlled.ipynb` → `colab_out/mech_freq_controlled.csv` | `results/raw/mechanistic_frequency_controlled.csv` (**GSM only — do not overwrite**) |
| same → `colab_out/mech_freq_controlled_algo_bw.csv` | `results/raw/mechanistic_frequency_controlled_algo_bw.csv` (ALGO 61 + BW 65; `family` column) |
| same → `mech_freq_controlled_algo_bw_summary.csv` | `results/derived/mechanistic_frequency_controlled_algo_bw_summary.csv` |
| same → `mech_freq_controlled_algo_bw_manifest.json` | `results/raw/mechanistic_frequency_controlled_algo_bw_manifest.json` |
| `o5_teacher_forced_likelihood.ipynb` → `colab_out/O5_teacher_forced_likelihood.csv` | `results/raw/O5_teacher_forced_likelihood.csv` (full P1 grid × 3 models; no aggregates — O10) |
| `o6_quantization_sensitivity.ipynb` → `colab_out/O6_quantization_sensitivity.csv` | `results/derived/O6_quantization_sensitivity.csv` (fp16/int8/nf4 pairwise bounds) |
| same → `O6_quantization_sensitivity_items.csv` | `results/raw/O6_quantization_sensitivity_items.csv` |
| same → `O6_quantization_sensitivity_summary.txt` | `results/derived/O6_quantization_sensitivity_summary.txt` |
| `o7_gsm_degeneracy_check.ipynb` → `colab_out/O7_gsm_degeneracy_check.csv` | `results/derived/O7_gsm_degeneracy_check.csv` (GSM mechanistic gate) |
| same → `O7_gsm_degeneracy_items.csv` | `results/raw/O7_gsm_degeneracy_items.csv` |
| same → `O7_gsm_degeneracy_verdict.txt` | `results/derived/O7_gsm_degeneracy_verdict.txt` |
| `o8_mech_behavior_link.ipynb` → `colab_out/O8_mech_behavior_link.csv` | `results/raw/O8_mech_behavior_link.csv` (per instance × layer) |
| same → `O8_layer_profile.csv` | `results/derived/O8_layer_profile.csv` |
| same → `O8_w3_binary_scores.csv` | `results/raw/O8_w3_binary_scores.csv` |
| same → `O8_framing.txt` | `results/derived/O8_framing.txt` |
| `o15_surprisal_contamination.ipynb` → `colab_out/O15_surprisal_contamination.csv` | `results/raw/O15_surprisal_contamination.csv` (problem-statement NLL + min-k%) |
| then `python scripts/consolidate/o15_surprisal_vs_infinigram.py` | `results/derived/O15_surprisal_contamination.csv` (adds length residuals) + `results/derived/O15_surprisal_vs_infinigram.csv` |
| `o16_open_model_calibration.ipynb` → `colab_out/O16_open_model_scores.csv` | `results/raw/O16_open_model_scores.csv` (Pythia/OLMo O5+O15 on canonicals) |
| `python scripts/consolidate/o16_corpus_ground_truth.py` (Cursor, no GPU) | `results/derived/O16_corpus_ground_truth.csv` |
| `python scripts/consolidate/o16_calibrate_proxies.py` | `O16_proxy_calibration.csv` + `O16_groundtruth_retention_test.csv` |
| `o14b_naming_likelihood.ipynb` → `colab_out/O14b_naming_likelihood.csv` | `results/raw/O14b_naming_likelihood.csv` (Qwen 1.5B/3B teacher-forced O14 bank) |
| same → `O14b_naming_analysis.csv` | `results/derived/O14b_naming_analysis.csv` |
| `ds16_recognition_recall.ipynb` → `colab_out/DS16_recognition_recall.csv` | `results/derived/DS16_recognition_recall.csv` |
| same → `DS16_gap_correlations.csv` | `results/derived/DS16_gap_correlations.csv` |
| **`k_suite_colab.ipynb`** (K1–K7 combined) | Same filenames as above — each arm has its own Download cell; land files exactly as for the standalone notebooks |

**Combined suite:** `colab/k_suite_colab.ipynb` runs K1→K2→K3→K4→K6→K7→K5 in one T4 session. Toggle arms with `RUN_K*` in the master knobs cell. Smoke with `LIMIT=2`, `DRY_RUN=True` first.

| **`T1_noise_floor.ipynb`** (HP-16 / Track T) | `results/raw/T1_noise_floor_*.csv` → metrics in `results/derived/T1_noise_floor_metrics.csv` |
| **`T2_qwen_family.ipynb`** (HP-17 / Track T) | `results/raw/T2_P1_*.csv` → `results/derived/T2_mixed_model.csv` |
| **`T3_precision.ipynb`** (HP-18 / Track T) | `results/raw/T3_P1_*_{precision,band*}.csv` → `T3_precision_metrics.csv`, `T3_band_map.csv`, `results/figures/T3_band_map.pdf` |

**Track T secrets:** Colab → 🔑 `RVC_PUSH_TOKEN` (preferred) or `GITHUB_TOKEN` for the push cell; never hard-code. Optional `HF_TOKEN`. First cell prints GPU / CUDA / torch / transformers. Scripts live under `scripts/trackT/` and always take `--resume`. Smoke with `DRY_RUN = True` in the run cell before a full T4 session.

Regenerate Track T notebooks with `python colab/_build_trackT.py`.

Regenerate the `.ipynb` files with `python colab/_build_notebooks.py` (includes O15/O16/O14b/DS16/k_suite via sibling builders).
