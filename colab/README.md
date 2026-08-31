# Colab notebooks

Notebooks only live here. Downloads from a Colab run go into `results/`, not this folder.

| Notebook | Put the download here |
|----------|------------------------|
| `llama_greedy_behavioural.ipynb` → `colab_out/llama_greedy_p1.csv` | `results/raw/llama_greedy_p1.csv` |
| `mechanistic_frequency_controlled.ipynb` → `colab_out/mech_freq_controlled.csv` | `results/raw/mechanistic_frequency_controlled.csv` (**GSM only — do not overwrite**) |
| same → `colab_out/mech_freq_controlled_algo_bw.csv` | `results/raw/mechanistic_frequency_controlled_algo_bw.csv` (ALGO 61 + BW 65; `family` column) |
| same → `mech_freq_controlled_algo_bw_summary.csv` | `results/derived/mechanistic_frequency_controlled_algo_bw_summary.csv` |
| same → `mech_freq_controlled_algo_bw_manifest.json` | `results/raw/mechanistic_frequency_controlled_algo_bw_manifest.json` |

Regenerate the `.ipynb` files with `python colab/_build_notebooks.py`.
