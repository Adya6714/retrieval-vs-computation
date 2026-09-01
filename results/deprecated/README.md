# Deprecated derived artifacts

These files look authoritative but are known-wrong. They are kept for provenance, not for paper numbers.

## `{ALGO,BW,GSM}_P1_metrics.csv`

Moved from `results/derived/`.

- Default compute scripts read only Claude / GPT-4o / Llama (`ALGO_P1_SCR_compute_metrics.py` default `--sweep-results` omits Gemini and o4-mini; BW/GSM sweeps used the 3-model files).
- Bank filters were skipped or incomplete relative to the current question banks.
- Do not use these for accuracy denominators. Use `results/derived/P1_rescore_summary.csv` and `results/derived/*_rescored.csv` (`included=True` only).

## `compute_p1_metrics_unified.py`

Moved from `scripts/`. It names a column **CSS** for mean per-variant accuracy (`VAR_mean`). That is a different formula than `probes/behavioral/css.py` (fraction of variants matching the canonical answer, W5 undefined). Using both files as "CSS" is undefined.

Canonical CSS: `probes/behavioral/css.py`.
Canonical VRI: `mean(W1, W2, W4) − W3` in `css.compute_vri` and `rebuild/compute_rebuild.py`. The paper numbers came from the rebuild formula.
