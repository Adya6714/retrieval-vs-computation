# HP-17 Base vs Coder vs Math contrast (T2)
addresses: [[D13_Continued_Pretraining_Transfer]] · phase: Track T · needs: Colab T4; T1 done (use its most stable config)

PROMPT:
Goal: run the Probe 1 surface battery on three same-family open models that differ only in continued pretraining.
Rules: same as HP-16. Pre-register the primary contrast (Coder × W4 interaction term) in PREREGISTRATION.md section "Track T" BEFORE the first run; commit hash recorded in the report.
Steps:
1. Generalise `scripts/trackT/run_local_p1.py` (new) to load any HF causal LM at FP16 with the T1-stable config.
2. Models: Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-Coder-1.5B-Instruct, Qwen/Qwen2.5-Math-1.5B-Instruct. Same chat template call path for all.
3. Items: all GSM variants; ALGO coin_change and shortest_path variants.
4. Write `results/raw/T2_P1_{model_slug}.csv` with the same columns as the API P1 raws so existing metric scripts run unchanged.
5. Run existing `*_P1_SCR_compute_metrics.py` against these raws (add a `--raw-glob` arg if needed; do not fork metric logic).
6. Fit the pre-registered logistic mixed model (statsmodels BinomialBayesMixedGLM or R lme4 via rpy2 if available) and write coefficients + CIs to `results/derived/T2_mixed_model.csv`.
Output: raws, derived metrics, mixed-model table, `docs/trackT/T2_REPORT.md` (tables only).
Validate: Acc_can per model per family; mark any cell below the .30 floor as suppressed exactly as Paper I does.
