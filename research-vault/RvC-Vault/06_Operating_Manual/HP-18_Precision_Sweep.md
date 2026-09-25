# HP-18 Precision sweep and per-band map (T3)
addresses: [[D12_Precision_Compression_Invariance]] · phase: Track T · needs: Colab T4, bitsandbytes, auto-gptq/optimum, autoawq; T1 + T2 done

PROMPT:
Goal: measure how weight precision changes accuracy vs invariance, globally and per band of layers.
Rules: same as HP-16. Pre-register the primary contrast (ΔR_W3 minus relative ΔAcc_can, cluster bootstrap) before running.
Steps:
1. Global arms per model from HP-17: fp16 (reuse T2 raws), int8 (bitsandbytes load_in_8bit), nf4 (bitsandbytes load_in_4bit, nf4, compute dtype fp16), gptq_int4 and awq_int4 (load published pre-quantized checkpoints if they exist for that exact model; if not, record "unavailable" and skip; do not quantize yourself in this HP).
2. Per-band arm: `scripts/trackT/fake_quant.py` implementing round-to-nearest symmetric INT4 with group size 128 on Linear weights of selected decoder blocks. Bands: split blocks into 4 contiguous bands. One run per band.
3. Items and variants: GSM + coin_change; canonical, W1, W3 (W3b now; add W3a after HP-23), W4, W6.
4. Raw: `results/raw/T3_P1_{model_slug}_{precision}.csv`, same schema as T2 plus a `precision` column.
5. Derived: `results/derived/T3_precision_metrics.csv` (Acc per variant, R_W3, R_W4, φ vs fp16 per variant, flip set sizes) and `T3_band_map.csv`.
Output: raws, derived, `docs/trackT/T3_REPORT.md` with tables and one heatmap (band × variant, ΔAcc) saved to `results/figures/T3_band_map.pdf`.
Validate: fp16 arm reproduces T2 exactly on the same config; memory usage logged per arm.
