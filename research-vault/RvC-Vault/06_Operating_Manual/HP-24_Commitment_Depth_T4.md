# HP-24 Commitment depth and invariance depth on a T4
addresses: [[D17_Commitment_Depth_T4]] · phase: Track T · needs: Colab T4, transformer_lens or manual hooks

PROMPT:
Goal: measure per-layer commitment depth and invariance depth on canonical vs W3, on models that actually solve enough items to make this interpretable.
Steps:
1. Load Qwen2.5-Math-1.5B-Instruct and Qwen2.5-1.5B-Instruct. Confirm both clear Acc_can ≥ .30 on GSM before proceeding; if not, stop and report.
2. `scripts/trackT/T7_logit_lens.py`: for each canonical-correct item and its W3 pair, run with hooks on every layer's residual stream, project through the unembedding, record the first layer where the gold answer token is top-1.
3. `scripts/trackT/T7_invariance_depth.py`: per layer, compute CKA (or RSA) between canonical and W3 hidden states across the item set; find the layer where similarity crosses a pre-registered threshold (e.g. 0.8).
4. Write `results/raw/T7_commitment_depth_{model_slug}.csv`, `results/derived/T7_metrics.csv`.
Output: raws, derived metrics, `docs/trackT/T7_REPORT.md`, tables only, no interpretation.
Validate: sign test on the primary contrast; report the exact p-value and n.
