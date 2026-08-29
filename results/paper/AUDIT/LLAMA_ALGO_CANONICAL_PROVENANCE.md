# Llama ALGO canonical accuracy — provenance note

## Published Table 7 cells (pkg8)

| Cell | Value | Arithmetic | Source |
|------|-------|------------|--------|
| ALGO SP-chall. × Llama × Can. | **.059** | **2/34** | `results/derived/ALGO_P1_4model_frozen_labels.csv` (`shortest_path`, `adversarial`, `canonical`) |
| ALGO SP-std. × Llama × Can. | **.048** | **1/21** | same file (`shortest_path`, `standard`, `canonical`) |

These are the **frozen / manually audited** 4-model labels the table caption cites — not a fresh re-aggregation of the raw OpenRouter CSV.

## OpenRouter Probe-1 raw aggregate (informational only)

`results/raw/ALGO_P1_behavioral_llama.csv`:
- Canonical overall **verified** ≈ **7/111 ≈ 6.3%** (all SP/CC/WIS).
- Produced via OpenRouter `meta-llama/llama-3.1-8b-instruct`.
- `probes/behavioral/openai_client.py` payload historically had **no `temperature` and no greedy/seed fields** — provider-default decoding, **not** forced-greedy.
- That OpenRouter wallet/key path is **dead** (402 / overdrawn); do not re-run or treat 7/111 as the forced-greedy floor.

## Forced-greedy floor (mechanistic gate)

Use GPU-local:
```bash
python3 scripts/algo_llama_greedy_accuracy.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --output results/raw/ALGO_llama31_8b_greedy_canonical.csv
```
(`do_sample=False`). Compare to content-gold final ranks via
`scripts/runs/mechanistic_contentgold_gate.py` (queued after the content-gold
Llama sweep in `launch_mechanistic_contentgold.sh`).
