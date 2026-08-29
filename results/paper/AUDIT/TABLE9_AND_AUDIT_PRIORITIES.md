# Table 9 / thresholds provenance

## Temperature caveat (priority)

Most published P1 / ALGO-P2 / GSM-P2 cells were produced by `OpenRouterClient`,
which **omits** `temperature` (provider default). Only BW P2 CCI/TEP force
`temperature=0.0` via `ModelClient`. Caveat any OpenRouter-backed accuracy cell
when claiming greedy/temp-0 behaviour (notably Llama ALGO P1). Full table:
`results/derived/temperature_payload_audit.csv`.

## Table 9 row: Liberal v2 sweep (param 204)

| Field | Value |
|-------|-------|
| Code | `scripts/runs/triangulation_v2.py` → `sweep_thresholds()` |
| Artifact | `results/derived/triangulation_threshold_sweep.csv` row `param_id=204` |
| Summary | `results/derived/triangulation_v2_summary.md` |
| Scope | **All families** with signals present (`n_rows=1332`), not ALGO-only 440 |
| Rates | retrieval **27.3%**, computation **30.4%**, strong **57.7%**, insufficient **37.9%**, mixed **0%** |

### Exact thresholds at param 204

Decoded from `itertools.product` order in `sweep_thresholds` (index 204):

| knob | value |
|------|-------|
| `w3_retrieval_max` | 0.2 |
| `w3_computation_min` | 0.5 |
| `contam_retrieval_min` | 0.6 |
| `contam_computation_max` | 0.5 |
| `cci_computation_min` | 0.6 |
| `cci_retrieval_max` | 0.25 |
| `min_votes` | **2** |
| `vote_margin` | **1** |

### Label rule (v2) — **not** three-signal conjunction

`apply_votes()` tallies independent binary votes:

- **Retrieval keys** (5): `p1_rename_fragile`, `p1_vri_high`, `p2_cci_retr`, `p2_impl_fail`, `p3_contam_high`
- **Computation keys** (8): `p1_w3_keep`, `p1_multi_variant`, `p2_cci_comp`, `p2_match_first`, `p2_crit_step`, `p2_impl_recovery`, `p3_contam_low`, `p3_depth_high`

Then:

```
insufficient  if retrieval_votes + computation_votes < min_votes
computation   if ~insufficient and (computation - retrieval) >= vote_margin
retrieval     if ~insufficient and (computation - retrieval) <= -vote_margin
weak_*        if margin-1
mixed         else
```

**Three-signal AND is not required.** With `vote_margin=1` and `min_votes=2`, any
net lean of ±1 after collecting ≥2 votes counts as a strong label — much
looser than conjunction.

Grid size: `3×2×3×3×3×3×2×2 = 1944` configs (not 270).

## Why it disagrees with the 270-config sweep

| | Liberal v2 param 204 | 270-config ALGO sensitivity |
|--|----------------------|-----------------------------|
| Script | `scripts/runs/triangulation_v2.py` | `scripts/runs/algo_triangulation_threshold_sensitivity.py` |
| Output | `triangulation_threshold_sweep.csv` | `ALGO_P3_threshold_sensitivity.csv` |
| Rule | **k-of-n vote** (disjunctive tallies) | **Legacy AND** from `ALGO_P3_SCR_triangulation.compute_convergence_labels` |
| Panel | multi-family (~1332) | ALGO only (**440**) |
| Strong rate | **57.7%** (max of v2 sweep) | **≤5.0%** (max of 270) |
| Grid | 1944 params | `18 CCI × 5 W3 × 3 contam% = 270` |

Legacy AND (paper default / Table 9 “Legacy strict”):

- retrieval: `VAR_can>0.5 AND VAR_W3<w3_cut AND contam_rank>pct AND greedy_succeeds`
- computation: `VAR_W3>w3_cut AND ACI>cci AND contam_rank<=pct`
- else mixed / ambiguous (~61.6% ambiguous on 440)

So Table 9’s two rows are **different labeling systems**, not two points on
the same sensitivity surface. Param 204 is the **maximum strong-rate** cell
of the v2 vote sweep; the 270-grid never approaches ~58% because conjunction
+ ALGO-only + high ambiguous mass caps strong labels near the paper’s ~2–5%.

## Step F gold audit (no GPU)

Artifact: `results/derived/mechanistic_gold_content_token_audit.csv` (398 Llama
mechanistic rows).

- Format keywords (`Path`/`Count`/`Selected`): **200/398 (50.3%)** — all ALGO
- Content already: BW/MBW/GSM (**198/398**)
