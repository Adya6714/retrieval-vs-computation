# CI pending (GitHub issue draft)

Headline numbers in Findings F1–F3 are sourced from CSVs that do not carry interval columns for those point estimates. Per SITE_REDESIGN_SPEC §4, do not invent CIs; mark **CI pending** on the site until a derive script emits them.

## Issue title
Add Wilson / bootstrap CIs for F1–F3 headline accuracies on the programme site

## Body

### F1 — Numeric vs lexical (`#f1`)
| Number on site | Source CSV | Why pending |
|---|---|---|
| Gemini GSM Acc_can `.909` | `results/derived/probe1_per_model_variant.csv` | Columns: `accuracy` only (no `ci_low` / `ci_high`) |
| Gemini GSM Acc_W6 `.958` | same | same (also `n_valid=24` for W6 vs 44 for can/W3) |
| Gemini GSM Acc_W3 `.523` | same | same |

Related file with CIs for W6 only (different n / family rows): `results/derived/P1_w6_accuracy.csv` — **not** used for the F1 headline triplet because it does not match the caption’s can/W6/W3 triple on the same GSM n.

### F2 — Formal notation (`#f2`)
| Number on site | Source CSV | Why pending |
|---|---|---|
| o4-mini GSM `.841 → .682` (canonical → W4) | `results/derived/probe1_per_model_variant.csv` | No CI columns |

### F3 — Direction (`#f3`)
| Number on site | Source CSV | Why pending |
|---|---|---|
| Claude BW `.172 → .873` | Captioned against `C7_concordance_filtered.csv` | C7 has per-variant accuracies for rankers but no CI columns for those Acc point estimates |
| Gemini `.391 → .764`; o4-mini `.781 → .909` | same family of Probe 1 / concordance tables | No CI columns for the headline Acc pair |

### Already have CIs (no action)
- F5 pooled CCI × W3 retention: `results/derived/P2_P1_convergence.csv` (`ci_low`, `ci_high`) — shown as `[.027, .414]` on the site.

### Proposed fix
Extend `scripts/runs/rederive_all_metrics.py` (or a small `scripts/site/build_finding_cis.py`) to emit Wilson or cluster-bootstrap intervals beside Acc for the F1–F3 headline cells, write `results/derived/finding_headline_cis.csv`, and wire the site captions to that file.

## Labels
`site`, `stats`, `good first issue`
