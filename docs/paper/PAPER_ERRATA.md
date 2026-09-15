# Paper errata & frozen number authority

**Venue build:** NeurIPS 2026 Evaluations & Datasets (`paper/main.tex` + `neurips_2026.sty`).  
**Archive:** CAISc 2026 sources under `paper/venue/caisc2026/`.

## Number authority

| Source | Role |
|--------|------|
| `rebuild/NUMBERS.csv` | Frozen recomputed numbers under `rebuild/FROZEN_FILTERS.md` |
| `rebuild/REBUILD_REPORT.md` | Headline tables + paper deltas |
| `rebuild/solidify/` | Complete-case triangulation, intrusion, crossover |
| `rebuild/tri_v2/` | Executed vs appendix rule comparison |
| `results/paper/NUMBERS_FROZEN.csv` | Copy of `rebuild/NUMBERS.csv` for manuscript packaging |

`ANALYSIS.md` (May snapshot) is **historical**; do not treat it as number SoT.

## Corrections applied in the NeurIPS paper

- Proximity adversarial pool **n=61** (not 64); o4-mini instance $r \approx -0.171$.
- GPT-4o empty/diverged Acc **0.706 / 0.667** (was 0.69 / 0.73).
- NL-tolerant BW Probe-2 covers **3 models**, not five.
- Table 7 regenerated from `NUMBERS.csv` (SP-chall Claude $.647{\to}.000$ under W3 matches intro).
- Triangulation: executed 5-field AND is the published 8/4/157/271 generator; appendix three-signal is sensitivity-only.
- New: intrusion (§), complete-case triangulation, mixed-failure / rule-crosstab / coverage-hole figures.

## Known limitations (credits / holes)

See Limitations in `main.tex` and:

- `results/derived/O16_CLOSED_MODEL_CAVEAT.md`
- `results/derived/IDLE_CELLS.md`
- `results/derived/PRE_API_MASTER_AUDIT.md` (if present)
- `rebuild/REBUILD_REPORT.md` § NOT_COMPUTABLE
