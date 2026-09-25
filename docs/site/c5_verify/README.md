# C5 verify notes (Pipeline Explorer)

## Checks
- `PYTHONPATH=. python scripts/site/build_pipeline_explorer_data.py` → GSM_047, BW_497, CC_035, WIS_001
- `PYTHONPATH=. python -m pytest tests/site -q` → 4 passed
- Keyboard: ArrowRight/Left on `#pipe-canvas` steps stages; `aria-live` announces stage
- `prefers-reduced-motion: reduce`: Play single-steps (no auto-advance / Pause)
- 360 px: `#pipeline` width 360; stage rail horizontal overflow scroll
- Themes: light + dark via `prefers-color-scheme` (screenshots below)
- Exposed: `window.startProbe2Player`, `window.setProgrammeFocus`
- Pages workflow: builds JSON + pytest before upload; paths include `site/data/**`, `scripts/site/**`, `tests/site/**`

## Screenshots
| Stage | Light | Dark |
|-------|-------|------|
| 2 Transform (W3 diff) | `stage2_light.png` | `stage2_dark.png` |
| 4 Probe 1 grid | `stage4_light.png` | `stage4_dark.png` |
| 8 Where it goes | `stage8_light.png` | `stage8_dark.png` |
