# C7 before/after

- Disabled sitewide .reveal fade/slide entrance (pipeline keeps its own motion)
- Kicker eyebrows: drop uppercase / wide tracking → sentence case
- Publications eyebrow: sentence case
- Who-nav eyebrow sentence case
- Who-chip eyebrow sentence case
- Reproduce label sentence case
- Removed '→' suffix from who-panel finding links
- Method 5a middle-dot → period
- Method 5b middle-dot → period
- Method 5c middle-dot → period
- Method 5d middle-dot → period
- Hero kicker: drop middle-dot meta
- Hero meta: middle dots → commas
- F1 kicker: middle-dot meta → period
- F7 kicker: middle-dot meta → period
- Who linkLabel F1 middle-dot → colon
- Who linkLabel F2 middle-dot → colon
- Who linkLabel F7 middle-dot → colon
- Who linkLabel F6 middle-dot → colon
- Who linkLabel programme middle-dot → colon
- Added visible :focus-visible rings using --accent
- Extended tabular-nums to table cells and metric readouts
- Removed transform from claim-card hover transition

## Accessibility fixes
- Tip: aria-label + hidden when empty
- Caption links always underlined (link-in-text-block)
- Prior-work heading h3→h2 (order after hero h1)
- Who title h4→h3
- Depth labels: no uppercase; prior-h style
- Finding-depth: h4→p.depth-label (heading order)
- Added data-URI favicon (stop console 404)
- Dark mode: primary filled buttons use ink text for AA contrast
- Applications row title h4→h3
- Applications heading style for h3
- pipeline.css dark primary button contrast

## Lighthouse (accessibility + best-practices)

| Theme | Before | After |
|-------|--------|-------|
| Light | a11y 90 · BP 96 | a11y **100** · BP **100** |
| Dark (`--force-dark-mode`) | (not measured before fixes) | a11y **100** · BP **100** |

Scores: `docs/site/c7_lighthouse/scores.json`.

### Audits fixed to reach ≥95
- `aria-tooltip-name` — named `#tip` tooltip; hide when idle
- `color-contrast` — dark-mode filled primary buttons use ink text
- `heading-order` — prior-work / who / apps / claim-map / finding-depth levels corrected
- `link-in-text-block` — caption links always underlined
- `errors-in-console` — data-URI favicon stops `/favicon.ico` 404
