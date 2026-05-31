# Step 14 — WIS / exposure framing **decision brief**

**Status:** `pending_user_decision` — you lock the choice in Phase 3 Step 14 at end of checklist.

---

## What claim is at stake (C14)

**Intended:** Coin change, shortest path, and WIS use the **same DP algorithm** but differ in **training-corpus exposure** (CC high → SP medium → WIS low). The exposure gradient should predict rename fragility.

**Also supported separately (C06):** WIS **collapses under W₃** for all models — that does not require the exposure causal story.

---

## Is the problem sample size?

**No.** Evaluated ALGO problems per subtype (from `vri_by_subtype.csv`):

| Subtype | n problems |
|---------|------------|
| Coin change | **25** |
| WIS | **30** |
| Shortest path | **55** |

WIS has **more** problems than CC in the eval slice. Step 14 is **not** blocked by “too few WIS questions.”

---

## The real confound: difficulty × exposure

| Subtype | ~Canonical accuracy (typical) | Corpus proximity | W₃ behaviour |
|---------|------------------------------|------------------|--------------|
| CC | ~40–70% | High | Partial survival |
| SP | ~40–65% | Medium | Mixed |
| WIS | ~**23–35%** | ~Zero | Often **0%** |

WIS is both **harder** and **lower exposure**. When WIS W₃ → 0%, two explanations fit:

1. **Exposure:** unseen in training → pattern-matching fails under rename  
2. **Difficulty floor:** models barely solve WIS anyway → rename finishes them off  

The paper draft already caveats this (§5.3 / limitations). Step 14 is choosing how hard to lean on the exposure gradient.

---

## Option A — Suggestive framing (0 API)

- Keep WIS collapse, proximity tables, and within-ALGO comparisons in main text
- **Do not** claim a clean causal CC > SP > WIS exposure gradient
- State in **Limitations:** WIS difficulty and exposure are confounded; proximity–fragility is mixed per model (C11 exploratory)
- **Claim tags:** C06 supported · C14 exploratory/suggestive

---

## Option B — Matched WIS bank (Step 17, ~1k+ API)

- Generate new WIS problems targeting **~60–70% canonical accuracy** (match CC/SP difficulty)
- Re-run P1 (canonical + variants) on all 5 models
- If WIS still collapses on W₃ at matched difficulty → much stronger exposure claim
- **Claim tags:** C14 can move to supported if results hold

See checklist **Step 17** for generation + sweep spec.

---

## What you decide (checklist Step 14)

- [ ] Option A / Option B / hybrid (e.g. A now, B for camera-ready)
- [ ] Exact limitations wording
- [ ] Update `claim_tagging_draft.md` (C14) and `main.tex` §5.3

---

## Supporting files

- `vri_by_subtype.csv`, `vri_proximity_correlation.csv`
- `vri_analysis_summary.md`
- `claim_tagging_draft.md` (C06, C11, C14)
