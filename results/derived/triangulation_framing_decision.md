# Step 13 — Triangulation framing **decision brief**

**Status:** `pending_user_decision` — options and evidence only; **you lock the choice in Phase 3 Step 13 at end of checklist.**  
**Inputs:** `triangulation_exploratory_summary.md`, `triangulation_v2_summary.md`, `triangulation_threshold_sweep.csv`

> This file was prepared during analysis to support your decision. Nothing here is binding until you check the boxes in `workbench/CHECKLIST.md` Step 13.

---

## 1. Threshold set — options (pick one)

### Option A — default conservative *(analysis default in `triangulation_v2.py`)*

| Parameter | Value | Notes |
|-----------|-------|-------|
| `min_votes` | **2** | vs 3 → +13 pp insufficient |
| `vote_margin` | **2** | vs 1 → see Option B |
| `w3_retrieval_max` | **0.2** | — |
| `w3_computation_min` | **0.5** | — |
| `contam_retrieval_min` | **0.6** | 0.5 (marginal +0.2 pp strong) |
| `contam_computation_max` | **0.4** | 0.5 |
| `cci_computation_min` | **0.5** | 0.6 (sweep-only; unstable) |
| `cci_retrieval_max` | **0.3** | 0.25 |

**If you pick Option A:** matches current `triangulation_v2_labels.csv`. Draft config → `triangulation_official_config.json`.

### Option B — sweep-tuned (`param_id=204`, `vote_margin=1`)

- ~**58%** strong labels (vs ~38% Option A)
- Flips **24%** of instance labels vs Option A; **301/1249** instances change label
- Legacy strong direction agreement still 100% where both strong, but Jaccard overlap with legacy remains ~3%

### Option C — stricter (`min_votes=3`)

- Raises **insufficient** rate ~+13 pp; fewer strong labels

### Stability note (for any option)

- **`vote_margin` 1→2** is the largest lever in the sweep (−19.6 pp strong)
- Default (Option A): **60.6%** of instances stable across six reference configs
- Legacy AND rule: ~**3%** strong (appendix comparison only)

---

## 2. Label definitions for paper

| Internal (`tri_v2_label`) | Paper name | Use in main text? |
|---------------------------|------------|-------------------|
| `retrieval` | **Retrieval-consistent** (strong) | Yes — existence examples |
| `computation` | **Computation-consistent** (strong) | Yes — existence examples |
| `weak_retrieval` | Weak retrieval signal | Appendix / sensitivity |
| `weak_computation` | Weak computation signal | Appendix / sensitivity |
| `mixed` | Conflicting probes | Count only; no directional claim |
| `insufficient` | Insufficient probe coverage | Exclude from strong-label denominator |

**Strong label** = `retrieval` ∪ `computation` only.  
**Confidence** = `|tri_score| / votes_total` (report in appendix, not main).

---

## 3. Narrative options (pick one)

| Option | Framing |
|--------|---------|
| **A — Existence proof** | Strong labels are illustrative instances where probes agree; not a scalable auto-labeler |
| **B — Diagnostic tool** | Present triangulation as a reusable per-instance diagnostic (stronger claim) |
| **C — Minimal** | Report probe agreement descriptively; de-emphasize formal labels in main text |

**Scope options (independent):**
- Main text **ALGO + GSM only** (BW appendix — P2 signals sparse): strong **382/763 (50.1%)** under Option A thresholds
- All families: strong **502/1332 (37.7%)** under Option A thresholds

Legacy main text (8+4 strong, 61.6% ambiguous) should be replaced or appendix-only regardless of choice.

---

## 4. Reference counts (Option A thresholds, 2026-05-30 refresh)

### All families (n=1,332 instance rows)

| Label | n | % |
|-------|---|---|
| insufficient | 505 | 37.9% |
| computation | 342 | 25.7% |
| weak_retrieval | 199 | 14.9% |
| retrieval | 160 | 12.0% |
| mixed | 67 | 5.0% |
| weak_computation | 59 | 4.4% |
| **Strong total** | **502** | **37.7%** |

### Main-text scope: **ALGO + GSM only** (n=763)

| Label | n | % |
|-------|---|---|
| insufficient | 166 | 21.8% |
| computation | 264 | 34.6% |
| retrieval | 118 | 15.5% |
| weak_retrieval | 89 | 11.7% |
| mixed | 67 | 8.8% |
| weak_computation | 59 | 7.7% |
| **Strong total** | **382** | **50.1%** |

By family (strong / insufficient):

| Family | n | Strong | Insufficient |
|--------|---|--------|--------------|
| ALGO | 551 | 39.0% | 28.7% |
| GSM | 212 | 78.8% | 3.8% |
| BW (appendix) | 569 | 21.1% | 59.6% |

---

## 5. Draft methods paragraph (edit after you decide)

LaTeX starting point → `triangulation_methods_paragraph.tex` *(written for Option A — revise if you pick B/C)*

---

## 6. Paper edit checklist *(after you lock Step 13)*

- [ ] Replace §5.5 legacy counts with your chosen v2 table
- [ ] Limitations: insufficient rate + BW scope per your choice
- [ ] Appendix: sweep sensitivity + legacy crosstab
- [ ] Update `claim_tagging_draft.md` (C12)

---

## 7. Claim C12 *(pending your Step 13 choice)*

Currently tagged **exploratory** in `claim_tagging_draft.md` until you lock thresholds + scope.

---

## Regenerate

```bash
python scripts/runs/triangulation_v2.py
python scripts/runs/triangulation_exploratory_analysis.py
```

Official labels are already produced by default `TriThresholds`; no code change required unless thresholds change.
