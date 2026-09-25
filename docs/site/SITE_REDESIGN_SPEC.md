# Site redesign spec (2026-09-25)

target: `site/index.html` (single file, GitHub Pages via `.github/workflows/pages.yml`, deploys `site/`)
data: `site/data/pipeline_explorer.json`, built by `scripts/site/build_pipeline_explorer_data.py`, guarded by `tests/site/test_pipeline_explorer_data.py`
source notes: [[THE_PLAN_AMENDMENT_A1]], [[EF-07_W3_Construct_Audit]], [[EF-08_Position_On_Thinking]], [[BI-04_Engineering_Relevance]], `docs/audit/REPO_AUDIT_2026-09-25.md`

## 0. What must not change
- **Keep the visual identity.** Instrument Serif + Inter; tokens `--ink #0A0A0B`, `--paper #FAFAF8`, `--accent #1B4D8F`, `--teal #0F766E`, `--amber #B45309`, the dark-mode set, `--ease cubic-bezier(.16,1,.3,1)`, `--measure 68ch`, the left rail.
- **Keep the staged Probe 2 player** in `#method` (buttons `data-stage` 0 to 3: Declare, Execute, Inject, Score) and its motion. Keep `animateTrace` in `#f6`. These are the motion language for everything new: step-wise, user-driven, one thing changes at a time.
- **Keep** the reduced-motion rule (line ~52) and D3 v7.9.0 from jsDelivr.
- **Keep** every existing CSV link; numbers stay sourced.

## 1. Design direction for the new work
- **One signature moment.** The new Pipeline Explorer (section 3) is the only new orchestrated animation. Do not add fade-and-slide entrances to every section; do not add hover animation to every card.
- **Colour carries meaning, consistently everywhere:**
  - teal = preserved by the transform (numbers, structure, gold);
  - amber = changed by the transform (words, entities, direction);
  - accent blue = the programme's own action (verify, score, route);
  - muted = not run / gated.
- **Numbers.** `font-variant-numeric: tabular-nums` on every numeric cell and inline `.num`.
- **Cleanup of template tells** (only where they carry no information): meta strings joined by middle dots, "→" appended to link text, any all-caps eyebrow labels. Replace with plain sentence-case text.
- **Quality floor.** Keyboard operable (arrow keys step stages, Enter/Space activate), visible focus ring using `--accent`, `aria-live="polite"` region announcing each stage, contrast AA in both themes, responsive to 360 px (explorer stacks vertically), `prefers-reduced-motion` makes every transition instant.

## 2. Header and hero changes
- **Author line under the H1:** "Adya Srivastava, BITS Pilani. Supervised by Prof. Dhruv Kumar (BITS Pilani / IIIT Delhi)."
- **Publication line:** "Predecessor paper: Same Score, Different Strategy, CAISc 2026." Link to `paper/venue/caisc2026/`.
- **Status strip** (one line, directly under the hero text, muted):
  "Status, Sept 2026: behavioural layer shipped (Phase 0, gate G0 passed). Per-instance labels are not issued yet; they wait on calibration gate G1, which needs a GPU. Open-weight Track T is running on a T4."
- **Model count.** Standardise every mention on what `COVERAGE_MASTER.csv` supports (expected: five primary models, plus R1-distill-70B on planning only).

## 3. New section `#pipeline`: "From benchmark item to verdict" (the Pipeline Explorer)
Placement: directly after `#method` intro, before the three-probe block, so readers see one item travel the whole instrument before reading the details. Add to the rail as "Pipeline".

### 3.1 Data
- `fetch('data/pipeline_explorer.json')`. On failure, show a one-line error in the section that names the file and the build command; never render fake data.
- Payload shape: `items[]` with `problem_id, family, subtype, difficulty, source, verifier_kind, variants[] {variant, meaning, text, gold, gold_rule, verifier_function, mapping, probe1[] {model, correct, included, exclusion_reason}}, probe3_proximity {contamination_score, family_percentile, ...}, probe2_cci[]` and top-level `phase_routing[]`.
- Current exemplars (auto-picked by the script, rebuild anytime): GSM_047, BW_497, CC_035, WIS_001.

### 3.2 Layout (desktop)
```
+------------------------------------------------------------------+
| Item: [GSM_047] [BW_497] [CC_035] [WIS_001]      Play  Step  Reset|
+-------------------+----------------------------------------------+
| stage rail        |  stage canvas                                |
| 1 Source          |                                              |
| 2 Transform       |  (content for the active stage)              |
| 3 Verify          |                                              |
| 4 Probe 1         |                                              |
| 5 Probe 2         |                                              |
| 6 Probe 3         |                                              |
| 7 What we can say |                                              |
| 8 Where it goes   |                                              |
+-------------------+----------------------------------------------+
```
Mobile: item chips wrap; stage rail becomes a horizontal scroller above the canvas.

### 3.3 Stages (these are a real sequence, so numbered markers are justified)
1. **Source.** Canonical text in the serif at reading size; family, subtype, difficulty, source dataset. One sentence: "This is the item as it appears in the public benchmark."
2. **Transform.**
   - Variant chips canonical, W1 to W6 (only those present for this item). Selecting a chip morphs the text.
   - Word-level diff against canonical: removed words fade out in amber strikethrough, inserted words fade in amber, unchanged numbers pulse once in teal (motion 400 ms, `--ease`).
   - For W3, show the persisted mapping as a two-column table (canonical term, new term) when `mapping` exists; otherwise show the computed diff only.
   - Gold badge beside the text: "Gold fixed" (teal) for canonical and W1 to W4; "Gold re-derived" (amber) for W5 and W6, with the old value ticking to the new one.
   - W3 caption, verbatim: "As built, W3 moves the problem into a new real-word domain (a cover-story isomorph). A true nonce rename (W3a) is being added so the two can be compared."
3. **Verify.**
   - Show `verifier_kind` and `verifier_function`.
   - Animate "gold answer submitted to its own verifier", then a check mark (accent). If any row for this item has `included == false`, show its `exclusion_reason`: "Excluded, never scored".
   - Caption: "No model is queried until the gold answer passes its own verifier."
4. **Probe 1.**
   - Grid models × variants, cells filled by `correct` (teal filled = correct, outlined = incorrect, muted hatched = not run or excluded). Cells fill column by column on entry.
   - Below: retention for this item per model, as "canonical → W3" dots.
   - Caption states n = 1 item and links to `probe1_per_model_variant.csv` for the aggregate.
5. **Probe 2.**
   - If `probe2_cci` exists, show CCI per model and a "Replay how this is measured" button that scrolls to and starts the existing staged player.
   - If not, say "Probe 2 was not run on this item" plainly.
6. **Probe 3.**
   - A strip plot of the family's `contamination_score` distribution with this item highlighted and its percentile.
   - Caption: "Within-family exposure proxy from Infini-gram over public corpora. Not membership evidence for any closed model."
7. **What we can say.**
   - Honest verdict block, e.g.: "Probe-level evidence recorded. No per-instance retrieval or computation label is issued: the labelling rule is not calibrated until gate G1."
   - Show which of F1 to F7 this item contributes to (derive from family and variants present).
8. **Where it goes.**
   - Horizontal phase track from `phase_routing`: 0, 1, 1T, 2, 3, 4, with status styling: complete (filled accent), ready (outlined accent), blocked (amber outline), designed or planned (muted).
   - Clicking a phase reveals `uses_item` and linked claim/hypothesis IDs, which highlight in `#programme` using the existing cross-tab highlight mechanism.

### 3.4 Controls and motion
- Play auto-advances every 1.6 s and stops at stage 8. Step advances one stage. Reset returns to stage 1. Arrow keys step. Changing item resets to stage 1.
- Only one element animates at a time; the stage rail marker slides between stages (transform only, no layout thrash).
- Reduced motion: instant swaps, no auto-play.

## 4. Changes to existing sections
- **`#method` Probe 1 "Cannot discharge alone" copy:** replace the length-matched nonce claim with the EF-07 caption until W3a exists.
- **`#programme` phase timeline:** add Track T as a parallel lane beside Phase 1, labelled "1T open-weight track (T4)". State it discharges no gate. Add H8 to H13 and H7b to the hypothesis registry with links to D11 to D16 and EF-07.
- **`#f1`:** add one line: "W3 as built is a cover-story isomorph; see the construct note."
- **`#f3`:** add the prior-art line ("direction asymmetry in planning has been reported before; our open question is mechanism") and link HP-22.
- **Confidence intervals:** show the CI from the linked CSV next to each headline number in F1 to F3 wherever the CSV has one. If the CSV has none, do not invent one; add "CI pending" and file an issue.
- **`#next`:** replace with Track T cards T0 to T6, each showing hypothesis, primary contrast, kill criterion, cost, status. Keep one card "Phase 1 (G1) remains the priority spend" at the top.
- **`#who`:** extend the existing "who this helps" panel into section 5 below, or merge them.

## 5. New section `#applications`: "Where this applies, and where it does not"
- Filter chips for each engineering area in [[BI-04_Engineering_Relevance]].
- Each row shows what the instrument adds, the direction ID, and a status badge (shipped, untested, gated).
- A fixed line at the top: "This is a validity check, not an efficiency method. It does not make kernels faster or models smaller."

## 6. New section `#position`: "What this measures, and what it does not"
- The EF-08 one-paragraph position.
- Four toggles, each pairing a finding with its human analogue:
  - F1 with problem isomorphs;
  - F2 with content effects;
  - F3 with forward vs backward search;
  - CC-2 with human source monitoring.
- Label each human reference "to be verified in full text" until its P-note exists.

## 7. New box in `#motivation` or `#method`: "What is new relative to prior work"
Three rows:
- **GSM-Symbolic:** group-level, names and numbers. Ours: per-instance, cover-story isomorph vs numeric regeneration, convergent second probe.
- **PlanBench obfuscation:** replicated; ours adds the direction contrast and verification with a persisted mapping.
- **Counterfactual tasks:** same spirit; ours adds psychometric structure (G-theory, mixture IRT, intrusion errors).

## 8. Footer
- BibTeX block for the CAISc paper with a Copy button (confirmation text "Copied").
- "How to cite this site" line.
- A last-built date injected at build time, or a static date updated with each deploy.

## 9. Acceptance checks
- `python -m pytest tests/site` passes.
- No console errors; Lighthouse accessibility ≥ 95 in both themes.
- Every number on the page links to a CSV or is in the explorer JSON.
- Keyboard-only pass through the explorer works.
- Reduced motion verified in DevTools.
- Explorer renders at 360 px width.
