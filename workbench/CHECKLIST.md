# Submission checklist

Temporary working doc. **Experiments first, writing last.**

Track API spend in `workbench/API_USAGE.md` (e.g. `12/44`).

**Pipeline reference (verification + metrics + result files by W1–W6):** `workbench/PROBE_PIPELINE_REFERENCE.md`

---

## ⛔ API unavailable — blocked work (keep on checklist)

**Status (2026-05-30):** OpenRouter account wallet **−$0.66** → all API calls return **402**. No new model runs until wallet is topped up (~$1+).

| What we **can** do now | What is **blocked** (steps stay on checklist) |
|------------------------|------------------------------------------------|
| Analysis on existing CSVs (Phases 1–2, Steps 7–12) | Step 16 GSM P1 gap fill (~336 calls) |
| Design decisions Phase 3 (Steps 13–15) | Step 17 WIS matched bank + P1 sweep |
| Local mechanistic Step 21 (GPU) | Step 18 ALGO expansions (~588+ calls) |
| Paper / figure audit from current data | Step 19 reasoning models (DeepSeek-R1, Qwen3) |
| Regenerate derived metrics (no API) | Step 20 BW P2 protocol runs (~2,300 calls) |
| Row-by-row result file audit (`PROBE_PIPELINE_REFERENCE.md`) | Step 22 inter-rater / ablation API portions |

**When API returns:** resume Step 16 with `--resume` on existing raw files; log progress in `API_USAGE.md`.

---

## How to use this doc

Work **top to bottom** for analysis and API steps. **Do not lock design decisions during the run** — see Phase 3 at the end.

| Symbol | Meaning |
|--------|---------|
| `[x]` | Done (analysis / execution) |
| `[ ]` | To do next |
| `[~]` | In progress |
| `[-]` | Deferred |
| **`[D]`** | **Decision step — analysis brief ready; you choose at end of checklist** |
| **API** | Costs tokens / needs a run script |
| **analysis** | Existing CSVs only |
| **cleanup** | Fix data or derivation, no new model calls |

### Decision steps (Phase 3 — **last**)

Steps **13–15** are **not** auto-resolved by analysis runs. Each step has a **decision brief** (options + evidence). You pick after Phases 1–2, 4, and 21–22 are complete — then update paper + claim tags.

| Step | Brief | What you decide |
|------|-------|-----------------|
| 13 | `triangulation_framing_decision.md` | Thresholds, label scope, convergence narrative |
| 14 | `wis_exposure_framing_brief.md` | Suggestive vs matched WIS bank (Step 17) |
| 15 | Checklist + `claim_tagging_draft.md` | Submission gate sign-off |

**Regenerate derived artifacts** (after any raw CSV change):

```bash
python scripts/runs/step6_completeness.py   # Step 6: re-runs all packs + manifest
python scripts/runs/rederive_all_metrics.py
python scripts/runs/deep_metrics_analysis.py
python scripts/runs/triangulation_v2.py
python scripts/runs/scientific_filewise_audit.py
python scripts/runs/pre_api_master_audit.py   # refresh audit + PRE_API_RECOVERY_AUDIT.md before API
python scripts/runs/tep_dissociation_analysis.py
python scripts/runs/vri_analysis.py
python scripts/runs/triangulation_exploratory_analysis.py
python scripts/runs/cross_probe_correlation_analysis.py
```

**Quick references:** Step 6 → `STEP6_COMPLETENESS.md` · coverage → `COVERAGE_AUDIT_SUMMARY.md` · triangulation → `triangulation_v2_summary.md` · **pre-API audit** → `PRE_API_MASTER_AUDIT.md`

**Pre-API audit (run before any API spend):**

```bash
python scripts/runs/pre_api_master_audit.py
```

---

## You are here

**⛔ API blocked** — Phases 1–2 complete (analysis-only). **Phase 4 (Steps 16–20, 22 API portions) cannot run** until OpenRouter wallet is positive. **Proceed with:** Phase 3 decisions (13–15), result-file audit (`PROBE_PIPELINE_REFERENCE.md`), paper scoping from existing data. Step 16 attempted 2026-05-30 → 336× 402; see `STEP16_RUN_STATUS.md` + `API_USAGE.md`.

---

## Data problems registry (from pre-API audit)

Regenerate: `python scripts/runs/pre_api_master_audit.py` → `PRE_API_MASTER_AUDIT.md`, `PRE_API_RECOVERY_AUDIT.md`, `pre_api_unusable_flags.csv`, `pre_api_api_budget.csv`

**Before any API spend:** read `PRE_API_RECOVERY_AUDIT.md` — existing logs/raw may already cover gaps.

### Missing data — needs API (5 cells; GSM P2 o4-mini recovered)

| Cell | Have | Need | Fix | Est. API |
|------|------|------|-----|----------|
| GSM P1 GPT-4o | 20/44 | GSM_041–064 | **Exclude** duplicate GSM_021–040 (= GSM_001–020); run missing 24 IDs | **168** |
| GSM P1 Llama | 20/44 | GSM_041–064 | Same — **remap to 041–060 invalid** | **168** |
| ~~GSM P2 o4-mini~~ | **44/44** | — | **Recovered** in `GSM_P2_phase1_o1mini.csv` (see recovery audit) | **0** |
| ALGO P2A elicited Claude/Gemini/Llama | 61/110 each | 49 sessions | No revival in logs; partial P2A rerun | **196 × 3 = 588** |

**Core API backlog (Steps 17–18): ~924 calls** (336 GSM P1 + 588 ALGO P2A). ~~Step 16 GSM P2 o4-mini cancelled~~ — wire merged loader only.

### Uncanny / excluded from analysis (do not treat as valid zeros)

| Issue | Where | Fix | Blocks |
|-------|-------|-----|--------|
| GSM_021–040 duplicate IDs | GPT-4o/Llama P1 raw | **Exclude** from bank stats (same answers as GSM_001–020) | Misleading P1 if counted |
| BW P1 GSM contamination | `BW_P1_behavioral.csv` | Bank filter in derivations (**done**) | Raw row counts only |
| BW P2 pilot | 50/65 problems, 3/5 models | Scope claims; Step 20 protocol | Five-model BW P2; full bank |
| BW TEP 87% missing | 468/536 sessions | Protocol fix, not re-parse | TEP dissociation for BW |
| BW final_ok always 0% | P2 TEP slice | Scope BW process claims | Spearman TEP vs final |
| ALGO P2B n=61 | Injection CSVs | Label denominator (by design) | Cross-count with 110-problem P2A |
| P3 mechanistic | Qwen 0.5B only | Local compute (Step 21) | Cross-model mechanistic |
| ~~GSM P2 o4-mini missing~~ | `GSM_P2_phase1_o1mini.csv` | **Recovered** — merged in analysis loaders | — |
| ERROR: rows | BW/ALGO o4-mini P1 | `--resume` on existing files | Those variant rows only |

Full flags → `pre_api_unusable_flags.csv` · recovery → `pre_api_recovery_inventory.csv` · missing IDs → `pre_api_missing_ids.csv`

### Coverage snapshot

- **35/40** slices bank-complete in master coverage table (**+1** after o4-mini GSM P2 recovery)
- **0** all-blank rows in raw behavioral CSVs (validated)
- GSM/ALGO TEP analysis: **546/1017** TEP-valid sessions (GSM+ALGO usable; BW scoped separately)

---

## API budget by checklist step

| Step | Phase | API? | Est. calls | Notes |
|------|-------|------|------------|-------|
| 1–6 | Inventory | **No** | 0 | analysis / cleanup only |
| 6b | Pre-API audit | **No** | 0 | `pre_api_master_audit.py` |
| 7–12 | Phase 2 analysis | **No** | 0 | existing CSVs only |
| 13–15 | Design / gate | **No** | 0 | decisions only |
| **16** | GSM fix | **Yes** | **336** (P1 GPT-4o + Llama only) | ~~616 P2 o4-mini~~ recovered — merge loaders only |
| **17** | WIS matched bank | **Yes** | **TBD (~1k+)** | generation + P1 sweep all 5 models |
| **18** | ALGO expansions | **Yes** | **588** (P2A elicited) + TBD (CC n≥50, algo-invocation n≥80) | P1 priority = elicited gap |
| **19** | Reasoning models | **Yes** | **TBD** | DeepSeek-R1 + Qwen3 P1 slice |
| **20** | BW P2 protocol | **Yes** | **~2,300** | optional; defer until parser fixed |
| **21** | Mechanistic Llama | **No** | 0 | local GPU compute |
| **22** | Validation extras | **Mixed** | low | inter-rater human; some API |

**Totals:** core GSM+ALGO **~924** · with BW P2 extension **~3,224** · Step 17/18/19 expansions not fully costed

---

# Phase 1 — Know what results we have

Goal: every claim maps to a file; every gap is listed before spending more.

---

## Step 1 — Coverage & metric audit `[x]` · analysis

**Done.** Single derivation path; denominators flagged.

| Output | Path |
|--------|------|
| Master coverage | `results/derived/master_coverage_table.csv` |
| Gaps (long) | `results/derived/master_coverage_gaps.csv` |
| Cells needing runs | `results/derived/cells_needing_runs.csv` |
| Table flags | `results/derived/table_denominator_flags.csv` |
| Summary | `results/derived/COVERAGE_AUDIT_SUMMARY.md` |
| GSM P2 sensitivity | `results/paper/AUDIT/gsm_cci_wilcoxon_sensitivity.csv` |

**Command:** `python scripts/runs/rederive_all_metrics.py`

**Key facts to remember**
- **35/40** model×probe slices bank-complete; **5 incomplete**
- GSM P1 GPT-4o/Llama: **20/44 bank-valid** — GSM_021–040 are **duplicates** of 001–020; need API for 041–064
- GSM P2 o4-mini: **44/44** in `GSM_P2_phase1_o1mini.csv` (recovered; no Step 16 P2 API)
- BW P1 combined file has GSM IDs mixed in — filter before BW tables (**done** in derivations)
- BW P2: 50/65 problems, 3/5 models — not in master coverage table
- Five-model GSM P2 OK after merged loader (re-run Phase 2 scripts if stale)

---

## Step 2 — Headline metrics from raw `[x]` · analysis

**Done.** P1 accuracies, W3 retention, P2 CCI/TEP, coverage pivot.

**Command:** same as Step 1 (`rederive_all_metrics.py` steps 1–5)

| Output | Path |
|--------|------|
| Coverage pivot | `results/derived/coverage_pivot.csv` |
| P1 by variant | `results/derived/probe1_per_model_variant.csv` |
| P2 GSM / ALGO | `results/derived/probe2_gsm_metrics.csv`, `probe2_algo_metrics.csv` |

---

## Step 3 — Deep cross-probe metrics pack `[x]` · analysis

**Done.** Pairwise P1, transitions, P2A/B, mechanistic links, triangulation prototype.

**Command:** `python scripts/runs/deep_metrics_analysis.py`

| Output | Path |
|--------|------|
| P1 pairwise / transitions | `deep_p1_pairwise.csv`, `deep_p1_transitions.csv` |
| P2A phase link | `deep_p2a_phase_link.csv` |
| P2A schema audit | `deep_p2a_decision_schema_audit.csv` |
| P2B injection | `deep_p2b_response_profile.csv`, `deep_p2b_reactivity_delta.csv` |
| P3 mechanistic | `deep_probe3_mech_links.csv` |
| Triangulation+ (prototype) | `deep_triangulation_plus.csv` |
| Summary | `deep_metrics_summary.md` |

---

## Step 4 — Triangulation v2 + threshold sweep `[x]` · analysis

**Done.** Full signal k-of-n labels; 1944 threshold configs (no design lock-in yet).

**Command:** `python scripts/runs/triangulation_v2.py`

| Output | Path |
|--------|------|
| Per-instance labels | `results/derived/triangulation_v2_labels.csv` |
| Threshold sweep | `results/derived/triangulation_threshold_sweep.csv` |
| Summary | `results/derived/triangulation_v2_summary.md` |

**Numbers for later (don’t commit in paper yet)**
- Legacy ALGO strong labels: ~**3%**
- v2 default strong: ~**37%**
- v2 best sweep: ~**58%**

---

## Step 5 — File-wise scientific inventory `[x]` · analysis

**Done.** 42 raw + 52 derived files; 114 deductions.

**Command:** `python scripts/runs/scientific_filewise_audit.py`

| Output | Path |
|--------|------|
| Narrative audit | `results/derived/scientific_filewise_audit.md` |
| Deductions | `results/derived/scientific_file_deductions.csv` |

---

## Step 6 — Results completeness checklist `[x]` · cleanup + inventory

**Done.** See `results/derived/STEP6_COMPLETENESS.md`.

**Command:** `python scripts/runs/step6_completeness.py`

**Goal:** confirm we are not analyzing stale or contaminated slices before Phase 2.

- [x] Re-run all four regenerate commands above; confirm no errors
- [x] Read `cells_needing_runs.csv` — **5 incomplete cells** (GSM P2 o4-mini recovered; see recovery audit)
- [x] For BW P1 tables: derive from bank-filtered IDs only (exclude GSM contamination)
- [x] For ALGO P1 GPT-4o/Llama: note extra variant rows (`contaminated_extra_ids`) in flags
- [x] Build results manifest → `results_manifest.csv` (+ `scientific_file_profiles.csv`)
- [x] Tag each incomplete cell → `api_backlog_tagged.csv` (API vs cleanup vs remap)

**Outputs:** `STEP6_COMPLETENESS.md`, `results_manifest.csv`, `api_backlog_tagged.csv`

**Incomplete cells today** (from `cells_needing_runs.csv`)

| Priority | Cell | Issue | Fix type |
|----------|------|-------|----------|
| P1 | GSM P1 GPT-4o, Llama | 20/44 bank-valid | API for GSM_041–064 (exclude dup 021–040) |
| P1 | ALGO P2A elicited Claude/Gemini/Llama | 61/110 | API |
| P1 | BW P1 Claude/GPT-4o/Llama | extra GSM IDs in file | cleanup filter |
| P1 | ALGO P1 GPT-4o/Llama | extra variant rows | cleanup / audit |
| ~~P0~~ | ~~GSM P2 o4-mini~~ | ~~0/44~~ | **Recovered** — `GSM_P2_phase1_o1mini.csv` |

**Acceptance:** you can answer “what do we have?” and “what’s missing?” without opening raw folders.

---

## Step 6b — Pre-API master audit `[x]` · cleanup + inventory

**Done.** Consolidates missing data, unusable flags, and API budget.

**Command:** `python scripts/runs/pre_api_master_audit.py`

| Output | Path |
|--------|------|
| Summary | `results/derived/PRE_API_MASTER_AUDIT.md` |
| **Recovery audit** | `PRE_API_RECOVERY_AUDIT.md` |
| Slice inventory | `pre_api_slice_inventory.csv` |
| Missing IDs | `pre_api_missing_ids.csv` |
| Unusable / scoped flags | `pre_api_unusable_flags.csv` |
| API budget | `pre_api_api_budget.csv` |
| Recovery inventory | `pre_api_recovery_inventory.csv` |

**Re-run before Phase 4** or after any raw CSV change. **Check recovery audit before scheduling API.**

**Incomplete cells** (5 — see **Data problems registry** above)

| Priority | Cell | Issue | Fix | Est. API |
|----------|------|-------|-----|----------|
| P1 | GSM P1 GPT-4o, Llama | 20/44; dup IDs 021–040 | API for 041–064 | **336** |
| P1 | ALGO P2A elicited ×3 | 61/110 | Partial rerun | **588** |
| — | BW P2 pilot | 50/65, 3/5 models | Step 20 | **~2,300** (defer) |

---

# Phase 2 — Analyze everything we already have

Goal: extract paper-ready findings from existing CSVs. **No design choices required yet.**

---

## Step 7 — P2A decision link analysis `[x]` · analysis

**Done.** Raw match 0%; normalized ~28% (CC 45%, SP 23%, WIS 14%).

**Optional follow-up `[ ]`**
- [ ] Per-model/subtype breakdown figure from `deep_p2a_phase_link.csv`
- [ ] Sample mismatch cases from `deep_p2a_decision_schema_audit.csv` for appendix

---

## Step 8 — TEP dissociation analysis `[x]` · analysis

**Why:** TEP is high (~0.6–0.8) but final accuracy often still high — strongest unexplained P2 story.

**Script:** `python scripts/runs/tep_dissociation_analysis.py`

**Inputs (existing)**
- ALGO P2 phase2 normal / injected CSVs
- GSM: `results/raw/GSM_P2_cci.csv`
- BW: `results/raw/BW_P2_tep.csv` (TEP recomputed from cascade JSON when blank)

**Deliverables**
- [x] TEP vs final-correct scatter by family / model / subtype → `tep_dissociation_scatter.csv`
- [x] Injection cases: wrong intermediate, correct final — by step position & reasoning type → `tep_injection_recovery.csv`
- [x] Short write-up in `results/derived/tep_dissociation_summary.md`
- [x] Data quality audit → `tep_dissociation_quality_audit.md` (blank rows dropped; Spearman only when eligible)

**Key finding:** 116/1017 sessions show dissociation (high TEP + correct final); GSM high-TEP cases still end correct ~85–88% for Claude/Gemini/GPT-4o despite weak/positive TEP–final correlation.

**Blocks:** P2 narrative (not core triangulation thesis)

---

## Step 9 — VRI / rename-type analysis `[x]` · analysis

**Why:** VRI is defined but underused vs W3 retention alone. **Note:** GSM GPT-4o/Llama VRI uses partial bank (20/44) — flagged in summary.

**Script:** `python scripts/runs/vri_analysis.py`

**Inputs:** `probe1_per_model_variant.csv`, `probe1_w3_retention.csv`, `P1_metrics_by_model_subtype.csv`, `ALGO_P3_triangulation_v2.csv`

**Deliverables**
- [x] VRI vs W3 retention by subtype and model → `vri_by_model.csv`, `vri_by_subtype.csv`
- [x] Proximity–VRI correlation table (ALGO adversarial n=71/model) → `vri_proximity_correlation.csv`
- [x] Output: `results/derived/vri_analysis_summary.md`

**Key finding:** BW o4-mini highest VRI (+0.451); ALGO Claude +0.403; proximity–VRI_gap Spearman weak/ns pooled (ρ≈−0.04).

**Blocks:** optional polish (§2.5)

---

## Step 10 — Triangulation exploratory analysis `[x]` · analysis

**Why:** understand label behaviour before picking official thresholds (Step 14).

**Script:** `python scripts/runs/triangulation_exploratory_analysis.py`

**Inputs:** `triangulation_v2_labels.csv`, `triangulation_threshold_sweep.csv`, legacy `ALGO_P3_triangulation.csv`

**Deliverables**
- [x] Label distribution by family × model × subtype → `triangulation_label_distribution.csv`
- [x] Which vote signals fire most often → `triangulation_vote_fire_rates.csv`
- [x] Stability across reference / adjacent sweep configs → `triangulation_sweep_stability.csv`, `triangulation_sweep_param_sensitivity.csv`
- [x] Legacy vs v2 overlap table → `triangulation_legacy_v2_overlap.csv`, `triangulation_legacy_v2_buckets.csv`
- [x] Output: `results/derived/triangulation_exploratory_summary.md`

**Key findings:** ~38% insufficient under default; P3 `contam_high` + P2 `cci_retr` fire most; legacy strong ~3% vs v2 best-sweep ~58%; labels flip most when `min_votes` / `vote_margin` change.

**Do not:** pick paper thresholds here — Step 13.

---

## Step 11 — Cross-probe correlation pass `[x]` · analysis

**Why:** connect P1 fragility, P2 process, P3 contamination on same instances.

**Command:** `python scripts/runs/cross_probe_correlation_analysis.py`

**Inputs:** deep pack + v2 labels + `cross_probe_acc_vs_w3retention.csv` (refreshed by script)

**Deliverables**
- [x] Per-family: P1 W3 drop vs P2 CCI vs P3 contam (Spearman, per model) → `cross_probe_spearman_by_model.csv`
- [x] Instances where all three probes agree vs disagree → `cross_probe_agreement_instances.csv`, `cross_probe_triple_flagged_instances.csv`
- [x] Output: `results/derived/cross_probe_correlation_summary.md`

**Key findings:** 1249 instance rows; GSM model-level P1 drop vs P3 contam ρ=0.87 (p=0.06); triple retrieval-agree 0% at default flags; GPT-4o ALGO P1 retention vs CCI ρ=0.56 (p=0.003); BW P2 CCI sparse — merged from raw where available.

---

## Step 12 — Claim tagging draft `[x]` · analysis

**Goal:** map each intended paper claim to **supported / exploratory / blocked**.

- [x] Read `scientific_file_deductions.csv` + Phase 2 summaries
- [x] One table: claim → evidence file → status → blocker (if any)
- [x] Output: `results/derived/claim_tagging_draft.md`

**Summary:** 13 supported · 8 exploratory · 4 blocked — **C12/C14 finalize in Phase 3 `[D]` steps.**

**Acceptance for Phase 2:** all analysis scripts run; summaries exist; you know what the data says before choosing framing.

---

# Phase 4 — Fill missing results (API / local compute)

Goal: close gaps from Step 6. **Only after Phase 2 analysis tells you which gaps matter.**

> **⛔ Entire Phase 4 is on hold** while API wallet is negative. Steps below remain on the checklist — do not delete — but cannot execute until credits are restored.

---

## Step 16 — GSM P1 fix + P2 loader wiring `[~]` · API · **~336 calls** · **⛔ BLOCKED — no API**

| Task | Detail | Est. API | Status |
|------|--------|----------|--------|
| GSM P2 o4-mini | **Recovered** — `GSM_P2_phase1_o1mini.csv` 44/44 | **0** | `[x]` wired in loaders |
| GSM P1 GPT-4o/Llama | Exclude dup GSM_021–040; run GSM_041–064 | **168/model = 336** | `[ ]` **402 on 2026-05-30** — ERROR rows in raw; `--resume` after top-up |

**Do not** relabel GSM_021–040 → GSM_041–060 (verified duplicates of 001–020, not bank content).

Run log: `results/raw/new_model_sweep_logs/step16_gsm_p1_gpt4o_llama.log` · status: `results/derived/STEP16_RUN_STATUS.md`

**Then (after successful rerun):** `rederive_all_metrics.py` · `pre_api_master_audit.py` · re-run Phase 2 scripts if GSM P2 stale

---

## Step 17 — WIS matched-difficulty bank `[ ]` · API + generation · P0 · **TBD (~1k+)** · **⛔ BLOCKED — no API**

Generate WIS bank at ~60–70% canonical accuracy; P1 sweep all 5 models.  
**Blocks:** clean exposure-gradient claim (§0.3). See detailed spec below.

---

## Step 18 — ALGO expansions `[ ]` · API · P1 · **~588+ calls** · **⛔ BLOCKED — no API**

| Task | n target | Est. API | Why |
|------|----------|----------|-----|
| ALGO P2A elicited gap | 49 sessions × 3 models | **588** | 61/110 incomplete |
| Coin-change adversarial inversion | ≥50 | TBD | n=10 too small (§1.1) |
| Algorithm-invocation cases | ≥80 | TBD | 0/13 powered null (§1.2) |

---

## Step 19 — Reasoning models `[ ]` · API · P1 · **TBD** · **⛔ BLOCKED — no API**

DeepSeek-R1 + Qwen3 minimum P1 slice (ALGO canonical + W3 + contam).  
**Blocks:** “reasoning training breaks proximity” as class claim (§1.3).

---

## Step 20 — Blocksworld P2 protocol `[ ]` · API · P1 · **~2,300 calls** · **⛔ BLOCKED — no API**

Repair NL parser; ≥70% session completion on 50-problem pilot before model comparisons (§1.5).

| Sub-task | Est. API |
|----------|----------|
| Gemini + o4-mini full pilot (50 problems) | **~2,000** |
| Extend 50 → 65 problems (3 models) | **~300** |

**Defer** until Phase 2 shows BW claims matter — current data 87% TEP missing, 0% goal reach.

---

## Step 21 — Mechanistic on Llama `[ ]` · local compute

Replicate Qwen mechanistic pipeline on Llama-3.1-8B; join to behavioural W3/CCI (§1.4).

---

## Step 22 — Validation extras `[ ]` · P2 · **partially blocked**

Inter-rater on reasoning types (n=100 steps), tokenizer ablation, contamination proxy check for Llama (§2.1–2.4).

| Sub-task | API? | Status |
|----------|------|--------|
| Llama contamination proxy check (Infini-gram vs docs) | No | Can do now |
| Inter-rater second judge (LM) | Yes | **⛔ BLOCKED** |
| Tokenizer ablation runs | Yes | **⛔ BLOCKED** |
| Algorithm-elicitation RCT | Yes | **⛔ BLOCKED** |

---

# Phase 3 — Design decisions **`[D]` — you decide last**

Run **after** Phases 1–2, 4, and 21–22. Analysis briefs are prepared; **no option is locked until you check a box below.**

Goal: lock methods text, scope, and submission gate. **No new API in Steps 13–15** (Step 17 only if you choose Option B in Step 14).

---

## Step 13 — Triangulation framing `[D]` · **your decision**

**Brief:** `results/derived/triangulation_framing_decision.md` (options + counts; **not a locked choice**)  
**Draft methods text:** `triangulation_methods_paragraph.tex` · counts → `triangulation_paper_counts.csv`  
**Config draft:** `triangulation_official_config.json` (`status: pending_user_decision`)

**What the data supports (for your call):**
- Default thresholds: ~38% strong labels all families; **382/763 (50.1%)** strong on ALGO+GSM scope
- Sweep-tuned (`vote_margin=1`): ~58% strong but flips **24%** of instance labels vs default
- Legacy AND rule: ~3% strong (appendix comparison only)

**You decide:**
- [ ] Threshold set: default conservative / sweep-tuned / other
- [ ] Main-text scope: ALGO+GSM only / all families / other
- [ ] Narrative: existence proof / scalable diagnostic / other
- [ ] Apply choice to `main.tex` + update `claim_tagging_draft.md` (C12)

---

## Step 14 — Exposure / WIS framing `[D]` · **your decision**

**Brief:** `results/derived/wis_exposure_framing_brief.md`

**What the data supports (for your call):**
- WIS **W₃ collapse** is real (supported); CC>SP>WIS **causal exposure gradient** is confounded by WIS being harder (~35% vs ~65% canonical on CC)
- Evaluated n: CC **25**, WIS **30**, SP **55** — sample size is **not** the main blocker

**You decide:**
- [ ] **Option A:** Narrow to “suggestive”; document difficulty confound in limitations (0 API)
- [ ] **Option B:** Run Step 17 matched WIS bank (~1k+ API) for clean exposure claim
- [ ] Apply choice to `main.tex` + update `claim_tagging_draft.md` (C14)

---

## Step 15 — Submission gate review `[D]` · **your decision**

**Inputs:** `claim_tagging_draft.md` + your Step 13–14 choices + post–Phase 4 coverage

**You decide:**
- [ ] Every main-text claim tagged; blocked claims scoped or fixed
- [ ] No table with partial denominator without label
- [ ] Triangulation + WIS framing matches Step 13–14
- [ ] Small-n findings powered or downgraded (CC n=10, algo-invocation n=13)
- [ ] BW P2 scoped or Step 20 done
- [ ] `workbench/API_USAGE.md` updated
- [ ] Ready for Phase 5 writing

---

# Phase 5 — Writing `[-]`

**Start only when Phase 4 (and 21–22 if needed) are done and Phase 3 decisions (13–15) are locked by you.**

- [ ] Methods: derivation path (`results/paper/AUDIT/README.md`)
- [ ] Results: use claim tags from Step 12
- [ ] Limitations: P3 table + denominator caveats
- [ ] Figures regenerated from audited CSVs only

---

# Detailed reference (by topic)

Use when a step above points here. Not meant to be read linearly.

---

## P0 gaps (must resolve before strong claims)

### 0.1 Coverage audit — `[x]` see Step 1

### 0.2 Triangulation — Steps 4, 10, **13 `[D]`**

Analysis done; **threshold/narrative choice deferred to Step 13 (you decide at end).**

### 0.3 WIS confound — **Step 14 `[D]`**, 17

WIS harder AND lower exposure; matched bank (Step 17) only if you choose Option B in Step 14.

---

## P1 experiments (claim strength)

| ID | Topic | Step | Needs API |
|----|-------|------|-----------|
| 1.1 | Coin-change inversion n=10 | 18 | Yes |
| 1.2 | Algorithm-invocation n=13 | 18 | Yes |
| 1.3 | Reasoning models (o4-mini only) | 19 | Yes |
| 1.4 | Mechanistic Qwen-only | 21 | Local |
| 1.5 | BW P2 aborts | 20 | Yes |
| 1.6 | TEP dissociation | 8 | No |
| 1.7 | GSM gaps | 16 | Yes (P1 only; P2 recovered) |

---

## P2 validation (secondary)

| ID | Topic | Step |
|----|-------|------|
| 2.1 | Inter-rater reasoning types | 22 |
| 2.2 | Elicitation causality RCT | 22 |
| 2.3 | Tokenizer ablation | 22 |
| 2.4 | Contamination proxy validation | 22 |
| 2.5 | VRI underused | 9 |

---

## P3 limitations (document only)

Human baseline · few-shot · English-only · model version dates · o4-mini BW anomaly — see limitation paragraph in writing phase.

---

## Quick map — broken vs underpowered

| Gap | Priority | Blocks submission? | Step | Est. API |
|-----|----------|-------------------|------|----------|
| Triangulation legacy 3% labels | P0 | Yes (thesis) | 13 | 0 |
| Triangulation metric under-use | P0 | Yes | 10, 13 | 0 |
| WIS exposure confound | P0 | Yes | 17 | TBD |
| Metric derivation | P0 | Yes | 1 | 0 |
| TEP under-analysed | P1 | Yes | 8 done | 0 |
| GSM uneven n | P1 | Partial | 16 | **336** |
| o4-mini GSM P2 | — | **Recovered** | 16 | **0** |
| ALGO P2A elicited gap | P1 | Partial | 18 | 588 |
| CC inversion n=10 | P1 | No | 18 | TBD |
| BW P2 completion | P1 | Yes for BW process | 20 | ~2300 |
| Mechanistic one arch | P1 | Partial | 21 | 0 (local) |

---

**Done when:** experiments and derived artifacts support the claims — writing is calibration, not rescue.
