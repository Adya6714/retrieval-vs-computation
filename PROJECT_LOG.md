# PROJECT_LOG.md

Single source of truth for codebase state, bugs, deferred decisions, and what to do when new things arrive. Read Section 6 before touching anything after a gap.

---

## Section 0 — Current Evaluation Snapshot (BW + GSM + ALGO)

This section is the fastest way to understand what was actually evaluated and where outputs live.

### 0.1 Theory reason (why the pipeline is structured this way)

All families use the same three-probe logic:
- **Probe 1:** behavior under controlled variants (`W1..W6`) to separate surface vs structure sensitivity.
- **Probe 2:** process consistency (declared strategy vs stepwise execution; injected-state sensitivity).
- **Probe 3:** contamination indexing (Infini-gram signal) to test retrieval dependence.

The core scientific claim should only be made where these three probes converge.

### 0.2 Family-by-family pipeline ownership

- **BW (Blocksworld / planning)**
  - Probe 1 runner: `scripts/BW_P1_SCR_run_behavioral_sweep.py`
  - Probe 2 runners: `scripts/BW_P2_SCR_run_cci.py`, `scripts/BW_P2_SCR_run_tep.py`
  - Probe 3 runners: `scripts/BW_P3_SCR_run_contamination_triage.py`, `scripts/BW_P3_SCR_run_triangulation.py`
  - Key outputs: `results/BW_P1_RES_*`, `results/BW_P2_RES_*`, `results/BW_P3_RES_*`

- **GSM (arithmetic_reasoning)**
  - Bank prep: `scripts/GSM_PX_SCR_fix_question_bank.py`, `scripts/GSM_PX_SCR_generate_w6.py`
  - Probe 1 metrics: `scripts/GSM_P1_SCR_compute_metrics.py`
  - Probe 2 run/metrics: `scripts/GSM_P2_SCR_run_probe2.py`, `scripts/GSM_P2_SCR_compute_metrics.py`
  - Key outputs: `results/GSM_P1_RES_*`, `results/GSM_P2_RES_*`, `results/GSM_P3_RES_*`

- **ALGO (coin_change / shortest_path / wis)**
  - Bank prep/audit: `scripts/ALGO_PX_SCR_fix_question_bank.py`, `scripts/ALGO_PX_SCR_audit_bank.py`
  - Probe 1: `scripts/ALGO_P1_SCR_run_behavioral_sweep.py`, `scripts/ALGO_P1_SCR_compute_metrics.py`
  - Probe 2: `scripts/ALGO_P2_SCR_run_phase1.py`, `scripts/ALGO_P2_SCR_run_phase2.py`, `scripts/ALGO_P2_SCR_compute_metrics.py`
  - Probe 3 + triangulation: `scripts/BW_P3_SCR_run_contamination_triage.py --family algorithmic ...`, `scripts/ALGO_P3_SCR_triangulation.py`
  - Key outputs: `results/ALGO_P1_RES_*`, `results/ALGO_P2_RES_*`, `results/ALGO_P3_RES_*`

### 0.3 Analysis-first file index

For quick analysis, open in this order:
1. `README.md` (family map + canonical files)
2. `data/problems/question_bank_algo.csv` (or family bank)
3. `results/<FAMILY>_P1_RES_*`
4. `results/<FAMILY>_P2_RES_*`
5. `results/<FAMILY>_P3_RES_*`
6. Figure generators in `scripts/*_FIG_generate.py`

### 0.4 Latest ALGO-specific note

- ALGO triangulation and figure scripts are now present:
  - `scripts/ALGO_P3_SCR_triangulation.py`
  - `scripts/ALGO_P1_FIG_generate.py`
  - `scripts/ALGO_P2_FIG_generate.py`
  - `scripts/ALGO_P3_FIG_generate.py`
- Current ALGO triangulation output:
  - `results/ALGO_P3_RES_triangulation.csv`
  - `results/ALGO_P3_RES_regression.txt`

### 0.5 Exact result files to inspect per family and probe

Use this checklist when doing analysis sessions.

#### BW

- Probe 1:
  - `results/BW_P1_RES_behavioral_sweep.csv`
  - compatibility snapshot: `results/BW_RES_P1_behavioral_sweep.csv`
- Probe 2:
  - `results/BW_P2_RES_phase1_plans.csv`
  - `results/BW_P2_RES_cci.csv`
  - `results/BW_P2_RES_tep.csv`
  - `results/BW_P2_RES_validity_comparison.csv`
  - `results/BW_P2_RES_validity_summary.txt`
  - `results/BW_P2_LOG_injection_trace.txt`
- Probe 3:
  - `results/BW_P3_RES_contamination_triage.csv`
  - `results/BW_P3_RES_triangulation_per_instance_claude37.csv`
  - `results/BW_P3_RES_triangulation_per_instance_gpt4o.csv`
  - `results/BW_P3_RES_triangulation_per_instance_llama8b.csv`
  - `results/BW_P3_RES_contamination_regression_claude37.txt`
  - `results/BW_P3_RES_contamination_regression_gpt4o.txt`
  - `results/BW_P3_RES_contamination_regression_llama8b.txt`

#### GSM

- Probe 1:
  - `results/GSM_P1_RES_behavioral_sweep_claude.csv`
  - `results/GSM_P1_RES_behavioral_sweep_gpt4o.csv`
  - `results/GSM_P1_RES_behavioral_sweep_llama.csv`
  - `results/GSM_P1_RES_css.csv`
  - `results/GSM_P1_RES_var.csv`
  - `results/GSM_P1_RES_vri.csv`
  - `results/GSM_P1_RES_rcs.csv`
  - `results/GSM_P1_RES_rcs_by_difficulty.csv`
  - `results/GSM_P1_RES_step_count_sensitivity.csv`
  - `results/GSM_P1_RES_w4_gap.csv`
- Probe 2:
  - `results/GSM_P2_RES_cci.csv`
  - `results/GSM_P2_RES_metrics_summary.csv`
  - `results/GSM_P2_LOG_human_approval_queue.csv`
- Probe 3:
  - `results/GSM_P3_RES_contamination_triage.csv`
  - `results/GSM_P3_RES_triangulation_per_instance_claude.csv`
  - `results/GSM_P3_RES_triangulation_per_instance_gpt4o.csv`
  - `results/GSM_P3_RES_contamination_regression_claude.txt`
  - `results/GSM_P3_RES_contamination_regression_gpt4o.txt`

#### ALGO

- Probe 1:
  - `results/ALGO_P1_RES_behavioral_sweep_claude.csv`
  - `results/ALGO_P1_RES_behavioral_sweep_gpt4o.csv`
  - `results/ALGO_P1_RES_behavioral_sweep_llama.csv`
  - `results/ALGO_P1_RES_behavioral_sweep_mock.csv`
  - `results/ALGO_P1_RES_human_review_queue.csv`
  - `results/ALGO_P1_RES_metrics.csv`
- Probe 2:
  - `results/ALGO_P2_RES_phase1_claude.csv`
  - `results/ALGO_P2_RES_phase1_gpt4o.csv`
  - `results/ALGO_P2_RES_phase1_llama.csv`
  - `results/ALGO_P2_RES_phase2_normal.csv`
  - `results/ALGO_P2_RES_phase2_injected.csv`
  - `results/ALGO_P2_RES_metrics.csv`
- Probe 3:
  - `results/ALGO_P3_RES_contamination.csv`
  - `results/ALGO_P3_RES_triangulation.csv`
  - `results/ALGO_P3_RES_regression.txt`
  - `results/ALGO_PX_RES_bank_audit.csv` (required quality gate before interpretation)

### 0.6 Script ownership by family (quick debugging map)

#### ALGO scripts

- `scripts/ALGO_PX_SCR_fix_question_bank.py`: targeted + propagated bank fixes.
- `scripts/ALGO_PX_SCR_backfill_greedy_metadata.py`: fills `greedy_succeeds`, `instance_type`.
- `scripts/ALGO_PX_SCR_add_critical_step.py`: fills `critical_step_index`.
- `scripts/ALGO_PX_SCR_audit_bank.py`: strict schema/content/cross-variant audit.
- `scripts/ALGO_PX_SCR_generate_w6.py`: W6 procedural generation.
- `scripts/ALGO_P1_SCR_run_behavioral_sweep.py`: Probe 1 ALGO sweeps.
- `scripts/ALGO_P1_SCR_compute_metrics.py`: Probe 1 ALGO metrics.
- `scripts/ALGO_P2_SCR_run_phase1.py`: Probe 2 phase 1 strategy extraction.
- `scripts/ALGO_P2_SCR_run_phase2.py`: Probe 2 phase 2 stepwise + injection.
- `scripts/ALGO_P2_SCR_compute_metrics.py`: Probe 2 metrics.
- `scripts/ALGO_P3_SCR_triangulation.py`: per-instance diagnosis + Table 1 + OLS.
- `scripts/ALGO_P1_FIG_generate.py`, `scripts/ALGO_P2_FIG_generate.py`, `scripts/ALGO_P3_FIG_generate.py`: ALGO figure bundles.

#### GSM scripts

- `scripts/GSM_PX_SCR_fix_question_bank.py`, `scripts/GSM_PX_SCR_generate_w6.py`: bank prep.
- `scripts/GSM_P1_SCR_compute_metrics.py`: Probe 1 metrics.
- `scripts/GSM_P2_SCR_run_probe2.py`, `scripts/GSM_P2_SCR_compute_metrics.py`: Probe 2 run/metrics.
- `scripts/GSM_P1_FIG_generate.py`, `scripts/GSM_P2_FIG_generate.py`, `scripts/GSM_P3_FIG_generate.py`: figure bundles.

#### BW scripts

- `scripts/BW_P1_SCR_run_behavioral_sweep.py`: Probe 1 sweeps.
- `scripts/BW_P2_SCR_extract_phase1_plans.py`: Phase 1 extraction for Probe 2.
- `scripts/BW_P2_SCR_run_cci.py`, `scripts/BW_P2_SCR_run_tep.py`: Probe 2.
- `scripts/BW_P2_SCR_generate_figures.py`: BW Probe 2 figures.
- `scripts/BW_P3_SCR_run_contamination_triage.py`: contamination triage.
- `scripts/BW_P3_SCR_run_triangulation.py`: triangulation + regression.

---

## Section 1 — Phase Status

| Phase | Status | Owner | Blocked On |
|-------|--------|-------|------------|
| Phase 0 — Infrastructure | Complete | Adya | — |
| Phase 1 — Contamination Triage | Code complete, not executed | Adya | Adya's CSV PR |
| Phase 2 — Mechanistic Tooling | Code complete, not executed | Adya | GPU access |
| Phase 3 — Probe 1 | Not started | — | Phase 1 gate + variant bank |
| Phase 4 — Probes 2 and 3 | Not started | — | Phase 3 gate |
| Phase 5 — Writing | Not started | All | Phase 4 gate |
| Phase 6 — Submission | Not started | — | Phase 5 gate |

---

## Section 2 — Execution Checklist

### Phase 1
1. Merge Adya's PR
2. Run Section 6 checklist before anything else
3. `pip install -r requirements.txt`
4. `python scripts/test_api_keys.py`
5. `python scripts/run_contamination_triage.py --limit 5` — smoke test
6. `python scripts/run_contamination_triage.py` — full run (resumable)
7. `python notebooks/probe1_triage_plot.py` — Gate 1 plot
8. **Gate 1:** visible correlation between contamination score and accuracy? If yes → Phase 2. If flat → revise `score.py` fingerprinting once, re-run, reconsider Probe 3.

### Phase 2
1. `python scripts/check_transformerlens_support.py` — run on laptop now, no GPU needed
2. Confirm GPU access
3. Uncomment `transformer_lens` in `requirements.txt`, reinstall
4. `python probes/mechanistic/load_model.py`
5. `python probes/mechanistic/sanity_check.py` — **Gate 2: must pass before any real mechanistic runs**

### Phase 3
1. Variant bank complete in `data/problems/probe1_variants.csv`
2. Add variant support to `run_behavioral_sweep.py` (see Section 5, Bug 1)
3. `python scripts/run_behavioral_sweep.py --dry-run --limit 5`
4. `python scripts/run_behavioral_sweep.py --model claude-sonnet-3-7`
5. `python scripts/run_mechanistic_sweep.py --family blocksworld`

### Phase 4
1. `python scripts/run_behavioral_sweep.py --probe2`
2. CCI + TEP from `probes/behavioral/cci.py`, `tep.py`
3. Full contamination sweep
4. `python scripts/run_triangulation.py`

---

## Section 3 — Open Decisions

| Decision | Options | Owner | Blocked On |
|----------|---------|-------|------------|
| Closed model pair | GPT-4o/o3 vs Claude Sonnet 3.7 | Adya | Phase 3 start |
| GPU compute source | RunPod vs BITS cluster vs Colab | Adya | Confirm now |
| TransformerLens Qwen2.5 support | Confirmed vs nnsight fallback | Adya | Run check script |
| Tuned lens for Qwen | Pretrained vs logit lens fallback | Adya | Phase 2 |
| Final family composition | Current proposal vs Shaswat revision | Shaswat → Adya | Shaswat Task 1 |
| Mystery Blocksworld inclusion | Include vs cut | Shaswat | Shaswat Task 1 |
| Contamination fingerprinting method | n-gram length, sliding window, formula | Adya | Phase 1 execution |

---

## Section 4 — Scripts Written, Not Yet Executed

| Script | Can Run Locally | Needs GPU | Needs CSV |
|--------|----------------|-----------|-----------|
| `scripts/test_api_keys.py` | Yes | No | No |
| `scripts/check_transformerlens_support.py` | Yes | No | No |
| `scripts/run_contamination_triage.py` | Yes | No | Yes |
| `scripts/run_behavioral_sweep.py` | Yes (mock/dry-run) | No | Yes |
| `scripts/run_mechanistic_sweep.py` | Yes (--dry-run, GPT-2) | Yes (real) | Yes |
| `scripts/run_triangulation.py` | Yes | No | Yes |
| `probes/mechanistic/load_model.py` | No | Yes | No |
| `probes/mechanistic/activations.py` | No | Yes | No |
| `probes/mechanistic/similarity.py` | Yes (numpy only) | No | No |
| `probes/mechanistic/logit_lens.py` | No | Yes | No |
| `probes/mechanistic/sanity_check.py` | No | Yes | No |
| `probes/mechanistic/patching.py` | No | Yes | No |
| `notebooks/probe1_triage_plot.py` | Yes | No | Yes |

---

## Section 5 — Known Bugs and Fixes Needed

### Bug 1 (CRITICAL): `run_behavioral_sweep.py` — no variant support
The behavioral sweep currently only runs canonical problems. CSS requires variant responses (W1-W4, W6 per problem). This is the most important missing piece before Phase 3.

**What needs to happen when variant bank arrives:**
- `run_behavioral_sweep.py` must add a second loop over `probe1_variants.csv`
- Output CSV must include a `variant_type` column (empty string or `"canonical"` for base problems)
- `run_triangulation.py` already has a guard for missing `variant_type` column — it warns and skips CSS

**Cursor prompt to run at Phase 3 setup:**
```
In scripts/run_behavioral_sweep.py, after the main canonical sweep loop is complete,
add a second loop over data/problems/probe1_variants.csv.
For each variant row, run the same model query, write a result row with:
  problem_id, problem_family, variant_type, model_answer, correct_answer, behavioral_correct, model.
For canonical rows, set variant_type = "canonical".
Apply the same --resume, --limit, --family flags.
Do not change the canonical loop logic.
```

### Bug 2: `model_sweep.py` / `run_behavioral_sweep.py` — verify_answer signature
Calls `verify_answer(raw_response, correct_answer, problem_family)` but actual signature is `verify_answer(problem_id, model_answer, ground_truth, family)`.

**Cursor fix:**
```
In scripts/run_behavioral_sweep.py, fix the verify_answer call to:
verify_answer(problem_id, raw_response, correct_answer, problem_family)
```

### Bug 3: `run_mechanistic_sweep.py` — crystallization layer approximation
Uses first token of correct answer as proxy for crystallization detection. Works for single-token answers. Approximate for multi-token answers (Blocksworld plans).

**Status:** Acceptable approximation for the paper as a relative measure. Document in methods section. Not a blocking bug.

### Bug 4: `run_mechanistic_sweep.py` — load_model parameter
`load_model.load_model()` must accept `model_name` as a parameter. Verify `probes/mechanistic/load_model.py` signature before Phase 2. If it doesn't accept `model_name`, add it.

### Bug 5 (FIXED): `sanity_check.py` structural error
`extract_activations` calls were outside function body. Fixed in Cursor. Confirmed fixed.

### Bug 6 (FIXED): `triage.py` behavioral_correct column
Was incorrectly reading `behavioral_correct` from input CSV. Fixed — column removed from triage output.

---

## Section 6 — What To Do When the Question Bank Arrives

Run these checks **before executing any script** after Adya's PR merges.

**Check 1: family name strings**
Open `data/problems/probe1_instances.csv` and `probes/contamination/verify.py` side by side. The `problem_family` values in the CSV must exactly match the strings the verifier expects:
```
numeric families: gsm, coin_change, knapsack, weighted_interval_scheduling, shortest_path
plan families:    blocksworld, logistics, mystery_blocksworld
```
If they don't match, decide which is canonical (suggest fixing the CSV to match the verifier since the verifier is already tested) and update consistently everywhere.

**Check 2: correct_answer format**
Spot-check 3 rows per family:
- GSM / numeric: plain number only — `42`, not `$42`, not `42 dollars`
- Shortest path (path): comma-separated uppercase nodes, no spaces — `A,C,D,F`
- Shortest path (distance): plain number — `7`
- Blocksworld: one move per line, exact format `move X from Y to Z`
- WIS / Knapsack / Coin Change: plain integer

Wrong format = verifier returns False on correct answers = broken results. Fix the CSV before running.

**Check 3: smoke test**
```bash
python scripts/run_contamination_triage.py --limit 3
```
Confirm `results/contamination_triage.csv` created, 3 rows, no errors. Do not run full sweep until this passes.

**Check 4: probe2 columns**
Expected: `problem_id, problem_text, initial_state, goal_state, correct_plan, num_blocks, difficulty, source_dataset, source_instance_id, contamination_pole`. If columns differ, update `probes/common/io.py` load validation to match.

**Check 5: manually validate one Blocksworld plan**
Pick one row from `probe2_instances.csv`. Draw the initial state. Apply each move in `correct_plan` one by one. Confirm you reach `goal_state`. If not, the plan is wrong — flag it before running anything.

---

## Section 7 — TransformerLens + Qwen2.5 Compatibility

**Status: UNVERIFIED. Run check script before Phase 2.**

```bash
python scripts/check_transformerlens_support.py
```

**If Qwen2.5-7B IS in the TransformerLens registry:** proceed normally.

**If Qwen2.5-7B is NOT supported**, two fallback options:

Option A — nnsight (preferred fallback):
- Cleaner API for HuggingFace models, good logit lens support
- Files to rewrite: `activations.py`, `logit_lens.py`, `patching.py`, `load_model.py`
- Replace `HookedTransformer.from_pretrained` with `AutoModelForCausalLM.from_pretrained`
- Replace `run_with_cache` and `run_with_hooks` with nnsight's `.trace()` context manager
- Estimated rewrite: 1-2 days

Option B — raw HuggingFace hooks (most portable):
- No extra dependencies beyond transformers
- More verbose than nnsight
- Files to rewrite: same four
- Use `model.register_forward_hook()` for activation extraction
- Use manual unembed projection for logit lens
- Estimated rewrite: 2-3 days

**What is NOT affected by switching away from TransformerLens:**
- `similarity.py` — operates on numpy arrays
- All of `probes/behavioral/`
- All of `probes/common/`
- All of `probes/contamination/`
- All of `probes/triangulation/`

---

## Section 8 — Logit Lens Compatibility Note

`logit_lens.py` projects the residual stream at each layer through the unembedding matrix to get per-layer token predictions. For Qwen2.5, TransformerLens must correctly handle the unembedding and layer norm weights.

**`sanity_check.py` is the verification gate.** It runs an IOI-style test — if the logit lens recovers the expected crystallization pattern on this known finding, the pipeline is trustworthy. If it fails, debug `logit_lens.py` before any real runs.

**Known approximation in crystallization layer detection:** `run_mechanistic_sweep.py` uses the first token of the correct answer as a proxy for the answer token. This works well for single-token answers (numbers). For multi-token answers (Blocksworld plans), it is approximate. This is acceptable as a relative measure across instances — document in methods. Not worth fixing unless the paper makes absolute claims about crystallization depth.

---

## Section 9 — Collaborator Status

| Person | Task | Status | Deliverable |
|--------|------|--------|-------------|
| Shaswat | Finalize family composition | Not started | Decision doc to Adya |
| Shaswat | Write INSTANCE_SELECTION_CRITERIA.md | Not started | `team/INSTANCE_SELECTION_CRITERIA.md` |
| Shaswat | Build probe1 + probe2 CSVs | Not started | PR against main |
| Nandini | Work plan after family lock | Waiting on Shaswat | Message to team |
| Nandini | Cross-review process | Waiting on first PR | `team/review_log.csv` |
| Nandini | W1 + W2 variants for Algorithmic Suite | Not started | Rows in probe1_variants.csv |

---

## Section 10 — Work Log

### Session 1
**Done:** Full repo structure, `.cursorrules`, `.gitignore`, `requirements.txt`, `CHARTER.md`, result CSV schemas. Phase 1: `infinigram_client.py`, `score.py`, `triage.py`. Phase 2: `load_model.py`, `activations.py`, `similarity.py`, `logit_lens.py`, `sanity_check.py`. TransformerLens library decision.

**Blockers at end:** Adya's PR, GPU, `sanity_check.py` structural bug, model sweep not written.

### Session 2
**Done:** `configs/`, `.env.example`, `Makefile`, `verify.py` stub, `probes/common/io.py`, `parsers.py`, `stats.py`, `mock_client.py`, `probe1_triage_plot.py`, `run_contamination_triage.py`. Variant numbering locked W1-W6 (W6=procedural, W5=reversal). All team docs: `shasshy.md`, `nandini.md`, `CONTRIBUTING_VARIANTS.md`, `REVIEW_PROTOCOL.md`.

**Bugs identified:** `triage.py` behavioral_correct (fixed), `model_sweep.py` path + signature (queued).

### Session 3
**Done:** All five metric modules (css, rcs, cas, cci, tep). `patching.py`. `per_instance.py`. `run_mechanistic_sweep.py`. `run_triangulation.py`. `anthropic_client.py`. `openai_client.py`. Full test suite (test_verifiers, test_parsers, test_stats, test_mock_sweep). `sanity_check.py` structural bug fixed.

**Bugs fixed this session:** patching.py tensor serialization, load_model call, crystallization layer detection, run_triangulation variant_type guard, cas.py family duplication, tep.py None label.

**Pipeline status:** structurally complete. All Layer 1 code exists and is tested. Ready to execute the moment CSV + API keys arrive.

### Session 4
**Done:** Introduced a single strict question bank schema at `data/problems/question_bank.csv` with canonical columns:
`problem_id, variant_type, problem_text, correct_answer, problem_family, problem_subtype, difficulty, contamination_pole, source, verifier_function, difficulty_params, notes`.
Set this file as the default source for Probe 1 in:
`run_behavioral_sweep.py`, `run_contamination_triage.py`, `fill_correct_answers.py`, and `run_mechanistic_sweep.py`.
Added strict schema validation via `probes/common/io.py` (`QUESTION_BANK_COLUMNS`, `load_question_bank`), and canonical-row filtering (`variant_type == canonical`) in sweep/triage/mechanistic defaults.

**Migration note:** Question bank initialized/cleared with header-only CSV as requested. Legacy `probe1_instances.csv` remains as backup/source during migration and is no longer the default for updated Probe 1 scripts.

### Session 5
**Done:** Migrated `Dataset Week 1.csv` (wide format) into unified `data/problems/question_bank.csv` (long format) via `scripts/migrate_week1_to_question_bank.py`.
Validation run output:
- Migrated rows: 70
- Canonical: 15
- Variants: 55 (`W2=15`, `W3=15`, `W4=15`, `W5=10`)
- Strict schema confirmed in file header and pandas check.

**Verifier/state-machine status:**
- Implemented family-specific verifier handlers under one orchestrator entrypoint in `probes/contamination/verify.py`.
- Added state-machine simulation for `blocksworld` and `mystery_blocksworld` with legal-transition checks.
- Kept regex fallback path when a prompt cannot be parsed into explicit state/goal sections.
- `tests/test_verifiers.py`: 12 passed.

**Processed-state strategy (precision update):**
- `run_behavioral_sweep.py --resume` now keys already-scored rows by
  `(problem_id, variant_type, model)` instead of `(problem_id, variant_type)` only.
- This prevents false skipping when running multiple models on the same question bank.

### Session 6
Should show only BW_080 empty. That's acceptable — 9/10 W6 rows populated is fine for the sweep. Note it in PROJECT_LOG.md and move to W5.
Good question. Let me think about this from a reviewer's perspective, not just from a "what's available" perspective.

---

## What reviewers at BlackboxNLP and EMNLP actually expect

Accepted papers in this space typically follow one of two patterns:

**Pattern A (breadth):** 4-6 models spanning open/closed and size ranges. Used when the paper's claim is "this finding generalizes across model families." Requires more API budget but stronger generalization claims.

**Pattern B (depth on fewer models):** 2-3 models, one open-weight for mechanistic work, one or two closed for behavioral comparison. Used when the paper's contribution is mechanistic — which yours is. Reviewers accept smaller model sets when the mechanistic depth justifies it.

Your paper is Pattern B. The mechanistic work (activation similarity, logit lens, patching) requires open weights and is the novel contribution. The closed models are there to show behavioral findings aren't Qwen-specific.

---

## The problem with your proposed list

You have 7 models. That is too many for your scope. Let me be direct about why.

Running 7 models × 45 problems × 5 variants = 1,575 API calls minimum, just for behavioral Probe 1. At reasoning model pricing that is $200-400+. More importantly, you then have 7 model result columns in every table and every figure. Papers with 7 models either have a large team or have shallow per-model analysis. You have neither.

Also: several of your proposed models have issues for this specific paper.

**Qwen 3 8B** — good choice for mechanistic work, open weights, manageable size.

**Llama 12B** — fine but adds a second open-weight model for mechanistic analysis. Running full activation + logit lens + patching on two open models doubles your GPU budget and your mechanistic results section. Pick one open model for mechanistic work.

**Qwen large / DeepSeek V3 / DeepSeek R1** — these are very large models. DeepSeek R1 is a reasoning model (like o1). Including reasoning models is interesting but requires a separate discussion in your paper: do reasoning models show different retrieval-vs-computation signatures? That is almost a sub-paper. Don't open that thread unless you have the budget and the argument.

**GPT-5.x mini / Claude Haiku** — haiku and mini are cost-optimized small models. For a reasoning paper, small closed models are the weakest choice because they show the most failure and the least interesting behavioral patterns. Reviewers sometimes ask "why the cheapest models?" You want at least one capable closed model.

---

## What I recommend

**Three models total. This is the right scope for your paper.**

| Model | Role | Why |
|---|---|---|
| Qwen2.5-7B | Open-weight, mechanistic anchor | Full activation analysis, logit lens, patching. Already in your codebase. Required for your mechanistic contribution. |
| Claude Sonnet 3.7 | Closed, capable | Strong reasoning model, well-documented, your paper can cite published benchmark numbers for calibration. Cost-effective vs GPT-4o. |
| GPT-4o | Closed, second behavioral point | You have pilot data on this already. Gives you a second closed-model data point. Well-known to reviewers. |

This gives you: one open (mechanistic depth) + two closed (behavioral breadth). You can say "behavioral findings replicate across two closed model families (Anthropic and OpenAI) and one open-weight model." That is a defensible claim.

---

## Why not include reasoning models (o3, R1)

Two reasons.

First, reasoning models behave differently by design — they do extended chain-of-thought internally. Your Probe 2 (plan vs execute) assumes the model's announced plan is its "reasoning." For a reasoning model that internal CoT is hidden and the announced plan is post-hoc. The CCI measurement becomes ill-defined.

Second, if you include a reasoning model and it shows different patterns from standard LLMs, you have to explain why. That explanation requires a sub-argument about reasoning model architecture that is beyond your paper's scope.

If a reviewer asks "did you test reasoning models?" — you say: "We deliberately excluded reasoning models because their internal chain-of-thought is opaque and non-comparable with the step-by-step execution protocol of Probe 2. This is a scoped-out direction for future work." That is a complete answer.

---

## If you want a fourth model

If budget allows and you want stronger generalization claims, add **Llama 3.1 8B** as a second open-weight behavioral model (not mechanistic — just behavioral CSS and CCI). This lets you say findings replicate across two open model families (Qwen and Llama) and two closed families. Four models, one mechanistic, three behavioral.

But do not do this until the three-model version is running cleanly.

---

## Final answer

| Model | Mechanistic | Behavioral | Priority |
|---|---|---|---|
| Qwen2.5-7B | Yes | Yes | Required |
| Claude Sonnet 3.7 | No | Yes | Required |
| GPT-4o | No | Yes | Required (you have pilot data) |
| Llama 3.1 8B | No | Yes | Optional, add only if budget allows |

Update your `configs/models.yaml` to reflect this. Everything else in your codebase stays the same.

### Session 7 — Blocksworld full sweep (2026-04-19)

Problem IDs: BW_001 through BW_467, MBW_001 through MBW_10 (canonical rows in triage).

Variants in behavioral sweep: canonical, W2, W3, W4, W5, W6 (BW_080/W5 gap per bank).

Models (OpenRouter): `anthropic/claude-3.7-sonnet`, `openai/gpt-4o`, `meta-llama/llama-3.1-8b-instruct`.

Key outputs:

- `results/behavioral_sweep.csv` (rescored after verifier fix; backup `results/behavioral_sweep.csv.bak`)
- `results/contamination_triage.csv` (InfiniGram mini: `v2_dclm_all`)
- `results/triangulation_per_instance.csv`
- `results/contamination_regression.txt`
- `figures/probe1_triage.png`

Bugs fixed:

- `run_behavioral_sweep.py` / `run_contamination_triage.py` `--family` matched `problem_family` instead of `problem_subtype` (silent zero rows).
- **Verifier:** numbered plans (`1. pick-up …`) were merged across lines by a global regex, so action parsing and state checks were wrong. Fixed with line-based parsing for blocksworld and mystery plans; sequence match fallback when simulation returns false but lists match.

Gate 1 (Claude 3.7 slice, *n*=15): triangulation convergence rate **0%** (all `diverging` under current alignment rule). OLS **css ~ contamination_score + C(problem_family)** ran after rescore: **positive** slope on `contamination_score` (~5.23), **p ≈ 0.22** — does **not** support the “high contamination → lower CSS” hypothesis as stated; interpret with caution (small *n*, CSS mostly low).

Next: GSM family ingestion; optional mechanistic when GPU available.

So the right framing for your paper is: canonical instances are drawn from published benchmarks deliberately, to inherit their validation and to ground the contamination scoring in real training-data exposure. 
---
## TODO: W1 Lexical Paraphrase — NOT YET DONE
Date flagged: [today's date]
Status: BLOCKED — human writing required before any commands can be run

W1 (lexical paraphrase) is missing for all 20 canonical problems:
- 15 Blocksworld: BW_001, BW_010, BW_011, BW_080, BW_120, BW_137, BW_155, BW_172, BW_227, BW_467, BW_E002, BW_E015, BW_E017, BW_E019, BW_E100
- 5 Mystery Blocksworld: MBW_001, MBW_10, MBW_100, MBW_127, MBW_185

W1 rules (do not deviate):
- Rephrase the problem statement and action descriptions in different natural language
- Block names, action names (pick-up/put-down/stack/unstack), and goal state must remain identical
- For MBW: rephrase only the wrapper prose; the nonsense predicate names (attack/succumb/overcome/feast, harmony/planet/province/pain/craves) must NOT be changed — they are the experimental manipulation
- Each W1 must be reviewed by a team member who did not write it
- Do NOT use an LLM to generate W1 variants — human-written only per research design

Once W1 variants are written and added to question_bank.csv with variant_type="W1", run these commands in order:
---

## Section 11 — Current Repo Snapshot (Apr 2026)

### Active data state

- Source of truth: `data/problems/question_bank.csv`
- Current size: **140 rows**
- Family coverage in active bank:
  - `planning_suite`: 140
  - GSM / Algorithmic: not yet ingested into this active bank
- Subtype coverage:
  - `blocksworld`: 115
  - `mystery_blocksworld`: 25
- Variant distribution:
  - `canonical=20`, `W1=20`, `W2=20`, `W3=20`, `W4=20`, `W5=15`, `W6=25`

### What each pipeline file is doing right now

- `scripts/run_behavioral_sweep.py`: Probe 1 generation + scoring entrypoint (resume-safe, supports family/subtype filtering).
- `scripts/extract_phase1_plans.py`: extracts canonical plan outputs from behavioral runs for Probe 2 input.
- `scripts/run_probe2a_cci.py`: Probe 2a closed-loop execution consistency from extracted plans.
- `scripts/run_probe2b_tep.py`: Probe 2b perturbation/cascade behavior.
- `scripts/run_contamination_triage.py`: contamination indexing over canonical rows.
- `scripts/run_triangulation.py`: merges behavioral/mechanistic/contamination evidence per problem.
- `scripts/run_mechanistic_sweep.py`: mechanistic metrics (dry-run local, real run GPU).

### Blocksworld current execution status

- Probe 1/triangulation have historical runs in `results/`.
- Probe 2 scripts now include stronger action normalization and additional diagnostics.
- One clean **final** run is still needed from a frozen bank to avoid mixed historical output files.

### Cleanup and stabilization tasks before final rerun

1. Freeze `question_bank.csv` (no ad-hoc manual edits after this point).
2. Archive legacy/mixed result files; keep one current target file per probe/model.
3. Run a deterministic coverage audit before paid calls:
   - missing `(problem_id, variant_type, model)` for behavioral
   - missing `problem_id` for contamination canonical set
4. Execute full pipeline in order:
   - behavioral (all three models, resume on)
   - extract phase1 plans
   - probe2a + probe2b
   - contamination
   - triangulation per model
5. Only after Planning Suite is stable, ingest GSM + Algorithmic families into the same unified schema.

---

## Section 12 — File Responsibility Map (Maintenance View)

This section is for day-to-day repo maintenance: what each file is responsible for, and what can be archived.

### Data sources and outputs

- `data/problems/question_bank.csv`
  - Canonical source-of-truth table for all problem rows and variants.
  - Any evaluation must read from this file; avoid hand-editing outside scripted fixes.
- `results/behavioral_sweep.csv`
  - Probe 1 output table; each row is `(problem_id, variant_type, model)` response + correctness.
  - Contains historical artifacts unless cleaned by Section-1 deletions.
- `results/probe2a_cci.csv`
  - Probe 2A output (closed-loop execution consistency).
- `results/probe2b_tep.csv`
  - Probe 2B output (error injection cascade behavior).
- `results/contamination_triage.csv`
  - Contamination indexing output for canonical instances.
- `results/triangulation_per_instance*.csv`
  - Cross-probe merged diagnosis tables (per model where applicable).
- `results/contamination_regression*.txt`
  - Regression summaries produced by triangulation.

### Primary runners (keep)

- `scripts/run_behavioral_sweep.py` — Probe 1 main runner.
- `scripts/extract_phase1_plans.py` — converts Probe 1 responses to plan inputs for Probe 2.
- `scripts/run_probe2a_cci.py` — Probe 2A run script.
- `scripts/run_probe2b_tep.py` — Probe 2B run script.
- `scripts/run_contamination_triage.py` — contamination run script.
- `scripts/run_triangulation.py` — final merge/regression script.

### Verifier/metrics internals (keep)

- `probes/contamination/verify.py` — answer validation state-machines and parsing.
- `probes/behavioral/css.py` — CSS plus VAR/PDAS/DTS/VRI/CFS metric helpers.
- `probes/triangulation/per_instance.py` — signal alignment rules (VAR-primary, CSS-secondary).

### Figure generation

- `scripts/generate_probe2_figures.py` — Probe 2 summary figures.
- `analysis/figures/` — modular figure scripts and wrappers.
- `analysis/figures/output/` — saved publication-ready graphics.

### One-off / migration scripts (archive candidates)

- `fix_csv.py`
- `scripts/run_probe1_repair_and_batches.py`
- `scripts/generate_mbw_w5_variants.py`
- `scripts/translate_w3_answers.py`
- `scripts/propagate_answers.py`
- `scripts/temp_flag_records.py`

If no longer needed for reruns, move these to `scripts/archive/` to reduce maintenance noise.

### Current known state after latest runs

- Question bank fixes A1–A7 verified in working `question_bank.csv`.
- Probe 1 coverage refreshed (including W1 and MBW W5 runs with skip-guards).
- Probe 2 outputs present (`probe2a_cci.csv`, `probe2b_tep.csv`) for target 3 models.
- Triangulation scripts updated to VAR-first analysis logic; rerun once after contamination refresh for final tables.

---

## Section 14 — Stage 3A–3E Evaluation Flow (Blocksworld + GSM)

This section documents the current end-to-end evaluation flow and the GSM-specific additions layered onto the existing Blocksworld pipeline.

### Flow overview

1. Question bank prep/fixes  
2. Probe 1 behavioral sweep  
3. Probe 1 metrics  
4. Probe 2 plan-execution coupling  
5. Probe 2 metrics  
6. Probe 3 contamination triage  
7. Triangulation (P1 + P2 + P3)  
8. Figure generation  

### Blocksworld flow (existing)

- `scripts/run_behavioral_sweep.py` (or `scripts/BW_P1_SCR_run_behavioral_sweep.py`)
- Probe 2 run scripts (`run_probe2a_cci.py`, `run_probe2b_tep.py`)
- `scripts/run_contamination_triage.py` (or `scripts/BW_P3_SCR_run_contamination_triage.py`)
- `scripts/run_triangulation.py` (or `scripts/BW_P3_SCR_run_triangulation.py`)
- figure scripts under `analysis/figures/` and generator wrappers

### GSM flow (new path)

- Bank prep:
  - `scripts/GSM_PX_SCR_fix_question_bank.py`
  - `scripts/GSM_PX_SCR_generate_w6.py`
- Probe 1:
  - `scripts/BW_P1_SCR_run_behavioral_sweep.py --family arithmetic_reasoning --question-bank-path data/problems/gsm_question_bank.csv`
  - `scripts/GSM_P1_SCR_compute_metrics.py`
- Probe 2:
  - `scripts/GSM_P2_SCR_run_probe2.py`
  - `scripts/GSM_P2_SCR_compute_metrics.py`
- Probe 3:
  - `scripts/BW_P3_SCR_run_contamination_triage.py --family arithmetic_reasoning ...`
  - `scripts/BW_P3_SCR_run_triangulation.py --family arithmetic_reasoning ...`
- Figures:
  - `scripts/GSM_P1_FIG_generate.py`
  - `scripts/GSM_P2_FIG_generate.py`
  - `scripts/GSM_P3_FIG_generate.py`

### File role map (Stage 3A–3E)

| File | What it is | New or Edit |
|------|------------|-------------|
| `scripts/GSM_PX_SCR_fix_question_bank.py` | Fixes GSM bank bugs and normalizes answers | New |
| `scripts/GSM_PX_SCR_generate_w6.py` | Generates procedural regeneration W6 rows | New |
| `probes/contamination/verify.py` | Adds `verify_gsm_answer()` and dispatcher route | Edit |
| `tests/test_verifiers.py` | GSM verifier test coverage | Edit |
| `scripts/BW_P1_SCR_run_behavioral_sweep.py` | Adds `--question-bank-path`; fixes verifier call signature; output compatibility | Edit |
| `scripts/GSM_P1_SCR_compute_metrics.py` | Computes GSM Probe 1 metric tables | New |
| `probes/behavioral/var.py` | VAR metric aggregation + CI | New |
| `probes/behavioral/gsm_metrics.py` | W4-gap, VRI, RCS-by-difficulty, step-sensitivity, CCI-by-contamination | New |
| `scripts/GSM_P2_SCR_run_probe2.py` | Probe 2 runner with numeric matching and human-approval queue logging | New |
| `scripts/GSM_P2_SCR_compute_metrics.py` | Probe 2 metric aggregation + CI summary | New |
| `scripts/BW_P3_SCR_run_contamination_triage.py` | Adds `--question-bank-path`, `--output`, `--max-ngram` | Edit |
| `scripts/BW_P3_SCR_run_triangulation.py` | Adds `--behavioral-results`, `--contamination-results`, `--probe2-results`, `--family`; wires CCI | Edit |
| `scripts/GSM_P1_FIG_generate.py` | GSM Probe 1 figure generation | New |
| `scripts/GSM_P2_FIG_generate.py` | GSM Probe 2 figure generation | New |
| `scripts/GSM_P3_FIG_generate.py` | GSM Probe 3 figure generation | New |

### Canonical GSM question bank path

Use `data/problems/gsm_question_bank.csv` as the canonical GSM bank path.  
`data/problems/question_bank_gsm.csv` should be treated as legacy/compatibility input only.