# PROJECT_LOG.md

Single source of truth for codebase state, bugs, deferred decisions, and what to do when new things arrive. Read Section 6 before touching anything after a gap.

---

## Section 1 — Phase Status

| Phase | Status | Owner | Blocked On |
|-------|--------|-------|------------|
| Phase 0 — Infrastructure | Complete | Adya | — |
| Phase 1 — Contamination Triage | Code complete, not executed | Adya + Shaswat | Shaswat's CSV PR |
| Phase 2 — Mechanistic Tooling | Code complete, not executed | Adya | GPU access |
| Phase 3 — Probe 1 | Not started | — | Phase 1 gate + variant bank |
| Phase 4 — Probes 2 and 3 | Not started | — | Phase 3 gate |
| Phase 5 — Writing | Not started | All | Phase 4 gate |
| Phase 6 — Submission | Not started | — | Phase 5 gate |

---

## Section 2 — Execution Checklist

### Phase 1
1. Merge Shaswat's PR
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
| Closed model pair | GPT-4o/o3 vs Claude Sonnet 3.5/3.7 | Adya | Phase 3 start |
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
The behavioral sweep currently only runs canonical problems. CSS requires variant responses (W1-W5 per problem). This is the most important missing piece before Phase 3.

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

Run these checks **before executing any script** after Shaswat's PR merges.

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

**Blockers at end:** Shaswat's PR, GPU, `sanity_check.py` structural bug, model sweep not written.

### Session 2
**Done:** `configs/`, `.env.example`, `Makefile`, `verify.py` stub, `probes/common/io.py`, `parsers.py`, `stats.py`, `mock_client.py`, `probe1_triage_plot.py`, `run_contamination_triage.py`. Variant numbering locked W1-W6 (W5=procedural, W6=reversal). All team docs: `shasshy.md`, `nandini.md`, `CONTRIBUTING_VARIANTS.md`, `REVIEW_PROTOCOL.md`.

**Bugs identified:** `triage.py` behavioral_correct (fixed), `model_sweep.py` path + signature (queued).

### Session 3
**Done:** All five metric modules (css, rcs, cas, cci, tep). `patching.py`. `per_instance.py`. `run_mechanistic_sweep.py`. `run_triangulation.py`. `anthropic_client.py`. `openai_client.py`. Full test suite (test_verifiers, test_parsers, test_stats, test_mock_sweep). `sanity_check.py` structural bug fixed.

**Bugs fixed this session:** patching.py tensor serialization, load_model call, crystallization layer detection, run_triangulation variant_type guard, cas.py family duplication, tep.py None label.

**Pipeline status:** structurally complete. All Layer 1 code exists and is tested. Ready to execute the moment CSV + API keys arrive.
