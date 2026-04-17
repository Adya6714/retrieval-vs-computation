# PROJECT_LOG.md

## Section 1 — Phase Status

| Phase | Status | Owner | Blocked On |
|-------|--------|-------|------------|
| Phase 0 — Infrastructure | Complete | Adya | — |
| Phase 1 — Contamination Triage | In Progress | Adya + Shasshy| Shasshy's PR + GPU for behavioral sweep |
| Phase 2 — Mechanistic Tooling | Code Complete, Not Executed | Adya | GPU access |
| Phase 3 — Probe 1 | Not Started | — | Phase 1 + Phase 2 gates |
| Phase 4 — Probes 2 and 3 | Not Started | — | Phase 3 gate |
| Phase 5 — Writing | Not Started | Adya + Shaswat + Nandini | Phase 4 gate |
| Phase 6 — Submission | Not Started | — | Phase 5 gate |

---

## Section 2 — Execution Checklist

### Phase 1
1. Merge Shasshy's PR — `data/probe1_instances.csv`, `data/probe2_instances.csv`, `probes/contamination/verify.py`
2. `pip install -r requirements.txt`
3. `python scripts/test_api_keys.py` — verify Infini-gram reachable
4. `python probes/contamination/triage.py` — fills contamination scores
5. Run model sweep to fill `behavioral_correct` — script TBD, requires closed model API keys
6. `python notebooks/probe1_triage_plot.py` — Gate 1 plot (script not yet written)
7. Evaluate Gate 1: is there a visible correlation between contamination score and behavioral accuracy?

### Phase 2
1. `python scripts/check_transformerlens_support.py` — no GPU needed, do this on laptop now
2. Sort GPU access — RunPod ($25 credits) or BITS cluster
3. Uncomment `transformer_lens` in `requirements.txt`, run `pip install -r requirements.txt`
4. `python probes/mechanistic/load_model.py` — confirms Qwen2.5-7B loads cleanly
5. Fix structural bug in `sanity_check.py` — `extract_activations` calls are outside function body
6. `python probes/mechanistic/sanity_check.py` — Gate 2, must pass both tests

---

## Section 3 — Open Decisions

| Decision | Options | Blocked On | Owner |
|----------|---------|------------|-------|
| Closed model pair | GPT-4o/o3 vs Claude Sonnet 3.5/3.7 | Phase 3 start | Adya |
| GPU compute source | RunPod credits vs BITS cluster vs Colab T4 | Adya to confirm | Adya |
| Tuned lens weights for Qwen2.5-7B | Pretrained if available vs logit lens fallback | Phase 2 execution | Adya |
| TransformerLens Qwen2.5 architecture | Confirmed supported vs nnsight fallback | Run check script | Adya |

---

## Section 4 — Scripts Written, Not Yet Executed

| Script | Purpose | Can Run Locally | Needs GPU |
|--------|---------|----------------|-----------|
| `scripts/test_api_keys.py` | Verify API access | Yes | No |
| `scripts/check_transformerlens_support.py` | Check Qwen2.5 in TL registry | Yes | No |
| `probes/contamination/infinigram_client.py` | Infini-gram HTTP client | Yes | No |
| `probes/contamination/score.py` | Contamination scorer | Yes | No |
| `probes/contamination/triage.py` | Full triage runner | Yes (needs CSV) | No |
| `probes/mechanistic/load_model.py` | Load Qwen2.5-7B | No | Yes |
| `probes/mechanistic/activations.py` | Extract residual stream | No | Yes |
| `probes/mechanistic/similarity.py` | Layer cosine similarity | Yes (numpy only) | No |
| `probes/mechanistic/logit_lens.py` | Logit lens per layer | No | Yes |
| `probes/mechanistic/sanity_check.py` | Gate 2 verification | No (needs fix first) | Yes |

---

## Section 5 — Shasshy's PR Spec

File: `CONTRIBUTING_PROBLEM_SET.md` committed to repo.

Shasshy needs to deliver:
- `data/probe1_instances.csv` — 15 problems, families: blocksworld, shortest_path, gsm. Columns: problem_id, problem_family, problem_text, correct_answer, source_dataset, source_instance_id, contamination_pole
- `data/probe2_instances.csv` — 10 Blocksworld problems, 3-6 blocks. Columns: problem_id, problem_text, initial_state, goal_state, correct_plan, num_blocks, source_dataset, source_instance_id, contamination_pole
- `probes/contamination/verify.py` — function `verify_answer(model_response, correct_answer, problem_family) -> bool`. Logic per family: gsm=last number extraction, shortest_path=sequence match, blocksworld=ordered move match. Must not crash on bad input.

PR title: `[data+verify] Phase 1 problem set and answer verifier`

---

## Section 6 — Work Log

### Session 1
**Present:** Adya  
**Phases covered:** 0, 1, 2 (code only)  
**Done:**
- Full repo structure created and zipped
- `.cursorrules`, `.gitignore`, `requirements.txt`, `CHARTER.md`, `README.md`
- Result CSV schemas for all three probes
- `CONTRIBUTING_PROBLEM_SET.md` for Shasshy
- Phase 1: `infinigram_client.py`, `score.py`, `triage.py` written and reviewed
- Phase 2: `load_model.py`, `activations.py`, `similarity.py`, `logit_lens.py`, `sanity_check.py` written and reviewed
- Library decision: TransformerLens (Qwen2.5-7B confirmed supported)
- Decided against poetry/pyproject.toml — research repo, requirements.txt is sufficient

**Blockers:**
- Shasshy's PR not yet submitted
- GPU access not confirmed
- `sanity_check.py` has structural bug (function body split incorrectly)
- `check_transformerlens_support.py` not yet written
- Phase 1 triage plot notebook not yet written
- Model sweep script (to fill behavioral_correct) not yet written

**Next session: first task**
- Fix `sanity_check.py` bug in Cursor
- Write `check_transformerlens_support.py` in Cursor
- Confirm GPU plan (RunPod vs BITS)
- Wait for Shasshy's PR then review and merge