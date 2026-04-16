# Project Charter: retrieval-vs-computation

**Version:** 1.0  
**Date:** 2026  
**Status:** Active

---

## Research Question

When a language model solves a structured reasoning problem correctly, how much of that solution comes from reasoning through the problem's structure — and how much comes from pattern-matching against problems seen during training?

More precisely: is a model's correct answer causally dependent on surface features of the input (specific words, entity names, phrasings that appeared in training data) or on structural features (abstract relationships between elements, invariant under surface change)?

---

## One-Sentence Contribution

Behavioral signatures of retrieval-like processing co-occur with mechanistic signatures of retrieval-like processing at the per-instance level, and both correlate with training-data contamination — providing three independent lines of evidence that converge on the same diagnosis for each problem instance.

---

## What This Paper Is Not

- It is **not** a claim about "what the model is really doing" internally. All framing uses "behavior consistent with X" not "the model does X."
- It is **not** a new benchmark. Problem instances are drawn from existing datasets (PlanBench, GSM8K, Game of 24, BIG-Bench).
- It is **not** a comprehensive multi-model study. Mechanistic analysis is scoped to Qwen2.5-7B only. Behavioral analysis extends to one closed matched pair.

---

## Three Probes

### Probe 1 — Surface Invariance
- **Behavioral:** Consistency Score (CSS) across 6 surface variants of each problem (original, lexical paraphrase, structural reformat, entity rename, formal notation, procedurally-generated novel instance).
- **Mechanistic (Qwen2.5-7B only):** Layer-wise cosine similarity of residual-stream activations across variants; activation patching with random-position control.
- **Prediction:** Retrieval-like processing → behavioral consistency driven by surface overlap, not structural invariance; middle-layer activations diverge across semantically-equivalent but surface-different variants.

### Probe 2 — Plan-Execution Coupling
- **Behavioral:** Cross-Context Inconsistency (CCI) and Task-Error-Propagation (TEP) on Blocksworld planning.
- **Mechanistic (Qwen2.5-7B only):** Tuned lens (or logit lens fallback) on action token positions; crystallization layer comparison between plan-generation and step-execution.
- **Prediction:** Retrieval-like processing → plan confabulation (low CCI); execution crystallizes at different layers than plan generation.

### Probe 3 — Contamination Indexing
- **External data:** Infini-gram n-gram fingerprinting against The Pile and DCLM.
- **Statistical:** Per-instance behavioral correctness regressed on contamination score with problem-family fixed effects.
- **Mechanistic bridge:** High-contamination instances should crystallize earlier (shallower computation) than low-contamination instances.
- **Triangulation:** If all three probes agree at the per-instance level, that convergence is the paper's core finding.

---

## Models

| Role | Models |
|------|--------|
| Mechanistic analysis | Qwen2.5-7B (primary), OLMo-7B (fallback) |
| Behavioral sweep | Qwen2.5-7B + one closed pair (GPT-4o/o3 OR Claude Sonnet 3.5/3.7) |
| Closed pair decision | TBD — decide before Phase 3, commit and don't revisit |

---

## Submission Target

| Venue | Priority | Deadline (approx.) |
|-------|----------|-------------------|
| BlackboxNLP 2026 (EMNLP workshop) | Primary | ~August 2026 |
| GenBench 2026 (EMNLP workshop) | Secondary | ~August 2026 |
| EMNLP 2026 Findings | Stretch | ~June 2026 |

---

## Division of Labor

*(Fill in with actual names and responsibilities — this is the version to avoid the "wait, I thought I was doing that" fight)*

| Task | Owner | Reviewer |
|------|-------|----------|
| Probe 1 — variant generation | | |
| Probe 1 — behavioral sweep code | | |
| Probe 1 — mechanistic code (Qwen) | | |
| Probe 2 — behavioral sweep code | | |
| Probe 2 — mechanistic code (Qwen) | | |
| Probe 3 — Infini-gram pipeline | | |
| Probe 3 — regression analysis | | |
| Figure generation (all probes) | | |
| Paper writing — Methods | | |
| Paper writing — Results | | |
| Paper writing — Related work | | |
| Paper writing — Intro + Abstract | | |
| LaTeX / formatting | | |
| Final reproducibility check | | |

---

## Non-Negotiable Principles

1. **Language discipline:** "behavior consistent with retrieval-like processing" — never "the model retrieves" or "the model reasons."
2. **Bootstrap CIs on every aggregate number** (10k resamples, scipy.stats).
3. **Effect sizes alongside p-values** — no bare p-values.
4. **Automated verification** for all evaluation strategies.
5. **Frozen problem sets** — `data/probe1_instances.json` and `data/probe2_instances.json` committed and never changed after Phase 3 begins.
6. **No future-work section** — limitations addressed in-paper as in-progress or scoped-out.
7. **Overclaiming is the primary failure mode** — before submission, one full read with a red pen looking only for overstrong claims.

---

## Phase Gates (do not skip)

| Gate | Condition |
|------|-----------|
| Gate 0 | Repo live, env works, API keys tested, compute plan decided |
| Gate 1 | Contamination triage shows clear or revise-and-retry signal (not flat after revision) |
| Gate 2 | All four mechanistic tools verified on a toy task |
| Gate 3 | Probe 1 results CSV complete, at least one figure sketched |
| Gate 4 | All three probes have results CSVs; triangulation analysis done |
| Gate 5 | Full draft with every figure captioned and every number CI'd |
| Gate 6 | Submitted; repo tagged v1-submission |

---

## Research Decisions Still Open

- [ ] Closed model pair: GPT-4o/o3 vs Claude Sonnet 3.5/3.7 (decide before Phase 3)
- [ ] Mechanistic library: TransformerLens vs nnsight vs raw hooks (verify Qwen2.5-7B support, Phase 2)
- [ ] Lens type: Tuned lens (if pretrained weights exist for Qwen2.5-7B) vs logit lens fallback (Phase 2)
- [ ] Fingerprint method: Finalize after reading Magar & Schwartz, Golchin & Surdeanu, Carlini (Phase 1)
- [ ] Compute: BITS GPU cluster access — confirm whether adequate for Qwen2.5-7B inference
