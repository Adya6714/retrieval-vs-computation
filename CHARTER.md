# Beyond Accuracy — Retrieval vs Computation in LLM Reasoning

**Authors:** Adya
**Affiliation:** BITS Pilani
**Target venues:** BlackboxNLP 2026, GenBench 2026 (primary); EMNLP Findings (stretch)
**Status:** Design locked. Problem selection and execution pending.

---

## Research Question

When an LLM produces a correct answer to a reasoning problem, is that correctness causally dependent on **surface features** of the input (words, phrasings, entity names seen in training), or on **structural features** (abstract relationships, invariant under surface change)?

> The model is a black box. Throughout the paper, we use the framing **"behavior consistent with retrieval-like processing"** vs **"behavior consistent with computation-like processing."** This framing is non-negotiable.

---

## Contribution

Three independent probes run on the same problem instances converge on the same per-instance diagnosis. The contribution is the **per-instance triangulation across three independent axes**, not any single probe. We use established benchmarks deliberately to inherit their validation.

---

## Theoretical Framework: Triangulation

The three axes of evidence are genuinely independent — each measures a different dimension:

| Axis | What it measures |
|------|-----------------|
| **Input axis** | Change what goes in; observe behavioral change |
| **Process axis** | Hold input fixed; check internal step-by-step consistency |
| **Data axis** | Use information external to the model (training data) |

A fourth direction — the **internals axis** (activation similarity, logit lens) — is not a standalone probe but a mechanistic depth check paired with each behavioral probe on Qwen2.5-7B.

---

## The Three Probes

### Probe 1 — Surface Invariance *(Input axis)*

Each canonical problem is rewritten in multiple surface variants, presented in separate sessions. If the model is surface-invariant, answers should agree across variants.

**Variants:**

| ID | Description |
|----|-------------|
| W1 | Lexical paraphrase |
| W2 | Structural reformat (prose ↔ table, etc.) |
| W3 | Entity rename |
| W4 | Formal notation *(applicable families only)* |
| W6 | Procedural regeneration (scripted, documented random seed) |
| W7 | Forward-backward reversal *(applicable families only — answer changes)* |

> W5 (distractor insertion) was considered and rejected — it adds new content rather than varying surface form, making it a different probe entirely.

**Metrics:**
- **CSS** (Consistency Surface Score): fraction of applicable variants where model answer matches the original
- **RCS** (Reversal Correctness Score): for W7 — did the model produce the correctly-reversed answer?
- **CAS** (Consistent Answer Signature): for hard-tier problems — did the model fail consistently (structural) or differently across variants (noisy)?

**Mechanistic partner (Qwen2.5-7B only):** Layer-wise cosine similarity between residual-stream activations at matched token positions across variants. Activation patching on a subset with random-position control.

| Hypothesis | Prediction |
|-----------|------------|
| Retrieval | Behavior fragile to surface change; activations diverge across variants |
| Computation | Behavior robust to surface change; activations converge in middle/late layers |

---

### Probe 2 — Plan-Execution Coupling *(Process axis)*

Applied to **Blocksworld only**.

**Two-phase protocol:**
1. **Phase 1** — Model generates a complete plan in a single response
2. **Phase 2** — In a separate session (no memory of Phase 1), model executes the same problem step by step

**Metrics:**
- **CCI** (Cross-session Commitment Index): fraction of execution steps matching the Phase 1 plan. Low CCI = plan confabulation.
- **TEP** (Trajectory Error Propagation): mid-execution, a false state statement is injected. Does the model adapt, or continue as if the corruption didn't happen? (Diagnostic only when paired with CCI.)

**Mechanistic partner (Qwen2.5-7B only):** Logit lens at action token positions in both phases. Compare the "crystallization layer" (first layer where the action becomes the top prediction) between Phase 1 and Phase 2.

> Tuned lens was considered and rejected — pretrained weights for Qwen2.5-7B don't exist publicly, and training one is a nontrivial side project. We use logit lens (Belrose et al., 2023) and acknowledge the trade-off.

**Pilot result (GPT-4o, 7 Blocksworld instances):** CCI = 0.26, TEP = 0.70. The model follows its own plan 26% of the time and adapts to corrupted states 70% of the time — suggesting the plan was decorative and execution was state-reactive.

| Hypothesis | Prediction |
|-----------|------------|
| Retrieval/confabulation | Low CCI; crystallization layers diverge between phases |
| Computation | High CCI; crystallization layers align between phases |

---

### Probe 3 — Contamination Indexing *(Data axis)*

For every problem instance, generate n-gram fingerprints from the canonical problem statement and query the **Infini-gram API** (Liu et al., COLM 2024) against The Pile and DCLM indexes. Record maximum matched n-gram length and occurrence frequency → single contamination score per instance.

> Specific fingerprinting methodology (n-gram length, sliding windows, score combination) is deferred to Phase 1. Requires reading Magar & Schwartz, Golchin & Surdeanu, and Carlini's extraction work.

**Primary analysis:** Per-instance behavioral correctness regressed on contamination score with problem-family fixed effects (OLS, statsmodels). Report coefficient, bootstrap 95% CI, and effect size.

**Mechanistic bridge (Qwen2.5-7B only):** Compare crystallization layers for high- vs low-contamination instances. Under retrieval hypothesis, high-contamination instances should crystallize earlier.

| Hypothesis | Prediction |
|-----------|------------|
| Retrieval | Strong positive correlation: contamination → correctness; earlier crystallization |
| Computation | Weak/no correlation; crystallization depth independent of contamination |

---

## Models

| Role | Models |
|------|--------|
| Mechanistic analysis | Qwen2.5-7B (fallback: OLMo-7B if tooling issues) |
| Behavioral analysis | Qwen2.5-7B + one closed-model pair: GPT-4o vs o3 **or** Claude Sonnet 3.5 vs 3.7 |

> Final closed-model pair decided at Phase 3. Currently leaning Claude on cost and API stability grounds. No mechanistic analysis on closed models — the paper is explicit about this asymmetry.

---

## Statistical Requirements *(Non-Negotiable)*

- Bootstrap 95% confidence intervals on every reported aggregate (10,000 resamples, `scipy.stats`)
- Wilcoxon signed-rank test for all paired comparisons
- Problem-family fixed effects in the contamination regression (`statsmodels` OLS)
- Random-position patching control for all activation patching experiments
- Report effect sizes, not just p-values

---

## Problem Set

### Scale

**45 total problems — 15 per family** across 3 families.

Rationale: below 25 total, CIs become too wide for defensible claims at target venues. Above 60, variant-writing labor is unmanageable for a 3-person team. Sits within the 30–100 instance norm for behavioral eval papers at BlackboxNLP/GenBench.

---

### Family 1 — Planning Suite

*Primary function: anchor Probe 2; contributes to Probes 1 and 3.*

| Sub-type | Count | Notes |
|----------|-------|-------|
| Blocksworld | 8 | 3 easy (3-4 blocks), 3 medium (5-7 blocks), 2 hard (10-12 blocks). Probes 1, 2, 3. Verifier: Fast Downward. |
| Logistics | 4 | PDDL trucks+packages. Tests within-PDDL generalization. Probes 1, 3 only. Verifier: Fast Downward. |
| Mystery Blocksworld | 3 | Kambhampati's variant: logically identical to Blocksworld, predicate names replaced by nonsense words. Probes 1, 3 only. Verifier: Fast Downward (after de-obfuscation). |

**Why Mystery Blocksworld?** A retrieval-dependent model should fail here (training targets are gone); a genuinely reasoning model should barely degrade. Novel contribution: combining Mystery Blocksworld with contamination scoring and mechanistic analysis has not been published.

**Contamination targets:**
- High: 3 classic PlanBench Blocksworld phrasings, 2 canonical Logistics instances
- Medium: 2 adapted Blocksworld, 2 adapted Logistics
- Low: 3 procedurally generated Blocksworld, 3 Mystery Blocksworld (inherently low due to nonsense predicates)

---

### Family 2 — Arithmetic Reasoning

*Primary function: Probes 1 and 3. Not used in Probe 2.*

All sub-types from Mirzadeh et al.'s GSM-Symbolic toolkit — same verifier style, same contamination approach throughout.

| Sub-type | Count | Notes |
|----------|-------|-------|
| GSM-Symbolic standard | 8 | 5 easy (2-3 step), 3 medium (4-5 step). Verifier: numeric equality with tolerance. |
| GSM-P1/P2 | 4 | Depth-increased variants with additional arithmetic steps. Hard tier. |
| GSM-NoOp | 3 | Distractor-containing variants used as naturally-hard instances, not as a variant generation method. |

> W4 (formal notation) and W7 (reversal) do not apply to this family — arithmetic word problems don't have clean formal notation counterparts or natural reverse directions.

**Contamination targets:** 5 high / 5 medium / 5 low (procedurally re-templated with fresh parameter seeds).

---

### Family 3 — Algorithmic Suite

*Primary function: Probes 1 and 3. Not used in Probe 2.*

Four algorithmic sub-types within one family — consistent variant-writing guidelines and verifier style across all. Adversarial instances are the key discriminators: constructed so retrieval-of-heuristic fails and genuine computation is required.

| Sub-type | Count | Notes |
|----------|-------|-------|
| Shortest Path | 4 | 2 standard (4-6 nodes, unique path), 2 adversarial (greedy fails). Verifier: NetworkX `shortest_path`. |
| Weighted Interval Scheduling | 4 | 2 standard, 2 adversarial (earliest-deadline-first fails). Verifier: DP solver. Pilot: CSS=0.00 motivates inclusion as a known-failure anchor. |
| Coin Change | 4 | 2 standard (canonical denominations), 2 adversarial (e.g. {1,3,4} for target 6 — greedy fails). Verifier: DP solver. |
| Knapsack | 3 | 1 standard, 2 adversarial (value/weight-ratio greedy fails). Verifier: DP solver. |

**W7 applicability:** Shortest Path (undirected graphs trivially reversible, directed with care) and Coin Change (target-to-denominations reversal is meaningful). Not applicable to WIS or Knapsack.

**Contamination targets:** 5 high / 5 medium / 5 low.

---

### Cross-Family Summary

| Family | Count | Probe 1 | Probe 2 | Probe 3 |
|--------|-------|---------|---------|---------|
| Planning Suite | 15 | All | Blocksworld subset only | All |
| Arithmetic Reasoning | 15 | All | None | All |
| Algorithmic Suite | 15 | All | None | All |
| **Total** | **45** | **45** | **8** | **45** |

---

### Variant Applicability Matrix

| Variant | Planning Suite | Arithmetic | Algorithmic |
|---------|---------------|------------|-------------|
| W1 Lexical paraphrase | All | All | All |
| W2 Structural reformat | All | All | All |
| W3 Entity rename | All | All | All |
| W4 Formal notation | Partial (BW, Logistics via PDDL) | Not applicable | All |
| W6 Procedural regeneration | All | All | All |
| W7 Reversal | Blocksworld only | Not applicable | SP and Coin Change only |

W4 and W7 are partial variants — reported on applicable subsets only, not pooled into the main CSS computation.

---

### Contamination Balance

|  | Planning Suite | Arithmetic | Algorithmic | Total |
|--|---------------|------------|-------------|-------|
| High contamination | 5 | 5 | 5 | 15 |
| Medium contamination | 5 | 5 | 5 | 15 |
| Low contamination | 5 | 5 | 5 | 15 |

---

## Execution Plan

| Phase | Description | Status |
|-------|-------------|--------|
| **0** | Infrastructure: repo, Python env, API access, compute planning | Largely complete |
| **1** | Contamination triage: ~15 problems against Infini-gram; go/no-go for Probe 3 | Pending |
| **2** | Mechanistic tooling: stand up Qwen2.5-7B, activation extraction, logit lens, sanity-check on published finding (IOI or equivalent) | Pending |
| **3** | Probe 1: finalize problem set, write/review all surface variants, run behavioral CSS on all models, mechanistic CSS on Qwen | Pending |
| **4** | Probes 2 and 3: plan-vs-execute + lens on Qwen; full contamination sweep; triangulation analysis | Pending |
| **5** | Writing: figures → methods → results → discussion → related work → intro → abstract. No "future work" section. | Pending |
| **6** | Revision and submission: internal review, critical overclaim pass, reproducibility check, anonymize | Pending |

---

## Pending Documents *(Must exist before execution begins)*

### `shasshy.md` — Instance Selection Criteria
Must specify per family: difficulty tier definitions, contamination-tier criteria, answer unambiguity requirements, source documentation requirements, exclusion criteria, and a review checklist. **Written by Adya before instance selection begins.**

### `CONTRIBUTING_VARIANTS.md` — Variant Writing Guidelines
Must specify per variant type: what is preserved, what changes, what must not change, worked examples (at least one per family), and cross-review protocol (each variant reviewed by someone who didn't write it). **Written by Adya before any variant writing begins.**

---

## Open Questions *(Resolve before their phase)*

1. Does BITS Pilani provide student GPU cluster access sufficient for Qwen2.5-7B inference?
2. What specific fingerprinting method for contamination scoring? (Requires reading Magar & Schwartz, Golchin & Surdeanu, Carlini.)
3. Does TransformerLens cleanly support Qwen2.5-7B? Alternatives: nnsight or raw HuggingFace hooks.
4. Closed-model pair: GPT-4o vs o3, or Claude Sonnet 3.5 vs 3.7? (Decide at Phase 3.)

---

## Decisions Explicitly Deferred

The following are **not settled** and must not be treated as such in any implementation work:

- Final confirmation of the three-family structure (Planning, Arithmetic, Algorithmic)
- Inclusion/exclusion of Mystery Blocksworld within the Planning Suite
- Inclusion/exclusion of Logistics within the Planning Suite
- Final mix within Family 3 (current proposal: 4 SP + 4 WIS + 4 Coin Change + 3 Knapsack)
- Final mix within Family 2 (current proposal: 8 GSM-Symbolic standard + 4 GSM-P1/P2 + 3 GSM-NoOp)
- Hard-tier instance specifications for each family

---

## What Was Considered and Rejected

### Probes and strategies
- **S3–S8** (reasoning trace audit, linked-question consistency, confidence-vs-correctness, forward-backward as standalone, skill composition, generative problem construction): each addresses a different research question. Saved for future papers.
- **A3 (adversarial distractors) as Probe 1**: same input axis as surface invariance, no pilot data, weaker mechanistic partner.
- **D1 (reversal) as a fourth probe**: same axis as Probe 1 — would break three-axis triangulation. Bundled as W7 with its own metric (RCS) instead.
- **W5 (distractor insertion) as Probe 1 variant**: adds content rather than varying surface form.
- **LLM-generated variants**: introduces systematic bias correlated with the evaluating model's training distribution. Human-written variants only.

### Problem set
- **More than three families**: dilutes statistical power per family; multiplies labor.
- **Game of 24**: cannot be meaningfully paraphrased — no surface to vary.
- **Logic grid puzzles, cryptarithmetic, river crossing, propositional SAT, graph coloring, Sokoban, Berglund sanity anchor**: outside the probes' structural needs or scope creep.
- **GSM8K separately from GSM-Symbolic**: too heavily contaminated across all instances — no dynamic range for Probe 3.
- **Single-algorithm Family 3**: over-concentrates on one algorithm's failure modes.

---

## Prior Work

**Behavioral probing:** Kambhampati et al. (2024), Mirzadeh et al. (2024), Wu et al. (NAACL 2024), Razeghi et al. (EMNLP Findings 2022), Berglund et al. (ICLR 2024), Turpin et al. (NeurIPS 2023), Lanham et al. (2023)

**Mechanistic interpretability:** Belrose et al. (NeurIPS 2023), Meng et al. (NeurIPS 2022), Wang et al. (ICLR 2023), Conmy et al. (NeurIPS 2023), Geva et al. (2021), Olsson et al. (2022)

**Contamination:** Liu et al. (COLM 2024), Xu et al. (EMNLP 2025)

**Contribution gap:** No prior paper coordinates behavioral probes, activation-level analysis, and training-data contamination indexing on multi-step reasoning tasks at the per-instance level. Prior work either runs behavioral probes without internal access, or runs mechanistic analysis on toy single-token tasks.
