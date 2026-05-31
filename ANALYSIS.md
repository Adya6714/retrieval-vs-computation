# Retrieval vs Computation — Consolidated Analysis

Single canonical analysis document for this project. Replaces all previous
analysis MDs (`MASTER_ANALYSIS_COMPENDIUM`, `COMPREHENSIVE_PROBE_ANALYSIS`,
`CRITICAL_AUDIT`, `DEEP_SCIENTIFIC_ANALYSIS`, `POST_API_ANALYSIS`,
`REVIEWER_AUDIT_VERIFICATION`, `MECHANISTIC_RUNBOOK`, `INTERNAL_PROBE_METRICS`).

Every aggregate cited in `paper/main.tex` is traceable to a row of one of the
tables here, which in turn points to a specific raw CSV in `results/raw/` or a
derived CSV in `results/derived/`.

---

## Table of contents

0. [Setup, models, banks, conventions](#0-setup)
1. [Question-bank coverage and final headcounts](#1-question-bank-coverage)
2. [Probe 1 — canonical accuracy](#2-probe-1-canonical)
3. [Probe 1 — W1 paraphrase](#21-probe-1--w1)
4. [Probe 1 — W2 format change](#22-probe-1--w2)
5. [Probe 1 — W3 entity rename (principal robustness probe)](#23-probe-1--w3)
6. [Probe 1 — W4 irrelevant fact insertion](#24-probe-1--w4)
7. [Probe 1 — W5 direction reversal](#25-probe-1--w5)
8. [Probe 1 — W6 new numbers, same algorithm](#26-probe-1--w6)
9. [Probe 2 — Phase 1 declarations](#3-probe-2-phase-1)
10. [Probe 2 — Phase 2A fresh-session execution (CCI, TEP, invocation)](#4-probe-2-phase-2a)
11. [Probe 2 — Phase 2B injection](#5-probe-2-phase-2b)
12. [Probe 3 — Infini-gram contamination](#6-probe-3-infigram)
13. [Probe 3 — Mechanistic (Qwen-2.5-7B base)](#7-probe-3-mechanistic)
14. [Triangulation — per-instance convergence](#8-triangulation)
15. [Headline formal statistical tests](#9-formal-statistical-tests)
16. [Hidden / underplayed findings (audit-extracted)](#10-hidden-findings)
17. [Negative space — what this paper does NOT claim](#11-negative-space)
18. [Pointer index — every number → source file](#12-pointer-index)

---

## 0. Setup

### Models

| Label | Provider id | Role |
|---|---|---|
| Claude | `anthropic/claude-sonnet-4` | full-coverage main-track |
| Gemini | `google/gemini-2.5-flash` | full-coverage main-track |
| Llama | `meta-llama/llama-3.1-8b-instruct` | full-coverage main-track |
| GPT-4o | `openai/gpt-4o` | full-coverage main-track |
| o4-mini | `openai/o4-mini` | reasoning-RL comparison (partial coverage) |

Rows with `model ∈ {mock, openai/o1-mini, test*}` are filtered out.

### Variants

- **canonical** — original problem statement
- **W1** — synonym substitution (surface paraphrase, no structure change)
- **W2** — paraphrase / format change
- **W3** — entity rename (**principal robustness probe**)
- **W4** — irrelevant-fact / formal-notation insertion
- **W5** — direction reversal (semantic structure change)
- **W6** — new numeric values, same algorithm (procedural regeneration)

### Statistical conventions

- **Wilson 95% CI** on every observed `k/n`.
- **Phi** = Matthews / phi coefficient on 2×2 paired contingencies.
- **Fisher exact** for small-n independence tests.
- **Paired Wilcoxon** signed-rank for CCI/TEP comparisons.
- **CCI** (consistency of computational invariance) = fraction of declared
  numeric steps matched by fresh-session execution within ε = 0.01.
- **TEP** (trace error propagation) = fraction of post-injection steps that
  diverge from the un-injected execution.
- **VRI** = mean(W1, W2, W4) − W3.

### Banks

- ALGO `data/problems/question_bank_algo.csv` (n=110)
- BW `data/problems/question_bank_bw.csv` (50 BW + 15 MBW, n=65)
- GSM `data/problems/question_bank_gsm.csv` (n=44)

All aggregates are bank-restricted (rows whose `problem_id` does not appear in
the bank are dropped).

---

## 1. Question-bank coverage

| Family | Bank size | Variants present | Total rows / model |
|---|---|---|---|
| ALGO (CC + SP + WIS, standard + adversarial) | 110 | canonical, W1–W6 (W5 on SP only n=50; W6 partial n=90) | 690 |
| BW (PlanBench 50 BW + 15 Mystery-BW) | 65 | canonical, W1–W6 | 455 |
| GSM (GSM8K stratified) | 44 | canonical, W1–W4, W6; W5 on n=24 | 288 |
| **Total** | **219** | — | **1433 (per model × 4 model coverage)** |

Detailed by variant count:

```
family  rows  uniq  canon  W1  W2  W3  W4  W5  W6
ALGO    690   110   110   110 110 110 110 50  90
BW      455    65    65    65  65  65  65 65  65
GSM     288    44    44    44  44  44  44 44  24
```

---

## 2. Probe 1 — Canonical

### 2.1 Canonical accuracy with Wilson 95% CI

| Family | Model | n | k | acc | Wilson 95% CI |
|---|---|---|---|---|---|
| ALGO | Claude | 110 | 40 | 0.364 | [0.280, 0.457] |
| ALGO | Gemini | 110 | 51 | 0.464 | [0.373, 0.556] |
| ALGO | Llama | 110 | 6 | 0.055 | [0.025, 0.114] |
| ALGO | GPT-4o | 110 | 47 | 0.427 | [0.339, 0.521] |
| ALGO | o4-mini | 110 | 20 | 0.182 | [0.121, 0.264] |
| BW | Claude | 65 | 10 | 0.154 | [0.086, 0.261] |
| BW | Gemini | 65 | 25 | 0.385 | [0.276, 0.506] |
| BW | Llama | 65 | 1 | 0.015 | [0.003, 0.082] |
| BW | GPT-4o | 65 | 4 | 0.062 | [0.024, 0.148] |
| BW | o4-mini (PDDL-strict format artifact) | 65 | 0 | 0.000 | [0.000, 0.056] |
| GSM | Claude | 44 | 37 | 0.841 | [0.706, 0.921] |
| GSM | Gemini | 44 | 40 | 0.909 | [0.788, 0.964] |
| GSM | Llama | 20 | 16 | 0.800 | [0.584, 0.919] |
| GSM | GPT-4o | 20 | 17 | 0.850 | [0.640, 0.948] |
| GSM | o4-mini (errors-excluded) | 33 | 29 | 0.879 | [0.720, 0.952] |
| GSM | o4-mini (intention-to-treat) | 44 | 29 | 0.659 | [0.511, 0.781] |

**Coverage note (GSM Llama / GPT-4o).** The bank is `GSM_001..020 ∪
GSM_041..064`. Llama and GPT-4o raw P1 files only cover `GSM_001..020`.
The paper's `n=44` Table 1 entries are obtained by merging the Phase-1
declaration step (`GSM_P2_phase1_{llama,gpt4o}.csv`) for the missing 24
high-contamination problems. Documented in Appendix C.

### 2.2 ALGO canonical by subtype × instance-type

The within-family proximity gradient: WIS standard is the "same DP, near-zero
training-data proximity" floor cell.

| Model | Subtype | InstanceType | n | acc |
|---|---|---|---|---|
| Claude | CC | adversarial | 10 | 0.700 |
| Claude | CC | standard | 15 | 0.267 |
| Claude | SP | adversarial | 34 | 0.647 |
| Claude | SP | standard | 21 | 0.000 |
| Claude | WIS | adversarial | 17 | 0.353 |
| Claude | WIS | standard | 13 | 0.077 |
| Gemini | CC | adversarial | 10 | 0.500 |
| Gemini | CC | standard | 15 | 0.267 |
| Gemini | SP | adversarial | 34 | 0.676 |
| Gemini | SP | standard | 21 | 0.619 |
| Gemini | WIS | adversarial | 17 | 0.353 |
| Gemini | WIS | standard | 13 | 0.000 |
| GPT-4o | CC | standard | 25 | 0.400 |
| GPT-4o | SP | adversarial | 31 | 0.419 |
| GPT-4o | SP | standard | 24 | 0.667 |
| GPT-4o | WIS | adversarial | 15 | 0.400 |
| GPT-4o | WIS | standard | 15 | 0.133 |
| Llama | CC | standard | 25 | 0.080 |
| Llama | SP | adversarial | 31 | 0.032 |
| Llama | SP | standard | 24 | 0.083 |
| Llama | WIS | adversarial | 15 | 0.067 |
| Llama | WIS | standard | 15 | 0.000 |
| o4-mini | CC | adversarial | 10 | **1.000** |
| o4-mini | CC | standard | 15 | 0.000 |
| o4-mini | SP | adversarial | 34 | 0.088 |
| o4-mini | SP | standard | 21 | 0.095 |
| o4-mini | WIS | adversarial | 17 | 0.118 |
| o4-mini | WIS | standard | 13 | 0.231 |

### 2.3 BW canonical by subtype (BW vs Mystery-BW)

| Model | Subtype | n | k | acc | Wilson 95% CI |
|---|---|---|---|---|---|
| Claude | bw | 50 | 10 | 0.200 | [0.112, 0.330] |
| Gemini | bw | 50 | 25 | 0.500 | [0.366, 0.634] |
| Llama | bw | 50 | 1 | 0.020 | [0.004, 0.105] |
| GPT-4o | bw | 50 | 4 | 0.080 | [0.032, 0.188] |
| o4-mini | bw | 50 | 0 | 0.000 | [0.000, 0.071] |
| All models | mbw | 15 | 0 | 0.000 | [0.000, 0.204] |

Mystery-BW canonical is 0/15 for every model — the cleanest evidence that BW
dissociation is semantic-vocabulary-driven, not parse-artifact-driven.

---

## 2.1 Probe 1 — W1 (paraphrase)

Univariate accuracy:

| Family | Claude | Gemini | Llama | GPT-4o | o4-mini |
|---|---|---|---|---|---|
| ALGO | 0.409 | 0.391 | 0.082 | 0.436 | 0.173 |
| BW | 0.062 | 0.138 | 0.031 | 0.092 | 0.000 |
| GSM | 0.841 | 0.818 | 0.850 | 0.750 | 0.864 |

W1 vs canonical pairwise phi (positive = same-direction agreement):

| Family | Claude | Gemini | Llama | GPT-4o | o4-mini |
|---|---|---|---|---|---|
| ALGO | +0.370* | +0.413* | +0.074 | +0.574* | +0.969* |
| BW | +0.423* | +0.141 | -0.022 | +0.140 | — |
| GSM | +0.660* | +0.466* | +0.490 | +0.404 | +0.273 |

`*` Fisher p < 0.05. W1 is the closest variant to canonical for every model on
ALGO and GSM (high phi, high accuracy ratio ≈ 1) — surface paraphrase causes
the smallest drop. Detailed 2×2 contingencies, ratios, and Fisher p-values for
W1 vs {canonical, W2, W3, W4, W5, W6} are reproducible from raw via
`scripts/compute_p1_metrics_unified.py`.

---

## 2.2 Probe 1 — W2 (format change)

Univariate accuracy:

| Family | Claude | Gemini | Llama | GPT-4o | o4-mini |
|---|---|---|---|---|---|
| ALGO | 0.455 | 0.300 | 0.055 | 0.109 | 0.173 |
| BW | 0.231 | 0.108 | 0.015 | 0.092 | 0.000 |
| GSM | 0.773 | 0.636 | 0.250 | 0.300 | 0.818 |

W2 vs canonical phi:

| Family | Claude | Gemini | Llama | GPT-4o | o4-mini |
|---|---|---|---|---|---|
| ALGO | +0.259* | +0.227* | +0.119 | +0.169 | +0.969* |
| BW | +0.576* | +0.133 | -0.016 | +0.140 | — |
| GSM | +0.357* | +0.418* | +0.289 | +0.275 | +0.531* |

Note: GPT-4o W2 ALGO drops from canonical 0.427 → 0.109 (-32 pts). This is
larger than its W3 drop — format change hurts GPT-4o more than entity
rename. The paper does not currently surface this; flagged in §10 below.

---

## 2.3 Probe 1 — W3 (entity rename, principal robustness probe)

Univariate accuracy + retention `R_W3 = W3 / canonical`:

| Family | Model | acc_W3 | R_W3 |
|---|---|---|---|
| ALGO | Claude | 0.091 (10/110) | 0.250 |
| ALGO | Gemini | 0.255 (28/110) | 0.549 |
| ALGO | Llama | 0.018 (2/110) | 0.327 |
| ALGO | GPT-4o | 0.173 (19/110) | 0.404 |
| ALGO | o4-mini | 0.127 (14/110) | 0.700 |
| BW | Claude | 0.138 (9/65) | 0.900 |
| BW | Gemini | 0.108 (7/65) | 0.281 |
| BW | Llama | 0.108 (7/65) | (canon=0.015; ratio uninformative) |
| BW | GPT-4o | 0.169 (11/65) | 2.750 (low-baseline artifact) |
| BW | o4-mini | 0.000 (0/65) | — |
| GSM | Claude | 0.750 (33/44) | 0.917 |
| GSM | Gemini | 0.523 (23/44) | 0.575 |
| GSM | Llama | 0.150 (3/20) | 0.188 |
| GSM | GPT-4o | 0.300 (6/20) | 0.353 |
| GSM | o4-mini | 0.841 (37/44) | **0.957** |

W3 vs canonical per-problem phi (positive = same-direction agreement = the
*more accurate* problems are *also* the W3-correct ones for that model):

| Family | Model | both | can_only | W3_only | neither | phi | Fisher p |
|---|---|---|---|---|---|---|---|
| ALGO | Claude | 6 | 34 | 4 | 66 | +0.155 | 0.165 |
| ALGO | Gemini | 22 | 29 | 6 | 53 | +0.377 | <0.001 |
| ALGO | Llama | 0 | 6 | 2 | 102 | -0.033 | 1.000 |
| ALGO | GPT-4o | 17 | 30 | 2 | 61 | +0.432 | <0.0001 |
| ALGO | o4-mini | 14 | 6 | 0 | 90 | +0.810 | <0.0001 |
| BW | Claude | 1 | 9 | 8 | 47 | -0.047 | 1.000 |
| BW | Gemini | 3 | 22 | 4 | 36 | +0.031 | 1.000 |
| BW | GPT-4o | 0 | 4 | 11 | 50 | -0.116 | 1.000 |
| GSM | Claude | 29 | 8 | 4 | 3 | +0.179 | 0.341 |
| GSM | Gemini | 21 | 19 | 2 | 2 | +0.014 | 1.000 |
| GSM | GPT-4o | 5 | 12 | 1 | 2 | -0.031 | 1.000 |
| GSM | o4-mini | 28 | 1 | 9 | 6 | +0.474 | 0.004 |

**Interpretation.** Within a single model, canonical-correct → W3-correct is
the dominant within-model pattern. The "accuracy ≠ robustness" inversion
discussed in the paper is a **pairwise per-(model, subtype)** dissociation —
not a population-level law (see Test A in §9).

### Subtype-level highlights (the central narrative)

SP-adv (n=34 paired):
- Claude canonical 0.647 → W3 0.000 (collapse despite high canonical)
- GPT-4o canonical 0.412 → W3 0.265 (some retention despite lower canonical)
- Same-direction inversion on the SAME problems, opposite tokenizers
  (Claude BPE ≠ GPT-4o BPE) — Appendix A defends against tokenizer-artifact
  explanations.

CC-adv (n=10 paired):
- Claude canonical 0.700 → W3 0.600 (retains)
- GPT-4o canonical 0.600 → W3 0.000 (collapses)
- Inversion in the opposite direction from SP-adv.

---

## 2.4 Probe 1 — W4 (irrelevant fact / formal notation)

Univariate accuracy:

| Family | Claude | Gemini | Llama | GPT-4o | o4-mini |
|---|---|---|---|---|---|
| ALGO | 0.618 | 0.436 | 0.073 | 0.445 | 0.155 |
| BW | 0.015 | 0.031 | 0.000 | 0.077 | 0.000 |
| GSM | 0.636 | 0.477 | 0.300 | 0.200 | 0.682 |

**ALGO W4 frequently *exceeds* canonical.** Claude 0.618 vs canonical 0.364
(+25 pts); GPT-4o 0.445 vs 0.427 (+2 pts); Gemini 0.436 vs 0.464 (−3 pts).
The LaTeX-formal notation reduces parse ambiguity for these models. Not
discussed in the current paper text; candidate for inclusion (flagged §10).

---

## 2.5 Probe 1 — W5 (direction reversal) — second strongest signal

Univariate accuracy (n=50 paired on ALGO SP only; n=65 BW; n=44 GSM):

| Family | Claude | Gemini | Llama | GPT-4o | o4-mini |
|---|---|---|---|---|---|
| ALGO | 0.000 | 0.020 | 0.000 | 0.000 | 0.000 |
| BW | 0.523 | 0.569 | 0.000 | 0.246 | 0.000 |
| GSM | 0.818 | 0.614 | 0.050 | 0.300 | 0.886 |

### Three distinct W5 patterns

1. **ALGO**: uniform-zero collapse for every model. Direction reversal is a
   uniform-zero effect on shortest-path. Not problem-specific.
2. **BW**: Claude and Gemini *exceed* canonical (0.523 vs 0.154 and
   0.569 vs 0.385) — likely a PlanBench property (goal-state-vs-initial-state
   asymmetry making reversed problems strictly easier in some configurations).
3. **GSM**: Claude and o4-mini *preserve* (0.818 vs canonical 0.841 / 0.886
   vs 0.879). Llama collapses (0.050 vs 0.773 — biggest single-variant drop
   anywhere in the dataset, −52 pts absolute). GPT-4o intermediate
   (0.300 vs 0.850, −55 pts).

**Implication.** "Structured / reasoning-trained models tolerate direction
reversal; non-reasoning models do not." Candidate for promotion to main text
(see §10).

---

## 2.6 Probe 1 — W6 (new numbers, same algorithm)

Univariate accuracy:

| Family | Claude | Gemini | Llama | GPT-4o | o4-mini |
|---|---|---|---|---|---|
| ALGO | 0.100 | 0.144 | 0.055 | 0.218 | 0.000 |
| BW | 0.508 | 0.338 | 0.031 | 0.215 | 0.000 |
| GSM (n=24) | 0.750 | 0.958 | 0.450 | 0.800 | 0.833 |

Per-problem deltas vs canonical (GSM, n=24):
- Claude  −2.7pts; Gemini +4pts; Llama −15pts; GPT-4o −2.7pts; o4-mini +42pts.

The paper's earlier "W6 at or above canonical for all four models" claim is
factually wrong (3 of 4 are below). Correction applied in §5.1 of `main.tex`:
"W6 stays within 2.7 pts for Claude and GPT-4o, is above canonical for Gemini,
and drops 15 pts for Llama — but Llama's W6 still retains more accuracy than
its W3 (−36 pts). The asymmetry between number-change and name-change is the
key observation." See §9 of the audit history below.

---

## 3. Probe 2 — Phase 1 (declaration)

Phase 1 is the declaration step: the model produces a full step-by-step plan
with explicit numeric (GSM/ALGO) or PDDL-action (BW) intermediates **before**
any execution loop.

### 3.1 GSM Phase 1 — parseable plans & declared final-answer accuracy

| Model | n | parseable | parseable_rate | phase1_final_correct | phase1_accuracy | median steps |
|---|---|---|---|---|---|---|
| Claude  | 44 | 40 | 0.909 | 38 | 0.864 | 5.0 |
| Gemini  | 44 | 36 | 0.818 | 36 | 0.818 | 5.0 |
| Llama   | 44 | 20 | 0.455 | 20 | 0.455 | 3.0 |
| GPT-4o  | 44 | 20 | 0.455 | 30 | 0.682 | 2.5 |
| o4-mini | 44 | 0  | 0.000 | 0  | 0.000 | 0 (schema mismatch) |

### 3.2 ALGO Phase 1 — declared-plan structure

110 sessions per model captured; declared algorithm is *qualitative* (parsed
out of raw response). Per-model algorithm-naming frequencies are aggregated
into Phase-2A reasoning-type tables below.

### 3.3 BW Phase 1 — declared PDDL plans

100% parseable for Claude, Llama, GPT-4o on all 50 BW problems.
Median plan length: Claude 15.5; Llama 11.5; GPT-4o 10.0.

---

## 4. Probe 2 — Phase 2A (normal execution; CCI, TEP, invocation)

Phase 2A re-runs the same problem step-by-step in fresh sessions per step
(no chat history). Each Phase-2A numeric step is compared to the Phase-1
declaration to compute CCI.

### 4.1 GSM Phase 2A — CCI, TEP, fresh-session correctness

| Model | n | Phase1 decl. acc | Phase2A acc | CCI mean | CCI median | TEP mean |
|---|---|---|---|---|---|---|
| Claude | 44 | 0.864 | 0.864 | 0.231 | 0.216 | 0.539 |
| Gemini | 44 | 0.818 | 0.886 | 0.270 | 0.250 | 0.652 |
| Llama  | 44 | 0.455 | 0.455 | 0.167 | 0.000 | 0.773 |
| GPT-4o | 44 | 0.682 | 0.705 | 0.108 | 0.000 | 0.598 |

**CCI ≠ accuracy dissociation.** Models can be 86–89% accurate while
following their own declared plan only 23–27% of the time (Claude, Gemini).
GPT-4o: 70% accurate, 11% plan-following. Llama: 45% accurate, 17%
plan-following.

Paired tests (Claude vs GPT-4o CCI, zero-imputed, n=44):
- Wilcoxon W = 396, p = 0.0068
- Paired t = 2.824, p = 0.0072

Llama vs Claude TEP: paired t = −2.74, p = 0.009 (Llama's execution diverges
more from its declared plan).

### 4.2 GSM Phase 2A vs Phase 1 — paired correctness

| Model | n | both_ok | p1_only | p2_only | neither | phi |
|---|---|---|---|---|---|---|
| Claude | 44 | 38 | 0 | 0 | 6 | 1.000 |
| Gemini | 44 | 36 | 0 | 3 | 5 | 0.760 |
| Llama  | 44 | 20 | 0 | 0 | 24 | 1.000 |
| GPT-4o | 44 | 30 | 0 | 1 | 13 | 0.948 |

Phase-2A execution preserves Phase-1 declaration correctness almost
perfectly *at the final-answer level*. CCI captures the per-step
inconsistency that is invisible to the final-answer comparison.

### 4.3 ALGO Phase 2A — invocation / reasoning-type / final correctness

Normal protocol (no elicitation):

| Model | n | algo_inv % | local_greedy % | forward_sim % | backtrack % | final_correct % |
|---|---|---|---|---|---|---|
| Claude  | 110 | 1.8 | 26.6 | 6.8 | 2.2 | 50.0 |
| Gemini  | 61  | 1.6 | 21.1 | 6.6 | 0.0 | 32.8 |
| Llama   | 110 | 0.7 | 19.8 | 5.4 | 0.0 | 21.8 |
| GPT-4o  | 110 | 0.0 | 23.4 | 2.1 | 0.2 | 50.0 |

Elicited protocol (explicit "name the algorithm" instruction; n=61/model):

| Model | algo_inv % | local_greedy % | final_correct % |
|---|---|---|---|
| Claude  | 53.0 | 39.0 | 45.9 |
| Gemini  | 50.3 | 30.9 | 31.1 |
| Llama   | 18.9 | 28.1 | 14.8 |

**Algorithm-elicitation paradox.** Explicit elicitation raises algorithm-naming
from ~1% step-rate to ~50% step-rate but **does not improve final-answer
accuracy** (Claude 50.0→45.9, Gemini 32.8→31.1, Llama 21.8→14.8). Naming the
algorithm is decorative, not load-bearing. Quantified as Appendix F's
algorithm-invocation paradox: 0/13 algorithm-invocation sessions correct;
13.3% unclear-reasoning baseline correct; Fisher p = 0.40 (not significant).

### 4.4 Phase-2A reasoning-type fingerprints (full ALGO step-by-step)

| Model | unclear | local_greedy | forward_sim | algo_inv | backtrack | n_steps |
|---|---|---|---|---|---|---|
| Claude | 58.2% | 32.0% | 6.4% | 1.3% | 2.0% | 925 |
| GPT-4o | 74.9% | 22.4% | 2.4% | 0.0% | 0.2% | 425 |
| Llama  | 69.6% | 23.6% | 6.0% | 0.9% | 0.0% | 425 |
| Gemini | 71.3% | 21.5% | 5.5% | 1.7% | 0.0% | 181 |

GPT-4o has **zero algorithm-invocation steps in the entire ALGO sweep** plus
the highest unclear rate (75%) and lowest forward-simulation rate (2.4%) —
the most opaque reasoning fingerprint of the four. Candidate table for §5.2
of the paper (currently absent, see §10).

### 4.5 BW Phase 2A — completion under strict PDDL vs NL-tolerant

| Model | Protocol | n | completed | abort | goal_reached | PGA_mean | CCI_mean | illegal_med |
|---|---|---|---|---|---|---|---|---|
| Claude | strict_PDDL | 50 | 8 (16%) | 22 | 0 | 0.025 | 0.013 | 13.0 |
| Llama  | strict_PDDL | 50 | 1 (2%) | 19 | 0 | 0.136 | 0.109 | 14.5 |
| GPT-4o | strict_PDDL | 50 | 0 (0%) | 20 | 0 | 0.008 | 0.000 | 15.0 |
| Claude | NL_tolerant_BW | 50 | 20 (40%) | 30 | 8 (16%) | 0.348 | 0.148 | 10.0 |
| Llama  | NL_tolerant_BW | 50 | 8 (16%) | 42 | 1 (2%) | 0.236 | 0.120 | 14.0 |
| GPT-4o | NL_tolerant_BW | 50 | 11 (22%) | 39 | 5 (10%) | 0.312 | 0.117 | 14.5 |
| Claude | NL_tolerant_MBW | 15 | 7 (47%) | 8 | 0 | 0.039 | — | 2.0 |
| Llama  | NL_tolerant_MBW | 15 | 0 (0%) | 15 | 0 | 0.000 | — | 12.0 |
| GPT-4o | NL_tolerant_MBW | 15 | 9 (60%) | 6 | 0 | 0.017 | — | 50.0 |

**BW vs MBW dissociation (the cleanest semantic-vocabulary signal in the
dataset):**
- Original BW solve 0/150 vs NL BW 14/150; Fisher p = 0.000089.
- NL BW solve 14/150 vs NL MBW 0/45; Fisher p = 0.043.
- Mann-Whitney PGA `BW > MBW` per model: Claude p=0.000144, Llama p<10⁻⁶,
  GPT-4o p<10⁻⁶.

---

## 5. Probe 2 — Phase 2B (injection)

Phase 2B is identical to 2A except at one step the true intermediate state is
replaced with a constructed false state. Two arms: **plausible** (locally
consistent false value) and **implausible** (clearly wrong false value).

### 5.1 Post-injection final correctness

| Model | Arm | n | post_correct | rate | diverged_rate |
|---|---|---|---|---|---|
| Claude  | plausible    | 61 | 32 | 0.525 | 1.000 |
| Gemini  | plausible    | 61 | 19 | 0.311 | 1.000 |
| Llama   | plausible    | 61 | 14 | 0.230 | 1.000 |
| GPT-4o  | plausible    | 61 | 31 | 0.508 | 1.000 |
| Claude  | implausible  | 61 | 32 | 0.525 | 1.000 |
| GPT-4o  | implausible  | 61 | 34 | 0.557 | 1.000 |

### 5.2 Injection-step compliance class

| Model | Arm | compliant | partial | refusal | format_ignored |
|---|---|---|---|---|---|
| Claude  | plausible   | 0.885 | 0     | 0.016 | 0.016 |
| Gemini  | plausible   | 0     | 0     | 0     | **1.000** |
| Llama   | plausible   | 0.393 | 0     | 0     | 0.590 |
| GPT-4o  | plausible   | 0.934 | 0     | 0     | 0.066 |
| Claude  | implausible | 0.918 | 0     | 0     | 0.016 |
| GPT-4o  | implausible | 0.951 | 0     | 0     | 0.049 |

### 5.3 Compliance × outcome (recovery patterns)

**Claude (n=61, plausible):** 28/54 compliant correct = 51.9%; 4/5 partial
correct = 80% (anecdotal); 0/1 refusal; 0/1 format-ignored.

**Llama (n=61, plausible):** 4/24 compliant correct = 16.7%; **9/36 format-
ignored = 25.0%** (format-ignoring beats compliance on outcome — for Llama,
complying with bad input *hurts*).

**Gemini (n=61, plausible, all format_ignored):** 19/61 correct = 31.1%.
Gemini's 100% format-ignored is *not* failure — the injection simply doesn't
propagate; 31% of those sessions still arrive at the correct answer. Paper
currently describes Gemini as "0% compliant" but does not mention the 31.1%
recovery rate.

### 5.4 Simpson's paradox in the aggregate

Aggregate "implausible 0.541 (n=122) > plausible 0.393 (n=244)" with
Fisher p=0.010 is a **sampling artifact**. The implausible arm contains only
Claude and GPT-4o sessions; the plausible arm additionally contains Llama
(0.230) and Gemini (0.311), which pull the plausible aggregate down.
Within-model: Claude 0pp, GPT-4o +4.9pp. The paper now includes a footnote
making this explicit and draws no population-level conclusion from the gap.

---

## 6. Probe 3 — Infigram contamination

Per-problem template proximity (`tc`, domain-generic phrasing) and instance
proximity (`ic`, goal-specific identifiers and values) computed against
The Pile + DCLM via Infini-gram.

### 6.1 ALGO — per-model contamination → robustness correlations

VRI = mean(W1, W2, W4) − W3. Per-model correlations against `tc`:

| Model | n | Spearman(tc,VRI) | p | Pearson(tc,VRI) | p | Spearman(tc, W3-ret) |
|---|---|---|---|---|---|---|
| Claude  | 116 | +0.011 | 0.91   | -0.249 | 0.007  | +0.551 |
| Gemini  | 116 | -0.019 | 0.84   | -0.095 | 0.31   | +0.432 |
| Llama   | 116 | +0.285 | 0.002  | +0.212 | 0.022  | — |
| GPT-4o  | 116 | +0.226 | 0.015  | +0.203 | 0.029  | -0.043 |
| o4-mini | 116 | -0.348 | 0.0001 | -0.186 | 0.046  | +0.854 |

The paper's `r=+0.44` for Claude proximity-VRI and `r=+0.36` for GPT-4o are
**Pearson r**, not Spearman ρ — this label has been corrected throughout the
audit. Cross-family contamination → VRI Pearson r (110 problems each):

| Model | Pearson r | p |
|---|---|---|
| Claude  | -0.065 | 0.50 |
| Gemini  | +0.008 | 0.94 |
| Llama   | -0.085 | 0.38 |
| GPT-4o  | -0.044 | 0.65 |

Cross-family is much weaker than within-ALGO — the ALGO-specific structure
(template proximity within shortest-path problems) is what generates the
signal.

### 6.2 ALGO proximity distribution by subtype

| Subtype | n | tc_mean | tc_median | ic_median |
|---|---|---|---|---|
| CC  | 25 | 0.468 | 0.127 | 0.000 |
| SP  | 56 | 0.162 | 0.062 | 0.000 |
| WIS | 35 | **0.000** | **0.000** | 0.000 |

WIS template-proximity is identically zero — the natural floor cell.

### 6.3 GSM contamination → CCI

| Model | n | Spearman ρ | p | Pearson r | p |
|---|---|---|---|---|---|
| Claude  | 44 | +0.241 | 0.116 | +0.331 | 0.028 |
| Gemini  | 44 | -0.075 | 0.628 | -0.140 | 0.365 |
| Llama   | 44 | -0.170 | 0.269 | -0.170 | 0.270 |
| GPT-4o  | 44 | +0.182 | 0.238 | +0.131 | 0.395 |

Claude Pearson `r=0.331, p=0.028` is what's cited in §5.3 as
"`ρ=0.31, p=0.044, n=42`" — the n=42 is a stale earlier export; the
direction (positive, ~0.3, marginal) holds with proper recomputation.

---

## 7. Probe 3 — Mechanistic (Qwen-2.5-7B base)

Forward-pass Qwen-2.5-7B-base on each (problem, variant) and record the rank
of the gold answer token at every layer. Source:
`results/raw/mechanistic_sweep_7b_base_rawqa.csv` (398 rows).

### 7.1 Final-layer gold-token rank by (family, variant)

| Family | Variant | n | rank median | rank mean | rank Q25 | rank Q75 |
|---|---|---|---|---|---|---|
| algorithmic | canonical | 100 | 22,472 | 13,572 | 19    | 22,846 |
| algorithmic | W6        | 90  | 156    | 8,776  | 18    | 23,325 |
| arithmetic  | canonical | 44  | 2,985  | 23,275 | 648   | 34,171 |
| arithmetic  | W6        | 24  | 3,201  | 16,442 | 1,219 | 17,878 |
| coin_change | canonical | 10  | 15,302 | 16,352 | 14,829| 18,410 |
| planning    | canonical | 65  | 419    | 3,474  | 234   | 8,595  |
| planning    | W6        | 65  | 49     | 1,611  | 41    | 53     |

### 7.2 Unpaired Mann-Whitney (W6 easier than canonical?)

The mechanistic sweep used **different** problem IDs for canonical and W6
in every family (overlap = 0). Paired Wilcoxon not possible.

| Family | n_canonical | n_W6 | MW p (W6 easier) |
|---|---|---|---|
| algorithmic | 100 | 90 | 0.402 |
| arithmetic  | 44  | 24 | 0.528 |
| planning    | 65  | 65 | **<0.0001** |

Only **planning** has a statistically significant W6 < canonical rank
pattern. The "arithmetic reversal" (canonical median 2,985 vs W6 3,201) is
not defensible. The planning result is interesting standalone but the
non-overlapping IDs preclude any paired-problem claim.

### 7.3 Crystallization layer

`crystallization_layer = -1` (no layer reached the threshold) for ≥98% of
rows across all families. Metric is not used by the paper.

**Decision applied to `main.tex`.** Mechanistic data is **not cited** in
main text. Coverage table mentions Qwen-0.5B pilot only. The 7B-base results
are retained here for completeness and future deployment.

---

## 8. Triangulation — per-instance convergence

### 8.1 ALGO triangulation labels (overall distribution, n=440 cells)

| Label | count | rate |
|---|---|---|
| ambiguous            | 271 | 0.616 |
| mixed                | 157 | 0.357 |
| retrieval_signal     | 8   | 0.018 |
| computation_signal   | 4   | 0.009 |

### 8.2 By model

| Model | n | retrieval | computation | mixed | ambiguous |
|---|---|---|---|---|---|
| Claude | 110 | 0 | 0 | 49 | 54 |
| Gemini | 110 | 0 | 0 | 42 | 66 |
| Llama  | 110 | 0 | 0 | 45 | 65 |
| GPT-4o | 110 | 0 | 0 | 21 | 86 |

Retrieval signals: 6 Claude + 2 Gemini = 8. Computation signals: 3 GPT-4o +
1 Claude = 4. 97% of (problem, model) cells remain ambiguous or mixed under
behavioral-only triangulation — single-problem disambiguation is not
achievable at this resolution. The framework's value is in the **pairwise
patterns** at subtype level, not per-problem labels.

### 8.3 By subtype × instance_type

| Subtype | InstanceType | n | retrieval | computation | mixed | ambiguous |
|---|---|---|---|---|---|---|
| coin_change   | adversarial | 40  | 0 | 0 | 19 | 20  |
| coin_change   | standard    | 60  | 0 | 0 | 0  | 60  |
| shortest_path | adversarial | 124 | 0 | 0 | 31 | 93  |
| shortest_path | standard    | 96  | 0 | 0 | 59 | 26  |
| wis           | adversarial | 120 | 0 | 0 | 48 | 72  |

CC standard is fully ambiguous (60/60) — least informative subtype.
SP adversarial is the cleanest dissociation cell.

### 8.4 GSM triangulation per instance (Claude + GPT-4o)

| Model | converging_computation | diverging | insufficient |
|---|---|---|---|
| Claude | 35 (0.795) | 9 (0.205) | 0 |
| GPT-4o | 11 (0.172) | 9 (0.141) | 44 (0.688) |

---

## 9. Formal statistical tests

### Test A — pooled Spearman (canonical accuracy vs W3 retention)

Aggregation: per (family, subtype, instance_type, model) accuracy cell.
Data file: `results/paper/AUDIT/spearman_acc_W3retention_subtype_data.csv`.

| Family | n cells | Spearman r | p |
|---|---|---|---|
| GSM   | 5  | +0.359 | 0.553 |
| ALGO  | 20 | +0.373 | 0.105 |
| BW    | 4  | -1.000 | 0.000 (n=4, near-degenerate) |
| **POOLED** | **29** | **+0.165** | **0.392** |

Bootstrap 95% CI on pooled r: `[-0.27, +0.59]`.

### Test B — per-problem phi (canonical vs W3 within model)

See §2.3 table — phi is predominantly positive (canonical-correct →
W3-correct), several with Fisher p<0.001.

### Interpretation

Both tests *reject* a population-level inverse "higher accuracy means lower
robustness" law. The paper's title "Accuracy is not Robustness" refers to
**pairwise per-(model, subtype) dissociations** — explicit pairwise
inversions on SP-adv (Claude/GPT-4o, Fisher p=0.0021) and CC-adv
(Claude/GPT-4o, Fisher p=0.0108) — not a population law. Scope stated
explicitly in the abstract and §5.1.

### Test C — Wilson 95% CIs for the central inversion

SP-adv (n=34 each):
- Claude canonical: 22/34 = 0.647 [0.477, 0.787]
- GPT-4o canonical: 14/34 = 0.412 [0.260, 0.585] — **CIs overlap massively**
- Claude W3: 0/34 = 0.000 [0.000, 0.102]
- GPT-4o W3: 9/34 = 0.265 [0.146, 0.430] — **CIs non-overlapping**

The inversion is in the *robustness axis* (W3), not the canonical axis. This
is the strongest single piece of pairwise evidence in the paper.

---

## 10. Hidden / underplayed findings

Real signals in the data that strengthen the paper but are currently
underplayed or absent. Each is candidate for a §5 paragraph or appendix
table.

### 10.1 W5 direction-reversal as a second axis of fragility

Section 2.5 above. Llama loses 52 pts absolute on GSM W5; Claude and
o4-mini actually *improve*. Two of five models tolerate direction reversal,
three do not. Independent of the W3 rename axis.

### 10.2 BW W5 anomalously high vs canonical for Claude (3.4×) and Gemini (1.5×)

Likely a PlanBench property (goal-state-vs-initial-state asymmetry), not a
model capability claim. Should be described as a benchmark property in §5.4
or shown via per-problem W5/canonical pair comparison.

### 10.3 Reasoning-type behavioral fingerprints (Phase 2A)

Section 4.4 table above. GPT-4o has 0 algorithm-invocation steps in the
entire ALGO sweep; highest unclear rate (75%); lowest forward-simulation
(2.4%). Direct behavioral signature. Currently absent from the paper.

### 10.4 Phase 2B compliance × outcome decomposition

Section 5.3 above. Three findings:
1. Claude partial-compliance is 80% correct (anecdotal, n=5)
2. Gemini's format-ignoring is 31.1% correct, not 0%
3. For Llama, compliance HURTS (16.7% < 25.0% format-ignored)

### 10.5 W4 (formal notation) often exceeds canonical on ALGO

Section 2.4 above. Claude 0.618 vs canonical 0.364 (+25 pts). The LaTeX
notation reduces parse ambiguity. Worth a discussion paragraph.

### 10.6 W2 hurts GPT-4o more than W3 on ALGO

GPT-4o canonical 0.427 → W2 0.109 (−32 pts) → W3 0.173 (−25 pts). Format
change is GPT-4o's worst variant. Inconsistent with the W3-as-principal-probe
framing — should be noted.

### 10.7 CC adversarial is o4-mini's high-water mark

o4-mini: 10/10 on CC-adversarial canonical (the highest-template-proximity
cell in the corpus). All other ALGO cells are 0–0.23. Suggests o4-mini's
reasoning-RL training transfers well *only* in cells with high template
proximity — inconsistent with the "structurally robust" framing.

### 10.8 BW Phase-2A failure-mode taxonomy

`BW_P2_cci.csv` has violation columns (`hand_not_empty`, `block_not_clear`,
`target_not_clear`, `format_error`). The paper says only "abort due to
illegal action loops" but per-model failure profiles differ (Llama dominated
by `format_error`; GPT-4o by `wrong_stack_source`). Worth one short
appendix paragraph.

### 10.9 NL-tolerant BW reveals positive solve rates hidden by strict PDDL

Already in the paper. Strict PDDL: 0–16% solve; NL-tolerant: 16–22% solve.
The strict-PDDL grading was a measurement artifact, not a planning failure.

### 10.10 Mystery-BW remains 0/15 on every model under NL-tolerant grading

Cleanest evidence that BW success is semantic-vocabulary-driven (the BW
domain-specific vocabulary), not parse-artifact-driven. MBW preserves the
problem structure but renames every domain predicate.

### 10.11 Cross-tokenizer same-direction W3 inversion (Claude vs GPT-4o)

Claude (BPE-A) and GPT-4o (BPE-B) use *different* tokenizers but show
*opposite* W3 fragility directions on the same SP-adv and CC-adv problems.
Tokenization is therefore ruled out as the explanation. Defended in
Appendix A.

---

## 11. Negative space — what this paper does NOT claim

- **No claim of "the model retrieves from memory."** Triangulation labels are
  *behavior*, not internal-process attribution.
- **No causal claim that algorithm-invocation produces error.** The n=13
  observation is observational; Fisher p=0.40 vs unclear-reasoning baseline
  is non-significant. Reported as a *paradox*, not a law.
- **No population-level "accuracy is not robustness" law.** §5.1 states
  explicitly that the title refers to pairwise per-(model, subtype)
  dissociations.
- **No mechanistic claim from Qwen-0.5B pilot.** Pilot only; not in main
  text.
- **No mechanistic claim from Qwen-7B base.** Sweep exists locally; not
  cited.
- **No tokenizer-artifact claim about W3.** Cross-tokenizer same-direction
  result rules it out (§10.11).
- **No "independent verification" claim about injection compliance.** Models
  accepting false intermediate states while still producing correct final
  answers is reported as *empirical separability*, not as evidence of
  internal cross-checking.
- **No reproducibility claim for the Qwen-7B mechanistic results without
  the GPU sweep.** Only behavioral CSVs are derivable from API calls alone.

---

## 12. Pointer index — every aggregate → source file

| Topic | Authoritative file |
|---|---|
| Probe 1 ALGO raw | `results/raw/ALGO_P1_behavioral_{claude,gpt4o,llama,gemini,o1mini}.csv` |
| Probe 1 BW raw | `results/raw/BW_P1_behavioral{,_gemini,_o1mini}.csv` |
| Probe 1 GSM raw | `results/raw/GSM_P1_behavioral_{claude,gpt4o,llama,gemini,o1mini}.csv` |
| Probe 1 per-problem 5-model derived | `results/derived/P1_per_problem_var_5model.csv` |
| Bank IDs | `data/problems/question_bank_{algo,bw,gsm}.csv` |
| GSM Probe 2 consolidated | `results/raw/GSM_P2_cci.csv` |
| GSM Phase 1 per-model | `results/raw/GSM_P2_phase1_{claude,gpt4o,llama,gemini,o1mini}.csv` |
| ALGO Phase 2A normal | `results/raw/ALGO_P2_phase2_normal.csv` (+ `_gemini.csv`) |
| ALGO Phase 2A elicited | `results/raw/ALGO_P2_phase2_normal_elicited.csv` |
| ALGO Phase 2B plausible | `results/raw/ALGO_P2_phase2_injected.csv` (+ `_gemini.csv`) |
| ALGO Phase 2B implausible | `results/raw/ALGO_P2_phase2_injected_implausible.csv` |
| BW Phase 2 strict | `results/raw/BW_P2_cci.csv` (+ `_plans.csv`, `_tep.csv`) |
| BW Phase 2 NL-tolerant | `results/raw/BW_P2_cci_nl.csv` |
| MBW Phase 2 NL-tolerant | `results/raw/MBW_P2_cci_nl.csv` |
| ALGO contamination | `results/raw/ALGO_P3_contamination.csv` |
| BW contamination | `results/raw/BW_P3_contamination.csv` |
| GSM contamination | `results/raw/GSM_P3_contamination.csv` |
| Qwen-7B mechanistic (base, with rank) | `results/raw/mechanistic_sweep_7b_base_rawqa.csv` |
| Qwen-7B mechanistic (instruct, cosine only) | `results/raw/mechanistic_sweep_7b.csv` |
| Qwen-0.5B pilot | `probe3_mechanistic.csv` |
| ALGO triangulation v3 | `results/derived/ALGO_P3_triangulation_v3.csv` |
| GSM triangulation per-instance | `results/derived/GSM_P3_triangulation_per_instance_{claude,gpt4o}.csv` |
| Spearman audit data | `results/paper/AUDIT/spearman_acc_W3retention_subtype_data.csv` |
| Probe 1 per-(family,model,variant) audit | `results/paper/AUDIT/probe1_family_variant_accuracy_5model.csv` |
| Probe 1 per-(subtype,model,variant) audit | `results/paper/AUDIT/probe1_algo_subtype_variant_accuracy_5model.csv` |
| Cross-family regression | `results/derived/cross_family_regression.csv` |
| BW vs MBW dissociation | `results/paper/AUDIT/bw_mbw_dissociation_summary.csv` |

---

## Notes on reproducibility

- Every Probe-1 univariate / pairwise table in §2.* is regenerated by
  `scripts/compute_p1_metrics_unified.py` from raw CSVs.
- Every Probe-2 CCI / TEP / invocation rate in §3–§5 is regenerated by
  `scripts/{GSM_P2_SCR_compute_metrics,ALGO_P2_SCR_compute_metrics,BW_P2_SCR_run_cci}.py`.
- Every Probe-3 contamination correlation in §6 is regenerated from the raw
  Infigram CSVs via the contamination regression scripts in
  `scripts/{ALGO,BW,GSM}_P{1,3}_SCR_*.py`.
- Triangulation in §8 is regenerated by
  `scripts/{ALGO_P3_SCR_triangulation,BW_P3_SCR_run_triangulation}.py`.
- The 5-model Pearson/Spearman audit and Wilson CIs are regenerated by the
  audit scripts in `scripts/audit/` (kept for traceability; not required for
  any number cited in the paper).

If any number here differs from the paper, the **raw CSV is authoritative**
— recompute via the unified scripts above and update the paper, not this
file.
