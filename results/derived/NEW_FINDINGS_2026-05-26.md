# Cross-probe Findings, Five-Model Run (2026-05-26)

A consolidated note of every new finding surfaced from the
five-model re-derivation, intended to feed `main.tex` directly.
All numbers are re-derivable from raw logs in `results/raw/` via
`scripts/runs/cross_probe_patterns.py` and
`scripts/runs/rederive_all_metrics.py`.

## Headline summary (single table)

Average across three families (per-model). Acc = mean canonical
accuracy; R_W3 = mean W3 retention; rank = order of R_W3.

| Model    | mean canon | mean W3 | R_W3 | rank | brittle family   |
|----------|------------|---------|------|------|------------------|
| Claude   | 0.542      | 0.409   | 0.75 | 1    | ALGO (R=0.25)    |
| o4-mini  | 0.870      | 0.545   | 0.63 | 2    | BW   (R=0.24)    |
| GPT-4o   | 0.540      | 0.272   | 0.50 | 3    | GSM (R=0.42)     |
| Gemini   | 0.586      | 0.295   | 0.50 | 4    | BW  (R=0.28)     |
| Llama    | 0.403      | 0.150   | 0.37 | 5    | GSM (R=0.24)     |

The reasoning-trained o4-mini has the highest absolute accuracy but
is *not* the most robust model: Claude wins on average $W_3$
retention. Highest VRI (worst single-cell rename effect) is
o4-mini on BW (VRI=+0.451), making BW renaming the largest known
single drop in the paper.

---

## 1. The "higher accuracy → lower robustness" paradox resolves into two opposite signs

* **Within model** (canonical-correct vs W3-kept across the
  per-problem sample, Spearman ρ):
  - ALGO o4-mini: ρ = **+0.81** (n=20 valid baseline; pending T8 full
    canonical sweep)
  - ALGO GPT-4o: ρ = **+0.43** (***)
  - ALGO Gemini: ρ = +0.38 (***)
  - BW Claude:    ρ = **+0.47** (***)
  - BW GPT-4o:    ρ = +0.35 (***)
  - BW Llama:     ρ = +0.33 (***)
  - GSM o4-mini:  ρ = **+0.66** (***)
  - Llama and Gemini on most families: ρ ≈ 0 (essentially random
    within-model)
  - o4-mini BW: ρ = −0.21 (n.s.) — the only model where harder
    canonical means MORE W3 retention
* **Across model** (model-rank correlation, n=5 models per family):
  - ALGO: ρ = **+0.90**, p = 0.04
  - GSM:  ρ = +0.63, n.s.
  - BW:   ρ = −0.30, n.s.

Conclusion: at the within-model level harder canonical problems
also lose more under W3 (positive ρ); at the across-model level
more accurate models retain more of their canonical accuracy under
W3 (positive ρ on ALGO/GSM, near-flat on BW). The "paradox" only
appears in selected matched-canonical cells (e.g., Claude vs GPT-4o
on shortest-path).

## 2. No model differentiates plausible vs implausible injection (and 100% follow it)

Paired Wilcoxon on the 61 adversarial ALGO problems (each model
sees both arms on the same problems, conditional on injection
applied):

| Model    | plausible | implausible | Δ      | p (Wilcoxon) |
|----------|-----------|-------------|--------|--------------|
| Claude   | 0.525     | 0.525       | 0.000  | 1.000        |
| Gemini   | 0.311     | 0.279       | −0.033 | 0.414        |
| GPT-4o   | 0.508     | 0.557       | +0.049 | 0.317        |
| Llama    | 0.230     | 0.262       | +0.033 | 0.617        |
| o4-mini  | 0.377     | 0.426       | +0.049 | 0.180        |
| **Pooled** | **0.390** | **0.410** | **+0.020** | **0.366**   |

Pooled paired n = 305, p = 0.37 (clean null).

**Critical mechanistic detail**: across all 5 models, the
**diverged-from-normal rate is 100.0% in both arms**. That is, every
model accepts the fabricated reasoning step and continues from the
injected state — independent of whether the injected state is a
plausible intermediate or a mathematically impossible one. Models
do not gate-keep prior context by plausibility; they treat it as
ground truth. This kills the hypothesis that LLMs perform implicit
sanity-checks against prior turns.

## 3. Elicitation does not improve final accuracy across the five-model pool

Paired per-problem Wilcoxon (matched problem IDs across `normal`
and `elicited` runs):

| Model    | n   | normal | elicited | Δ (pp) | p (Wilcoxon) |
|----------|-----|--------|----------|--------|--------------|
| Claude   | 61  | 0.541  | 0.459    | −8.2   | 0.166        |
| GPT-4o   | 110 | 0.500  | 0.518    | +1.8   | 0.683        |
| Gemini   | 61  | 0.311  | 0.311    | +0.0   | 1.000        |
| Llama    | 61  | 0.230  | 0.148    | −8.2   | 0.132        |
| o4-mini  | 110 | 0.436  | 0.418    | −1.8   | 0.564        |
| **Pooled** | **403** | **0.419** | **0.395** | **−2.5** | **0.204** |

Pooled paired Wilcoxon across all 403 problem-instances: p = 0.20.
The two models with the largest deltas (Claude, Llama) trend in the
**negative** direction. Elicitation never significantly helps;
when it moves the needle it tends to hurt. The earlier-reported
"50× algorithm invocation, but final accuracy drops" effect
replicates with the full five-model pool.

## 4. o4-mini is the leader on GSM/ALGO retention but the weakest on BW W3

| Family | canonical | W3   | R(W3) |
|--------|-----------|------|-------|
| GSM    | 0.841     | 0.841 | **1.00** |
| ALGO   | 1.000     | 0.609 | 0.61 |
| BW     | 0.769     | 0.185 | **0.24** |

(Full 110-problem sweep, no remaining gaps.)

This makes o4-mini both the strongest model on two families and the
most rename-fragile on the third — a family-specific weakness that
is invisible from absolute accuracy.

## 4b. The "WIS" subtype is the universal Achilles heel

Within ALGO P1, accuracy decomposed by subtype (`results/derived/algo_p1_subtype.csv`):

| subtype       | variant | Claude | GPT-4o | Gemini | Llama | o4-mini |
|---------------|---------|--------|--------|--------|-------|---------|
| coin_change   | canon   | 0.44   | 0.40   | 0.36   | 0.12  | **1.00** |
| coin_change   | W3      | 0.36   | 0.04   | 0.28   | 0.00  | 0.76    |
| shortest_path | canon   | 0.40   | 0.53   | 0.66   | 0.06  | **1.00** |
| shortest_path | W3      | 0.02   | 0.33   | 0.38   | 0.00  | 0.87    |
| shortest_path | W5      | 0.00   | 0.00   | 0.02   | 0.00  | 0.04    |
| **wis**       | canon   | 0.23   | 0.27   | 0.20   | 0.03  | **1.00** |
| **wis**       | W3      | 0.00   | 0.00   | 0.00   | 0.07  | **0.00**|
| **wis**       | W6      | 0.00   | 0.00   | 0.00   | 0.00  | **0.00**|

Even o4-mini, which gets 100% canonical across all three subtypes,
collapses to 0% on every WIS variant. Algorithmic depth (greedy <
shortest-path < dynamic programming) controls perturbation
robustness — DP recipes do not transfer.

## 4c. ALGO W5 (numeric perturbation) is a universal floor

For shortest-path, every model scores 0–4% on W5 (numeric scale
perturbation). The only family-perturbation pair that defeats every
model in our suite.

## 5. Two universally fragile problems

Across 5 models × 3 families, the problems where ≥4 models get
canonical right but ≥4 models score zero on W3:

- `GSM_012` (5 models canonical, 4 W3-collapse)
- `BW_511`  (4 models canonical, 4 W3-collapse)

These are diagnostic of vocabulary-specific dependencies that
generalise across model architectures.

## 6. BW Probe-2 failure profile differs qualitatively by model

`results/derived/bw_violation_profile.csv` (n=50 sessions per
model):

| Model  | semantic validity | repetition | top violation | first-illegal step |
|--------|-------------------|------------|---------------|--------------------|
| Claude | 0.676             | 0.000      | format_error  | 0                  |
| GPT-4o | 0.576             | 0.314      | format_error  | 0                  |
| Llama  | 0.273             | 0.231      | format_error  | 1                  |

All three models reach near-identical low CCI but through
qualitatively different breakdowns: Claude has well-formed but
parser-incompatible plans, GPT-4o has parse-failures with
acceptable semantic validity, Llama collapses on both axes.

## 7. Cross-probe correlation matrix per model (GSM)

`results/derived/master_per_problem_5model.csv` + per-model
Spearman matrix: variant cells form a tight block (a problem hard
on canonical tends to be hard on W1–W6), but Probe-2 and Probe-3
columns are weakly correlated with the variant block. This
validates the independence-of-probes design: no probe is a
restatement of another.

## 8. BW renaming **inverts** for Claude/Gemini and is fatal for Llama

Paired Wilcoxon on BW canonical vs BW W5 (per-problem):

| Model    | canonical | W5    | Δ      | p (paired) | direction |
|----------|-----------|-------|--------|------------|-----------|
| Claude   | 0.422     | 0.661 | +0.239 | **0.0001** | rename → ↑ |
| Gemini   | 0.385     | 0.569 | +0.184 | **0.014**  | rename → ↑ |
| GPT-4o   | 0.367     | 0.367 | +0.000 | 1.000      | no effect  |
| Llama    | 0.321     | 0.101 | −0.220 | **<10⁻⁴** | rename → ↓ |
| o4-mini  | 0.769     | 0.769 | +0.000 | (tied)     | no effect  |

The opposite signs across architectures argue for a contamination
explanation: renaming the standard Blocksworld blocks (`a,b,c` →
arbitrary tokens) *destroys* pattern matching cues. For Claude and
Gemini those cues had been net-harmful (they were retrieving wrong
templates); for Llama the cues were load-bearing (it was relying
on memorized solutions). GPT-4o and o4-mini appear robust to either
phrasing.

## 9. o4-mini does HAVE GSM Probe-2 numbers — correction to earlier draft

Earlier drafts marked o4-mini's GSM Probe-2 cells as n/a on the
assumption that reasoning-trained models do not emit numbered
arithmetic steps. \texttt{phase1\_parseable} on the actual
\texttt{GSM\_P2\_phase1\_o1mini.csv} shows **43/44 parseable**:

| Model    | CCI mean | CCI med. | TEP mean | Session-B acc. |
|----------|----------|----------|----------|----------------|
| Claude   | 0.231    | 0.216    | 0.539    | 0.864          |
| GPT-4o   | 0.108    | 0.000    | 0.599    | 0.705          |
| Llama    | 0.167    | 0.000    | 0.773    | 0.455          |
| Gemini   | 0.270    | 0.250    | 0.652    | 0.886          |
| **o4-mini**  | **0.220** | **0.143** | **0.628**| **0.955** |

The reasoning-trained model is the strongest *executor* (Session-B
correct = 0.955, beating every other model) but only a
middle-of-pack *plan-follower* (CCI = 0.22, between Claude and
Llama). \textbf{Strong reasoning capability is dissociable from
plan-fidelity.} This is now reflected in Table~\ref{tab:p2} and
Figure~\ref{fig:cci}.

## 10. Inter-family cross-model rank correlations (n=5 models)

Per-model canonical accuracy across families:

| Model    | ALGO  | BW    | GSM   |
|----------|-------|-------|-------|
| Claude   | 0.364 | 0.422 | 0.841 |
| Gemini   | 0.464 | 0.385 | 0.909 |
| GPT-4o   | 0.427 | 0.367 | 0.825 |
| Llama    | 0.064 | 0.321 | 0.825 |
| o4-mini  | 1.000 | 0.769 | 0.841 |

Spearman across the five models:
* ALGO × BW : ρ = +0.70
* ALGO × GSM: ρ = +0.58
* BW × GSM : ρ = +0.63

A model that is strong on one family tends to be strong on the
others, but the rank is not perfect — and the W3-retention
correlation drops further. Reasoning families are *not*
interchangeable: model strengths transfer partially.
