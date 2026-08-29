# Evaluation walkthrough — question → metric (with examples)

One worked example per family: **GSM** (arithmetic), **ALGO** (coin change), **BW** (blocksworld).

For file paths and W1–W6 audit tables see `PROBE_PIPELINE_REFERENCE.md`.

---

## 0. Big picture

```
QUESTION BANK (canonical + W1–W6 variants)
        │
        ├─► PROBE 1 — Behavioral: one-shot answer, verified correct/incorrect
        │         metrics: VAR, CSS, W3 retention, VRI, RCS
        │
        ├─► PROBE 2 — Process: multi-turn plan vs execution
        │     ├─ GSM: Session A plan + Session B stepwise + TEP injection
        │     ├─ ALGO: Phase 1 declaration + Phase 2 normal/elicited/injected
        │     └─ BW: extract plan → interactive CCI → TEP corruption
        │         metrics: CCI, TEP, injection compliance, reasoning_type
        │
        └─► PROBE 3 — Contamination + triangulation
                  ├─ Infini-gram proximity score (canonical text)
                  └─ Merge P1 + P2 + P3 → per-instance label
                      metrics: contamination_score, converging_retrieval/computation
```

**Shared bank columns:** `problem_id`, `variant_type`, `problem_text`, `correct_answer`, `problem_subtype`, `contamination_pole`, `difficulty_params` (ALGO).

---

## 1. Question making (W1–W6)

Variants are **authored or generated** into the bank before any model is called. Same semantic problem, different surface form.

| Code | What changes | Example (GSM_001) |
|------|--------------|-------------------|
| canonical | Base | Hotel phone: $0.9/min then $0.5 after 20 min → **86 min costs $51** |
| W1 | Lexical paraphrase | “A hotel room phone charges…” → answer still **51** |
| W2 | Structural reformat | Markdown/table layout → answer **51** |
| W3 | Entity rename | “Hiking trail… calories/min” → answer still **51** (semantics preserved) |
| W4 | Formal notation | Piecewise function C(t) → answer **51** |
| W5 | Reversal | “Call cost $51… how many minutes?” → answer **86** (answer changes) |
| W6 | New instance | Procedurally regenerated from same template → new numbers |

### GSM example — `GSM_001`

- **Source:** `apple/ml-gsm-symbolic` template_id=10 (`question_bank_gsm.csv`)
- **Canonical answer:** 51 (20×0.9 + 66×0.5)
- **W3 rename:** “hiker / calories” instead of “hotel / dollars” — tests whether the model reasons over structure or memorizes surface words
- **W6:** `scripts/GSM_PX_SCR_generate_w6.py` pulls a new instance from JSONL with same template

### ALGO example — `CC_01` (coin change, adversarial)

- **Subtype:** coin_change, **instance_type:** adversarial (greedy fails)
- **Canonical:** denominations [1,3,4,7], target 6 → optimal **Count: 2, Coins: [3,3]**
- **W3 rename:** “chef / spice scoops” instead of “coins” — same math
- **`difficulty_params` JSON** (required for P2):
  - `greedy_succeeds: false`
  - `greedy_answer: "Count: 2\n Coins: [3, 3]"` (wrong greedy path)
  - `critical_step_index: 0` (where Phase 2 injects a wrong state)
- **Bank prep scripts:** fix → greedy metadata → critical step → audit → W6 generator

### BW example — `BW_001` (blocksworld)

- **Source:** PlanBench PDDL (`source` has `path=...instance-1.pddl`)
- **Canonical:** 8 blocks on table; goal is a tower — optimal plan is 14 `pick-up`/`stack` actions
- **W3:** block letters renamed (e.g. action aliases) — plan structure unchanged
- **W6:** `scripts/maintenance/generate_w6_variants.py` — new init/goal, Fast Downward planner writes new `correct_answer`

---

## 2. Probe 1 — Behavioral invariance

**Question:** Is the final answer still correct under perturbation?

### 2.1 Run flow (all families)

1. Load bank row → send `problem_text` to model (zero-shot, T=0)
2. Parse model output
3. **Verify** against `correct_answer`
4. Store row in `results/raw/*_P1_behavioral_*.csv`

| Family | Script | Verifier |
|--------|--------|----------|
| GSM | `BW_P1_SCR_run_behavioral_sweep.py --family arithmetic_reasoning` | `verify_gsm_answer()` — `#### num` or last number ±0.01 |
| ALGO | `ALGO_P1_SCR_run_behavioral_sweep.py` | `verify_algo()` — strict Count/Coins parse, DP check |
| BW | `BW_P1_SCR_run_behavioral_sweep.py --family blocksworld` | `verify_answer()` — simulate actions → goal state |

### 2.2 Worked example — GSM_001, Claude, W3

```
Prompt (W3):  "hiker burns 0.9 calories/min … 86-minute hike …"
Model output: "… Final answer: 51"   (or #### 51)
Verifier:     extract 51, compare to ground_truth 51 → behavioral_correct = True
```

If canonical was True and W3 is False → this problem contributes to **W3 drop**.

### 2.3 Worked example — CC_01, GPT-4o, canonical

```
Model must output:
  Count: 2
  Coins: [3, 3]
Verifier: parse integers, compare multiset to optimal DP solution
If model outputs greedy [4,1,1] → verified = False, gave_greedy_answer may = True
```

### 2.4 Worked example — BW_001, canonical

```
Model output: numbered list of actions
Verifier: execute each action on parsed init/goal from PDDL
          if state reaches goal → behavioral_correct = True
          else fallback: normalized action sequence == ground_truth plan
```

### 2.5 Probe 1 metrics (how numbers are computed)

**Per variant, per model — VAR(variant):**
```
VAR(W3) = (# problems where behavioral_correct=True on W3) / (total W3 attempts)
```
From `probe1_per_model_variant.csv`. Example: ALGO Claude W3 = 0.091 (9.1%).

**W3 retention (model-level):**
```
R_W3 = VAR(W3) / VAR(canonical)
```
Claude ALGO: 0.091 / 0.364 ≈ 0.25 (large rename drop).

**CSS (per problem, then averaged):**
```
CSS(problem) = (# of {W1,W2,W3,W4,W6} variants correct) / (5)
```
W5 excluded (different answer). If canonical correct and W1,W2,W4,W6 all match → CSS=1.0.

**VRI (vocabulary vs structural robustness):**
```
VRI_structural = mean(VAR(W2), VAR(W4))
VRI_vocabulary = VAR(W3)
VRI_gap = VRI_structural − VRI_vocabulary
```
Large gap → entity rename hurts more than reformatting.

**RCS (W5 only):**
```
RCS = verify(W5 answer, W5 ground_truth)   # reversal changes correct answer
```

**Output files:** `probe1_per_model_variant.csv`, `P1_per_problem_var_5model.csv`, family `*_P1_metrics.csv`

---

## 3. Probe 2 — Plan / execution coupling

**Question:** Does the model’s *process* match what it declares, and does it recover after corruption?

---

### 3.1 GSM Probe 2 — example `GSM_001`

**Script:** `GSM_P2_SCR_run_probe2.py`  
**Raw:** `GSM_P2_phase1_<model>.csv`, `GSM_P2_cci.csv`

#### Step A — Phase 1 (Session A): full plan, no final answer shortcut

```
Prompt: "Solve … write every step: Step N: [desc] = [value] … Final answer: [num]"
Model might output:
  Step 1: First 20 minutes at 0.9 = 18
  Step 2: Remaining 66 minutes at 0.5 = 33
  Final answer: 51
```

Parsed into `phase1_steps_json`. `phase1_parseable = True` if steps + final present.

#### Step B — Phase 2 (Session B): one step per fresh API call

```
Turn 1 prompt: "Problem: … This is step 1. What is the first computation step?"
Turn 2 prompt: "Steps completed so far: Step 1: … = 18. Current value: 18. What is step 2?"
… continues until model stops or cap
```

**Important:** each step is a **new session** — model cannot see its own prior raw text, only the numeric chain you inject.

#### CCI (plan–execution consistency)

For each step index i:
- Compare Phase 1 step value vs Phase 2 step value
- Match if numeric equal OR cosine similarity ≥ 0.82 on step text

```
CCI = (matched steps) / (steps compared)
```

Example: Phase 1 has 2 steps, Phase 2 matches both numerically → CCI = 1.0.  
If Phase 2 step 2 drifts → CCI = 0.5.

#### TEP (trajectory error propagation)

1. Pick `inject_at_step` (from `difficulty_params` or middle of plan)
2. Replace true intermediate value with `injected = true × 1.15`
3. Continue Phase 2 from corrupted state
4. **TEP** = fraction of post-corruption steps that match the *adapted* expected path

```
Example GSM_001:
  True after step 1: 18
  Injected: 20.7
  Model continues from 20.7 …
  TEP measures whether it recovers toward final answer 51
```

Also record `session_b_correct` — final numeric answer still right despite mid-chain error?

**Derived:** `probe2_gsm_metrics.csv` — mean CCI, mean TEP per model.

---

### 3.2 ALGO Probe 2 — example `CC_01`

**Scripts:** `ALGO_P2_SCR_run_phase1.py`, `ALGO_P2_SCR_run_phase2.py`  
**Raw:** `ALGO_P2_phase1_*.csv`, `ALGO_P2_phase2_normal.csv`, `ALGO_P2_phase2_injected.csv`

#### Phase 2A — strategy declaration (4 questions)

```
Q1: What algorithm will you use?
Q2: Will a greedy approach work? (yes/no + why)
Q3: What is your first decision?
Q4: Where is the critical point?
```

Parsed fields: `stated_algorithm`, `greedy_assessment_correct`, `predicted_first_decision`, `critical_point_identified`.

For CC_01 adversarial:
- Correct greedy assessment = **No** (greedy fails)
- `critical_step_index = 0` in bank → injection at first decision step

#### Phase 2B-normal — stepwise execution

Each turn prompt includes current DP state; model outputs:
```
Decision: [choice]
Reason: [text]
```

`classify_reasoning_type()` → greedy / forward-simulation / algorithm-invocation / backtracking

Final step verified with `verify_algo()` → `final_answer_correct`

#### Phase 2B-injected — corruption at critical step

At `critical_step_index`, replace true state with `injected_state` from bank metadata.

Track:
- `response_type`: compliant / refusal / format_ignored
- `diverged_from_normal`: did model accept wrong state?
- `post_injection_correct`: final answer still optimal?

**ALGO CCI components** (`ALGO_P2_SCR_compute_metrics.py`):
- `CCI_algorithm` — declared algorithm vs step reasoning labels
- `CCI_first_decision` — Q3 vs step 0 decision
- `CCI_critical` — Q4 vs injection step behavior
- `TEP_refined` — recovery after injection

**Sub-tracks:**
| Track | File | Purpose |
|-------|------|---------|
| Normal | `ALGO_P2_phase2_normal.csv` | Baseline stepwise (110 problems) |
| Elicited | `ALGO_P2_phase2_normal_elicited.csv` | Algorithm name prompted first (61/110 for 3 models) |
| Injected plausible | `ALGO_P2_phase2_injected.csv` | Wrong but plausible intermediate |
| Injected implausible | `ALGO_P2_phase2_injected_implausible.csv` | Mathematically impossible state |

P2B uses **61 adversarial** problems only (where injection metadata exists).

---

### 3.3 BW Probe 2 — example `BW_001`

**Scripts:** extract plans → `BW_P2_SCR_run_cci.py` → `BW_P2_SCR_run_tep.py`

#### Step 1 — Extract plan from Probe 1

`BW_P2_SCR_extract_phase1_plans.py` parses Probe 1 `raw_response` into action list, joins PDDL path from `source`.

#### Step 2 — Interactive CCI (multi-turn)

```
Turn 1: "Current state: … Goal: … What is your first action?"
Model: "pick-up h"
Verifier: execute on state machine — legal? goal reached?
Turn 2: new prompt with updated state …
```

**CCI** = matched executed actions / compared steps vs declared plan from Phase 1.

**Failure modes logged:** `hand_not_empty`, `format_error`, `aborted: excessive illegal steps`

Strict PDDL protocol aborts ~84–100% of sessions → NL-tolerant rerun in `BW_P2_cci_nl.csv`.

#### Step 3 — TEP

Mid-trajectory, `seeded_inject_error()` corrupts block configuration.  
**TEP** = adaptation score on post-corruption moves (same formula as GSM/ALGO — fraction matching expected recovery plan).

⚠️ Current BW TEP mostly `insufficient_data` due to aborts — scope claims accordingly.

---

## 4. Probe 3 — Contamination + triangulation

**Question:** Is behavior consistent with training-corpus proximity, and do all probes agree?

### 4.1 Contamination scoring (canonical text only)

**Script:** `BW_P3_SCR_run_contamination_triage.py`  
**Module:** `probes/contamination/score.py` + Infini-gram API

```
For problem_text tokens:
  binary-search longest n-gram (length 5–13, stride 3) found in The Pile/DCLM
  contamination_score = normalized match strength
```

GSM uses max n-gram **8** (shorter templates). ALGO can decompose into `template_contamination_score` vs `instance_contamination_score`.

**Example CC_01:** coin-change template may have moderate overlap; WIS instances typically score lower (less common phrasing in corpus).

**Raw:** `GSM_P3_contamination.csv`, `ALGO_P3_contamination.csv`, `BW_P3_contamination.csv`

### 4.2 Triangulation — merge signals per instance

**Script:** `ALGO_P3_SCR_triangulation.py` / `BW_P3_SCR_run_triangulation.py`  
**Logic:** `probes/triangulation/per_instance.py`

For each `(problem_id, model)` join:

| Signal | Rule |
|--------|------|
| VAR | canonical correct → computation; else retrieval |
| CSS | ≥ 0.5 → computation |
| Contamination | > 0.6 → retrieval; ≤ 0.4 → computation; else ambiguous |
| CCI | ≥ 0.4 → computation |

**Agreement:**
- ≥2 agreeing non-ambiguous signals → `converging_retrieval` or `converging_computation`
- Mixed signals → `diverging` or `ambiguous`

**Example (hypothetical CC_01, Claude):**
```
canonical correct = False  → VAR signal: retrieval
CSS = 0.2                    → retrieval
contamination = 0.72         → retrieval
CCI = 0.55                   → computation
→ diverging (retrieval vs computation on CCI)
```

**Output:** `ALGO_P3_triangulation_v3.csv` (440 rows × 4 models)

Exploratory threshold sweep: `triangulation_v2_labels.csv` (1944 configs — design choice, not locked).

---

## 5. End-to-end: one problem through all probes

### GSM_001 + Claude (illustrative)

| Probe | What we learn |
|-------|---------------|
| P1 canonical | Correct? → baseline accuracy |
| P1 W3 | Still 51 after rename? → rename fragility |
| P1 CSS | 4/5 variants correct? → surface consistency |
| P2 CCI | Do step values match declared plan? |
| P2 TEP | After corrupting step 1 value, still get 51? |
| P3 contam | How common is this template in corpus? |
| P3 triangulate | Do proximity + CSS + CCI agree on “how” model solves it? |

### CC_01 + GPT-4o (illustrative)

| Probe | What we learn |
|-------|---------------|
| P1 canonical | Optimal vs greedy answer |
| P1 W3 | Does spice rename break coin DP? |
| P2 Phase1 | Says “dynamic programming” but greedy assessment correct? |
| P2 injected | Accepts wrong coin choice at step 0? recovers? |
| P3 | Template exposure vs WIS low-exposure contrast |

### BW_001 + Llama (illustrative)

| Probe | What we learn |
|-------|---------------|
| P1 canonical | Full plan correct in one shot? |
| P1 W3 | Rename blocks — plan still valid? |
| P2 CCI | Can model execute plan step-by-step interactively? |
| P2 TEP | After state corruption, replan or repeat failures? |
| P3 | PlanBench proximity (often low for procedural PDDL) |

---

## 6. Where to verify your numbers

```bash
# Per-variant accuracies
python3 -c "import pandas as pd; print(pd.read_csv('results/derived/probe1_per_model_variant.csv').query(\"probe=='GSM'\"))"

# One problem across variants
python3 -c "
import pandas as pd
df=pd.read_csv('results/raw/GSM_P1_behavioral_claude.csv')
print(df[df.problem_id=='GSM_001'][['variant_type','behavioral_correct']])
"

# P2 GSM session
python3 -c "
import pandas as pd
df=pd.read_csv('results/raw/GSM_P2_phase1_claude.csv')
print(df[df.problem_id=='GSM_001'][['cci_score','tep_score','session_b_correct']])
"
```

---

## 7. Family comparison at a glance

| | GSM | ALGO | BW |
|---|-----|------|-----|
| P1 verifier | numeric extract | strict parser + DP | action simulator |
| P2 interaction | numeric step chain | algorithm steps + injection | PDDL action turns |
| P2 sub-probes | plan + stepwise + TEP | phase1 + normal/elicited/injected | plan extract + CCI + TEP |
| P3 n (canonical) | 44 | 110 | 65 |
| Hardest P2 issue | parse steps | 61 vs 110 denominators | protocol aborts |
| Key P1 metric | W3 retention | subtype VAR + greedy rate | W3 + mystery contrast |
