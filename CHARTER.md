# RESEARCH REPORT  
## Beyond Accuracy — Retrieval vs Computation in LLM Reasoning  

**Version 2.0 | April 22, 2026**

**Authors:** Adya (lead), Shaswat, Nandini Banka  
**Affiliation:** BITS Pilani  
**Target Venues:** BlackboxNLP 2026, GenBench 2026 (primary); EMNLP Findings (stretch)  
**Repo:** https://github.com/Adya6714/retrieval-vs-computation  

---

# PART 0: HOW TO READ THIS DOCUMENT

This document is the single authoritative reference for the research design, methodology, current empirical state, what we know, what is broken, what needs to be done, and how to frame the work for publication. It supersedes all prior research reports.

Every section is written so that a new collaborator, or you yourself returning after a two-week gap, can understand exactly what is happening and why.

## Structure

- **Parts 1–2:** Research question, theoretical framework, novelty  
- **Parts 3–5:** Three probes (design, measurement, results)  
- **Part 6:** Empirical findings (valid, invalid, missing)  
- **Part 7:** New metric system  
- **Part 8:** Prior work positioning  
- **Part 9:** Limitations  
- **Part 10:** Problem set design  
- **Part 11:** Visualization plan  
- **Part 12:** Novelty + publication strategy  

---

# PART 1: THE RESEARCH QUESTION

## 1.1 The Core Problem

When an LLM produces a correct answer, that correctness is **epistemically ambiguous**:

### Interpretation A — Retrieval
- Model has seen similar problem during training  
- Performs pattern matching  
- Reasoning is *retrieval disguised as inference*

### Interpretation B — Computation
- Model constructs solution from structure  
- Tracks state, applies rules, plans  
- Output derived from logic, not memory  

### Key Behavioral Difference

| Property | Retrieval | Computation |
|--------|----------|------------|
| Surface changes | Breaks | Stable |
| Rephrasing | Fails | Consistent |
| Entity renaming | Fails | Consistent |

**Central Question:**  
Can we determine which mode is operating per instance, model, and domain?

---

## 1.2 Why This Matters

### Practical Stakes
- Benchmarks may reflect **contamination**, not reasoning  
- Models may fail on **novel problems**  
- Deployment in reasoning-critical tasks becomes risky  

### Scientific Stakes
- Revisits **connectionist vs symbolist debate (1980s)**  
- LLMs are the modern testbed  

---

## 1.3 Why Existing Work Falls Short

| Area | Limitation |
|------|-----------|
| Behavioral probing | Cannot isolate surface vs structural effects |
| Contamination studies | Confounded with difficulty |
| Mechanistic work | Limited to toy tasks |

### Contribution
**Triangulation:**  
Run all three probes on same instances → check agreement  

---

# PART 2: THEORETICAL FRAMEWORK

## 2.1 Triangulation Metaphor

We infer hidden processing mode using **three independent axes**:

- **Input axis:** sensitivity to surface form  
- **Process axis:** plan vs execution consistency  
- **Data axis:** training data proximity  

Agreement across axes = strong evidence

---

## 2.2 Retrieval vs Computation (Clarification)

Not mechanistic distinctions → **behavioral signatures**

- **Retrieval-like:** surface-dependent + contamination-correlated  
- **Computation-like:** structure-stable + contamination-independent  

---

## 2.3 Falsifying Retrieval Hypothesis

Would require:
- High CSS across variants  
- No contamination correlation  
- High CCI  
- Strong MBW performance  

**Current results:** consistent with retrieval

---

# PART 3: PROBE 1 — SURFACE INVARIANCE

## 3.1 What It Measures

Same problem → multiple variants → compare outputs

---

## 3.2 Variant Types

| Variant | Description | Expected Effect |
|--------|------------|----------------|
| W1 | Paraphrase | Minimal change |
| W2 | Format change | Small effect |
| W3 | Vocabulary rename | Catastrophic |
| W4 | PDDL format | Mixed |
| W5 | Reverse task | Separate metric |
| W6 | New instance | Contamination control |

---

## 3.3 Valid Results

- **Canonical:** 0% across all models  
- **W3:** 0% universally  
- **MBW:** 0% universally  
- **W2/W4:** sometimes improve (15–20%)  
- **W5 (teardown):**
  - Claude: ~93%
  - GPT-4o: ~57%
  - Llama: 0%

---

## 3.4 Invalid Results

- MBW W3 → wrong action schema  
- BW_E rows → incorrect answers  
- Lowercase variant labels → not evaluated  
- Contamination labels incorrect  

---

## 3.5 Primary Metric: CSS

**Consistency Surface Score**

At 0% accuracy → measures **consistency of failure**, not correctness  

---

## 3.6 Secondary Metrics

- RCS (reversal correctness)  
- CAS (failure consistency)  

---

# PART 4: PROBE 2 — PLAN-EXECUTION COUPLING

## 4.1 Design

- Phase 1: generate full plan  
- Phase 2: execute step-by-step  

Compare alignment  

---

## 4.2 State Injection

Inject false state → observe reaction  

---

## 4.3 Critical Bug

Parser mismatch caused:
- All actions invalid  
- No state updates  
- Infinite loops  

→ **All Phase 2 results invalid**

---

## 4.4 Metrics

- CCI (plan alignment)  
- TEP (state sensitivity)  
- RR, FIS, PGA (new)

---

# PART 5: PROBE 3 — CONTAMINATION INDEXING

## 5.1 Method

Uses **Infini-gram API** to compute contamination score  

---

## 5.2 Regression Results

| Model | β | R² | Significant |
|------|--|----|------------|
| GPT-4o | 1.573 | 0.424 | Yes |
| Llama | 0.433 | 0.346 | Yes |
| Claude | ~0.3 | ~0.12 | No |

---

## 5.3 Limitations

- Non-normal residuals  
- Bimodal contamination  
- CSS floor effect  
- Mislabelled data  

---

## 5.4 Score Issue

Mixes:
- Template contamination  
- Instance contamination  

---

# PART 6: KEY FINDINGS

## 6.1 Universal Failure
All models fail Blocksworld

---

## 6.2 W3 Collapse
Vocabulary change → total failure  

---

## 6.3 MBW Collapse
Predicate renaming → zero performance  

---

## 6.4 Planning Direction Asymmetry

| Model | Forward | Teardown |
|------|--------|----------|
| Claude | 0% | ~93% |
| GPT-4o | 0% | ~57% |
| Llama | 0% | 0% |

### PDAS
- Claude: 0.929  
- GPT-4o: 0.571  

---

## 6.5 Representation Effect
W2/W4 improvements → likely retrieval alignment  

---

## 6.6 Contamination Correlation
Strong but driven by BW_E cluster  

---

# PART 7: NEW METRICS

- **VAR:** accuracy per variant  
- **PDAS:** direction asymmetry  
- **VRI:** vocabulary vs structure robustness  
- **DTS:** domain transfer  
- **CFS:** failure consistency  
- **PVR:** validity breakdown  

---

# PART 8: PRIOR WORK

- PlanBench (Valmeekam)  
- Kambhampati (planning limits)  
- Mirzadeh (robustness)  
- Berglund (reversal curse)  
- Liu (Infini-gram)  
- Wei (CoT)  

---

# PART 9: LIMITATIONS

- Small N  
- CSS floor effect  
- Contamination ambiguity  
- Single domain  
- No mechanistic results  
- Missing W1  
- Behavioral (not mechanistic) claims  

---

# PART 10: PROBLEM SET DESIGN

## Target
45 problems (3 families)

### Families
1. Planning  
2. Arithmetic  
3. Algorithmic  

---

# PART 11: VISUALIZATION PLAN

Key figures:
- Variant heatmap  
- PDAS bar chart  
- Domain transfer  
- Contamination scatter  
- Failure taxonomy  

---

# PART 12: NOVELTY & STRATEGY

## Contributions

1. Multi-axis triangulation  
2. W3 total collapse  
3. Planning asymmetry (PDAS)  
4. MBW collapse  
5. (Pending) step-by-step failure  

---

## Publication Plan

- Submit to **BlackboxNLP 2026** (pilot)  
- Extend → **EMNLP Findings**

---

## Compute Note

- 7B cannot run on T4  
- Use:
  - Colab Pro (A100)  
  - Institutional GPU  

---
