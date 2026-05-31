# Probe pipeline reference

How each probe is run, verified, and turned into metrics — with **result files tagged per variant (W1–W6)** so you can audit row-by-row.

**Companion docs:** `CHECKLIST.md` (gaps & fixes) · `API_USAGE.md` (manual run log)

**Regenerate derived metrics (no API):**
```bash
python scripts/runs/rederive_all_metrics.py
python scripts/runs/step6_completeness.py
```

---

## 1. Shared setup

### 1.1 Question banks (source of truth)

| Family | Bank file | Canonical n | Variants |
|--------|-----------|-------------|----------|
| GSM | `data/problems/question_bank_gsm.csv` | 44 | canonical + W1–W6 |
| ALGO | `data/problems/question_bank_algo.csv` | 110 (CC/SP/WIS) | canonical + W1–W6 |
| BW | `data/problems/question_bank_bw.csv` | 65 | canonical + W1–W6 |

**GSM bank IDs:** `GSM_001`–`020` and `GSM_041`–`064` (no `021`–`040` in bank).  
Off-bank rows `GSM_021`–`040` in some raw CSVs are **duplicate reruns** — exclude from bank stats.

### 1.2 Variant types (Probe 1)

| Code | Name | Answer changes? | In CSS pool? | Primary metric |
|------|------|-----------------|--------------|----------------|
| `canonical` | Base problem | — | No (reference) | VAR(canonical) |
| **W1** | Lexical paraphrase | No | Yes | VAR(W1), CSS |
| **W2** | Structural reformat | No | Yes | VAR(W2), CSS, VRI structural |
| **W3** | Entity rename | No | Yes | VAR(W3), **W3 retention**, VRI vocabulary |
| **W4** | Formal notation | No | Yes | VAR(W4), CSS, VRI structural |
| **W5** | Reversal | **Yes** | **No** — use RCS | RCS (not CSS) |
| **W6** | Procedural regen | New instance | Yes | VAR(W6), CSS |

**CSS** = fraction of W1,W2,W3,W4,W6 variants matching canonical answer (`probes/behavioral/css.py`).  
**VRI** = mean(W2,W4) − W3 accuracy (`compute_vri` in `css.py`).  
**RCS** = W5 correctness alone (`probes/behavioral/rcs.py`).

### 1.3 Model call contract

- **Runner clients:** `OpenRouterClient` (`probes/behavioral/openai_client.py`), `AnthropicClient`, `ModelClient` (BW multi-turn)
- **Prompting:** zero-shot, temperature 0 (paper scope)
- **Output column:** `raw_response` (or family-specific); `ERROR:` prefix = failed API call → resume with `--resume`
- **Resume:** skip `(problem_id, variant_type, model)` unless latest row is `ERROR:`

### 1.4 Verification routing

```
raw_response → verify_answer() [BW/GSM/planning]
            → verify_algo()   [ALGO subtypes]
            → behavioral_correct / verified bool
```

| Family | Verifier | Logic |
|--------|----------|-------|
| BW | `probes/contamination/verify.py` | Simulate PDDL actions → goal; fallback action-sequence match |
| GSM | `verify_gsm_answer()` | Prefer `#### <num>`; else last numeric token (±0.01) |
| ALGO | `probes/contamination/verify_algo.py` | Subtype parsers (CC, SP, WIS) + `difficulty_params` JSON |

ALGO verifier also sets `parse_status`: `parsed_clean` | `parsed_with_normalization` | `parse_failed`.

---

## 2. Probe 1 — Behavioral invariance

**Question:** Does the model’s **final answer** stay correct under surface perturbations?

### 2.1 Run scripts

| Family | Script | Output pattern |
|--------|--------|----------------|
| BW + GSM | `scripts/BW_P1_SCR_run_behavioral_sweep.py` | `results/raw/{BW\|GSM}_P1_behavioral_<model>.csv` |
| ALGO | `scripts/ALGO_P1_SCR_run_behavioral_sweep.py` | `results/raw/ALGO_P1_behavioral_<model>.csv` |

Post-metrics: `scripts/{GSM,BW,ALGO}_P1_SCR_compute_metrics.py` → `results/derived/*_P1_metrics.csv`

### 2.2 Per-variant raw files (audit one variant at a time)

Filter any P1 behavioral CSV: `variant_type == 'W3'` (etc.).

#### GSM — raw by model

| Model | Raw file | Bank-valid canonical | Notes |
|-------|----------|---------------------|-------|
| Claude | `results/raw/GSM_P1_behavioral_claude.csv` | 44/44 | Full |
| Gemini | `results/raw/GSM_P1_behavioral_gemini.csv` | 44/44 | Full |
| o4-mini | `results/raw/GSM_P1_behavioral_o1mini.csv` | 44/44 | Full |
| GPT-4o | `results/raw/GSM_P1_behavioral_gpt4o.csv` | **40/44 valid** | 24× `ERROR` on GSM_041–064; exclude GSM_021–040 |
| Llama | `results/raw/GSM_P1_behavioral_llama.csv` | **40/44 valid** | Same as GPT-4o |

**Per-variant derived summary:** `results/derived/probe1_per_model_variant.csv` (rows: GSM × model × W1…W6)

| Variant | Derived accuracy column | Also in |
|---------|------------------------|---------|
| canonical | `accuracy` where `variant=canonical` | `P1_per_problem_var_5model.csv` |
| W1 | `variant=W1` | `results/paper/GSM_VAR_all_models.csv` |
| W2 | `variant=W2` | same |
| W3 | `variant=W3` | `probe1_w3_retention.csv`, robustness scatter |
| W4 | `variant=W4` | `GSM_P1_RES_w4_gap.csv` (if generated) |
| W5 | `variant=W5` | `GSM_P1_RES_rcs.csv` |
| W6 | `variant=W6` | partial n=24 (subset bank) |

**Verifier:** `verify_gsm_answer(problem_id, response, ground_truth)`  
**Key raw columns:** `problem_id`, `variant_type`, `raw_response`, `behavioral_correct`, `correct_answer`

#### ALGO — raw by model

| Model | Raw file | Problems |
|-------|----------|----------|
| Claude | `results/raw/ALGO_P1_behavioral_claude.csv` | 110 |
| GPT-4o | `results/raw/ALGO_P1_behavioral_gpt4o.csv` | 110 |
| Llama | `results/raw/ALGO_P1_behavioral_llama.csv` | 110 |
| Gemini | `results/raw/ALGO_P1_behavioral_gemini.csv` | 110 |
| o4-mini | `results/raw/ALGO_P1_behavioral_o1mini.csv` | 110 |

**Human review queue (parse ambiguous):** `results/raw/ALGO_P1_review_queue.csv`

| Variant | What to check | Derived |
|---------|---------------|---------|
| canonical | `verified` + `parse_status` | `ALGO_P1_metrics.csv`, `probe1_per_model_variant.csv` |
| W1 | lexical paraphrase accuracy | per-problem: `P1_per_problem_var_5model.csv` |
| W2 | structural | VRI structural arm |
| W3 | entity rename | `probe1_w3_retention.csv`, `robustness_scatter_data.csv` |
| W4 | formal notation | subtype tables in `results/paper/ALGO_VAR_all_models.csv` |
| W5 | reversal | **RCS only** — not in CSS |
| W6 | new instance | procedural regen accuracy |

**Verifier:** `verify_algo(pid, model_answer, ground_truth, subtype, variant_type, difficulty_params)`  
**Extra columns:** `verified`, `parse_status`, `gave_greedy_answer`, `correct_canonical`

#### BW — raw by model

| Model | Raw file | Notes |
|-------|----------|-------|
| Claude/GPT-4o/Llama | `results/raw/BW_P1_behavioral.csv` (combined) | **Filter** `problem_subtype=blocksworld`; file also contains GSM contamination rows |
| Gemini | `results/raw/BW_P1_behavioral_gemini.csv` | 65 problems |
| o4-mini | `results/raw/BW_P1_behavioral_o1mini.csv` | 65 problems |

| Variant | BW-specific | Derived |
|---------|-------------|---------|
| W3 | block rename | `BW_VAR_all_models.csv`, `gemini_BW_perproblem.csv` |
| W5 | reversal | RCS |
| W6 | new PDDL instance | `generate_w6_variants.py` lineage |

**Verifier:** `verify_answer(..., family="blocksworld")` — plan simulation + sequence match

### 2.3 Probe 1 aggregate metrics (how numbers are computed)

| Metric | Formula / module | Primary derived file |
|--------|------------------|---------------------|
| **VAR(v)** | mean(`behavioral_correct`) for variant v, model m | `probe1_per_model_variant.csv` |
| **W3 retention** | VAR(W3) / VAR(canonical) | `probe1_w3_retention.csv` |
| **CSS** | per-problem: fraction of {W1,W2,W3,W4,W6} matching canonical | `GSM_P1_RES_css.csv`, ALGO metrics |
| **VRI** | mean(VAR(W2), VAR(W4)) − VAR(W3) | `vri_analysis.csv`, `contamination_vri_algo_adversarial.csv` |
| **RCS** | W5 accuracy alone | `GSM_P1_RES_rcs.csv` |
| **Per-problem VAR** | canonical vs each W | `P1_per_problem_var_5model.csv` (940 rows) |

**Bootstrap CIs:** `probes/common/stats.py` (10k resamples) in family metric scripts.

---

## 3. Probe 2 — Plan / execution coupling

**Question:** Does declared strategy match stepwise behavior, and does the model recover after corruption?

### 3.1 GSM Probe 2

**Script:** `scripts/GSM_P2_SCR_run_probe2.py`

| Phase | What happens | Verification |
|-------|--------------|--------------|
| Session A | Model outputs step plan **without** final answer | Step lines parsed (`Step k: ... = value`) |
| Session B | One step per turn until done | `verify_gsm_answer` on final numeric output |
| CCI | Align steps by index; numeric match or cosine ≥0.82 | `probes/behavioral/cci.py` |
| TEP | Inject wrong intermediate at `inject_at_step`; measure recovery | `probes/behavioral/tep.py` |

**Raw files (per model, 44 problems each):**

| Model | Phase-1 declarations | Aggregated CCI |
|-------|---------------------|----------------|
| Claude | `results/raw/GSM_P2_phase1_claude.csv` | `results/raw/GSM_P2_cci.csv` |
| GPT-4o | `results/raw/GSM_P2_phase1_gpt4o.csv` | (merged in cci.csv) |
| Llama | `results/raw/GSM_P2_phase1_llama.csv` | |
| Gemini | `results/raw/GSM_P2_phase1_gemini.csv` | |
| o4-mini | `results/raw/GSM_P2_phase1_o1mini.csv` | **44/44** sessions; 43 parseable |

**Derived:** `results/derived/probe2_gsm_metrics.csv`, `GSM_P2_metrics.csv`  
**TEP analysis:** `results/derived/tep_dissociation_summary.md`

**Key columns:** `session_b_correct`, `cci_score`, `tep_score`, `phase1_parseable`

### 3.2 ALGO Probe 2

Two phases:

#### Phase 2A — strategy declaration (`scripts/ALGO_P2_SCR_run_phase1.py`)

| Track | Raw file | Coverage |
|-------|----------|----------|
| Normal | `ALGO_P2_phase1_<model>_new.csv` | 110 (GPT-4o, Llama, Claude, o4-mini) |
| Normal | `ALGO_P2_phase1_gemini.csv` | 110 |
| Elicited | `ALGO_P2_phase2_normal_elicited.csv` (phase 2 uses elicitation) | Claude/Gemini/Llama **61/110** |

**Fields:** `stated_algorithm`, `greedy_assessment_correct`, `predicted_first_decision`, `critical_point_identified`

#### Phase 2B — stepwise execution (`scripts/ALGO_P2_SCR_run_phase2.py`)

| Track | Raw file | n problems |
|-------|----------|------------|
| Normal | `results/raw/ALGO_P2_phase2_normal.csv` | 110 |
| Normal elicited | `results/raw/ALGO_P2_phase2_normal_elicited.csv` | 110 |
| Injected plausible | `results/raw/ALGO_P2_phase2_injected.csv` | **61** adversarial |
| Injected implausible | `results/raw/ALGO_P2_phase2_injected_implausible.csv` | **61** |

**Per-step classification:** `classify_reasoning_type()` → greedy / forward / algorithm_invocation / backtracking  
**Injection:** at `critical_step_index` from bank `difficulty_params`

**Derived metrics** (`scripts/ALGO_P2_SCR_compute_metrics.py` → `ALGO_P2_metrics.csv`):

| Metric | Meaning |
|--------|---------|
| CCI_algorithm | stated algorithm vs step reasoning |
| CCI_first_decision | phase-1 first decision vs step 0 |
| CCI_critical | critical point vs injection step |
| CCI_composite | combined |
| TEP_refined | post-injection recovery |
| FDI, SC, RDI, RTDA | format divergence, compliance, reasoning divergence, time-to-detect |

**Algorithm-invocation cases:** `results/paper/appendix_algorithm_invocation_cases.csv` (13 cases)

### 3.3 BW Probe 2 (interactive PDDL)

**Scripts:** `BW_P2_SCR_extract_phase1_plans.py` → `BW_P2_SCR_run_cci.py` → `BW_P2_SCR_run_tep.py`  
**NL-tolerant rerun:** `scripts/BW_P2_SCR_run_cci_nl.py`

| Artifact | Path | Coverage |
|----------|------|----------|
| Extracted plans | `results/raw/BW_P2_plans.csv` | 50 problems × 3 models |
| Strict CCI | `results/raw/BW_P2_cci.csv` | 50 × Claude/GPT-4o/Llama |
| NL CCI | `results/raw/BW_P2_cci_nl.csv` | same |
| TEP | `results/raw/BW_P2_tep.csv` | **87% sessions TEP-invalid** (aborts) |

**CCI:** fraction of executed steps matching declared plan (`bw_cci_pipeline.py`)  
**TEP:** post-corruption adaptation (`tep.py`) — mostly `insufficient_data` for BW due to aborts

**Reparse audit:** `results/paper/AUDIT/bw_phase2_reparse_summary.csv`

⚠️ **Do not** treat BW P2 CCI/TEP as five-model comparable until protocol fixed (checklist Step 20).

---

## 4. Probe 3 — Contamination + triangulation

### 4.1 Contamination triage

**Script:** `scripts/BW_P3_SCR_run_contamination_triage.py` (all families via `--family`)

| Family | Raw output | Scoring |
|--------|------------|---------|
| GSM | `results/raw/GSM_P3_contamination.csv` | Infini-gram max n-gram (n≤8 for arithmetic) |
| ALGO | `results/raw/ALGO_P3_contamination.csv` | + decomposed template/instance scores |
| BW | `results/raw/BW_P3_contamination.csv` | canonical rows only |

**Module:** `probes/contamination/score.py` → `score_problem(problem_text)`

### 4.2 Triangulation (per-instance labels)

**Scripts:** `scripts/BW_P3_SCR_run_triangulation.py` (GSM/BW), `scripts/ALGO_P3_SCR_triangulation.py` (ALGO)

**Logic:** `probes/triangulation/per_instance.py` → `align_instance()`

| Signal | Threshold | → label |
|--------|-----------|---------|
| VAR | >0 → computation | else retrieval |
| CSS | ≥0.5 → computation | else retrieval |
| Contamination | >0.6 retrieval; ≤0.4 computation | else ambiguous |
| CCI | ≥0.4 computation | else retrieval |

**Agreement:** ≥2 non-ambiguous agreeing → `converging_*`; else mixed/ambiguous/insufficient

**Output files:**

| Family | Per-instance CSV |
|--------|------------------|
| ALGO | `results/raw/ALGO_P3_triangulation_v3.csv` (440 rows, 4 models) |
| GSM Claude | `results/raw/GSM_P3_triangulation_per_instance_claude.csv` |
| GSM GPT-4o | `results/raw/GSM_P3_triangulation_per_instance_gpt4o.csv` |
| BW | `results/raw/BW_P3_triangulation_{claude,gpt4o,llama}.csv` |

**Exploratory v2 (threshold sweep):** `results/derived/triangulation_v2_labels.csv`, `triangulation_threshold_sweep.csv`

### 4.3 Mechanistic pilot (Probe 3 adjunct)

| File | Model | Scope |
|------|-------|-------|
| `results/raw/mechanistic_sweep_7b.csv` | Qwen-2.5-7B | 268 prompts |
| `results/raw/ALGO_P3_mechanistic.csv` | Qwen-0.5B | 20 rows exploratory |

Not five-model behavioural pool — see checklist Step 21 (Llama replication, local compute).

---

## 5. Derived / paper bundle (cross-probe)

| Purpose | File |
|---------|------|
| Master coverage | `results/derived/master_coverage_table.csv` |
| Coverage gaps | `results/derived/master_coverage_gaps.csv` |
| Five-model per-problem | `results/derived/master_per_problem_5model.csv` |
| Cross-family Table 1 | `results/paper/TABLE1_cross_family.csv` |
| Claim → evidence draft | `results/derived/claim_tagging_draft.md` |
| File-wise audit | `results/derived/scientific_filewise_audit.md` |
| Raw manifest | `results/derived/results_manifest.csv` |
| Pre-API audit | `results/derived/PRE_API_MASTER_AUDIT.md` |

---

## 6. Figures ↔ data sources

| Figure | Path | Primary data | Regenerate |
|--------|------|--------------|------------|
| Robustness scatter | `paper/figures/fig_robustness.pdf` | `robustness_scatter_data.csv`, P1 per-problem VAR | `scripts/audit/robustness_scatter.py` |
| Variant ladder | `paper/figures/fig_decay.pdf` | `probe1_per_model_variant.csv` | `figures/scripts/gen_figures.py` |
| CCI / TEP / injection | `paper/figures/fig_cci.pdf` | P2 metrics CSVs | `figures/scripts/gen_figures.py` |
| Heatmap / proximity | `paper/figures/fig_heatmap.pdf` | contamination + VAR | `figures/scripts/gen_figures.py` |
| ALGO var heatmap | `results/figures/ALGO_P1_FIG_01_var_heatmap.pdf` | ALGO P1 behavioral | `scripts/ALGO_P1_FIG_generate.py` |

**Figure refresh needed after API gap-fill:** GSM GPT-4o/Llama W1–W6 bars and any cross-model GSM table — currently computed on **40/44 valid** rows only (`probe1_per_model_variant.csv`).

---

## 7. Row-by-row audit workflow

For each variant you want to verify:

1. Open the **raw behavioral CSV** for that family + model.
2. Filter: `variant_type == 'W3'` (or W1, W2, …).
3. Check `behavioral_correct` / `verified` vs `raw_response` for spot cases.
4. Compare aggregate to `probe1_per_model_variant.csv` row for that (family, model, variant).
5. If mismatch → check bank filter (exclude off-bank IDs, ERROR rows) in `scripts/runs/rederive_all_metrics.py`.

**Quick variant counts:**
```bash
python3 -c "
import pandas as pd
df=pd.read_csv('results/derived/probe1_per_model_variant.csv')
print(df.to_string())
"
```
