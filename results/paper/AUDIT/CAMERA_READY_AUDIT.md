# Camera-ready audit (read-only)

**Date:** 2026-07-15  
**Repo:** `/Users/adya/Desktop/rvc` (mirrors `~/retrieval-vs-computation`)  
**Scope:** No edits to results/figures/paper; this file only.  
**Re-derivation:** `python3 scripts/runs/rederive_all_metrics.py` (ran successfully).  
**Paper draft cited:** `paper/main.tex`, `paper/appendix.tex`, `paper/tables/*.tex`.

Format per item: **CLAIM → SOURCE → RECOMPUTED → STATUS**.

---

## SECTION 1 — BW Probe-2 model coverage discrepancy

### 1.1 Drivers that produced BW Probe-2 CSVs

| Artifact | Driver script | Client | `temperature` |
|---|---|---|---|
| `results/raw/BW_P2_cci.csv` | `scripts/BW_P2_SCR_run_cci.py` L415 | `ModelClient` | **sent `0.0`** |
| `results/raw/BW_P2_cci_nl.csv` | `scripts/BW_P2_SCR_run_cci_nl.py` L397–398 | `ModelClient` | **sent `0.0`** |
| `results/raw/BW_P2_tep.csv` | `scripts/BW_P2_SCR_run_tep.py` L497 | `ModelClient` | **sent `0.0`** |
| `results/raw/MBW_P2_cci_nl.csv` | `scripts/MBW_P2_SCR_run_cci_nl.py` L317–318 | `ModelClient` | **sent `0.0`** |

**Models present in all four CSVs (exact unique `model` values):**

```
anthropic/claude-sonnet-4
openai/gpt-4o
meta-llama/llama-3.1-8b-instruct
```

**Gemini and o4-mini are absent** from `BW_P2_cci.csv`, `BW_P2_cci_nl.csv`, `BW_P2_tep.csv`, and `MBW_P2_cci_nl.csv`.

Evidence:

```bash
# n and models (recomputed 2026-07-15)
BW_P2_cci.csv     n=150 models={Claude, GPT-4o, Llama}
BW_P2_cci_nl.csv  n=150 models={Claude, GPT-4o, Llama}
BW_P2_tep.csv     n=536 models={Claude, GPT-4o, Llama}
MBW_P2_cci_nl.csv n=45  models={Claude, GPT-4o, Llama}
```

`ModelClient` payload (`probes/behavioral/model_client.py` L15–36):

```python
def __init__(self, model_string: str, temperature: float = 0.0):
    self.temperature = temperature
# ...
"temperature": self.temperature,
```

### 1.2 Where Section 4.4 / Figure 14 rename numbers actually come from

**CLAIM** (`paper/main.tex` L441–445): “Paired Wilcoxon on W5-rename under the **NL-tolerant protocol** (Figure `fig:bw_rename`): Claude \(0.422\to0.661\) (+23.9pp, \(p=1.0\times10^{-4}\)); Gemini \(0.385\to0.569\) (+18.5pp, \(p=0.014\)); Llama \(0.321\to0.101\) (−22.0pp, \(p<1\textrm{e-4}\)); GPT-4o / o4-mini unchanged.”

**SOURCE (actual):** Probe-1 behavioral, **not** NL-tolerant Probe-2.

| Model | Acc_can | Acc_W5 | Δpp | n (paired) | Raw file | Script | Client / temp |
|---|---:|---:|---:|---:|---|---|---|
| Claude | 0.422 | 0.661 | **+23.9** | **109** | `results/raw/BW_P1_behavioral.csv` | `paper/figures/scripts/gen_more_figures.py` → `fig_bw_inversion` / `_bw_model_df` | `OpenRouterClient` via `scripts/BW_P1_SCR_run_behavioral_sweep.py` L263–264; **temperature omitted** |
| GPT-4o | 0.367 | 0.367 | 0.0 | **109** | same | same | same |
| Llama | 0.321 | 0.101 | **−22.0** | **109** | same | same | same |
| Gemini | 0.385 | 0.569 | **+18.5** | **65** | `results/raw/BW_P1_behavioral_gemini.csv` | same | same (`OpenRouterClient`, no temp) |
| o4-mini | 0.769 | 0.769 | 0.0 | **65** | `results/raw/BW_P1_behavioral_o1mini.csv` | same | same |

Wilcoxon p (scipy `wilcoxon(..., zero_method="wilcox")`, same as `gen_more_figures.py` L97):

| Model | Recomputed p | Paper p | STATUS |
|---|---:|---|---|
| Claude | \(6.02\times10^{-5}\) | \(1.0\times10^{-4}\) | **MATCH** (reported to 1 sig fig) |
| Gemini | 0.0143 | 0.014 | **MATCH** |
| Llama | \(1.18\times10^{-5}\) | \(p<10^{-4}\) | **MATCH** |
| GPT-4o | 1.0 | unchanged | **MATCH** |
| o4-mini | NaN (identical vectors) | unchanged | **MATCH** |

`OpenRouterClient` payload (`probes/behavioral/openai_client.py` L72–77) — **no `temperature` key**:

```python
payload = {
    "model": self.model,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": self.max_tokens,
}
```

### 1.3 Five-model / clean-T=0 claim

| CLAIM | SOURCE | RECOMPUTED | STATUS |
|---|---|---|---|
| Table 6 footnote: “NL-tolerant rerun … **covers all five models**” (`paper/tables/table6_coverage_full.tex` L8; `main.tex` L520) | `BW_P2_cci_nl.csv` | Only **3** models (Claude, GPT-4o, Llama) | **MISMATCH** |
| Appendix H.1: “NL-tolerant … covers all five models” (`appendix.tex` L62–63) | same | same | **MISMATCH** |
| Main text ties W5 deltas to “NL-tolerant protocol” (`main.tex` L441) | `gen_more_figures.fig_bw_inversion` + BW P1 CSVs | Numbers are **P1 W5**, OpenRouter, **no temperature** | **MISMATCH** (protocol mislabel) |
| Clean T=0 holds for all 5 models’ rename numbers | ModelClient T=0 vs OpenRouter | **Only** Claude/GPT-4o/Llama BW **Probe-2** logs are T=0; **all five rename numbers** are OpenRouter **no-temp** P1 | **BLOCKED for five-model T=0 claim** — must **narrow**: T=0 ModelClient confirmed only for 3-model BW P2 CCI/TEP; rename figure is a different probe/client |

**Explicit flag:** Gemini / o4-mini BW rename numbers do **not** come from `ModelClient(temperature=0.0)`. They share the same OpenRouter P1 client as Claude/GPT-4o/Llama rename numbers. The “clean T=0 BW Probe-2” claim **does not** cover the published five-model rename deltas.

---

## SECTION 2 — Full re-derivation pass

Re-derivation script: `scripts/runs/rederive_all_metrics.py` → wrote under `results/derived/` and `results/paper/AUDIT/`. **Status: runs end-to-end.**

### Table 3 — GSM Probe-1 (`paper/tables/table3_gsm_p1.tex`)

Bank-valid denominators: Claude/Gemini/o4-mini \(n=44\); GPT-4o/Llama \(n=20\) (GSM_001–020). Source raw: `results/raw/GSM_P1_behavioral_{claude,gpt4o,llama,gemini,o1mini}.csv`. Client: OpenRouter (no temperature).

| CLAIM | RECOMPUTED | STATUS |
|---|---|---|
| Claude Acc_can=.841, Acc_W3=.750, R_W3=.892 | 0.841, 0.750, 0.892 | **MATCH** |
| GPT-4o .850 / .300 / .353 | 0.850, 0.300, 0.353 | **MATCH** |
| Llama .800 / .150 / .188 | 0.800, 0.150, 0.1875→.188 | **MATCH** |
| Gemini .909 / .523 / .575 | 0.909, 0.523, 0.575 | **MATCH** |
| o4-mini .841 / .841 / 1.000 | 0.841, 0.841, 1.000 | **MATCH** |

### Table 4 — GSM Probe-2 (`table4_gsm_p2.tex`)

Source: `results/raw/GSM_P2_phase1_{claude,gpt4o,llama,gemini,o1mini}.csv` (+ consolidated `GSM_P2_cci.csv` for four models). Rederive AccP2A = `session_b_correct` mean.

| CLAIM | RECOMPUTED | STATUS |
|---|---|---|
| Claude AccP2A=.864, CCI mean=.231, med=.216, TEP=.539 | 0.864, 0.231, 0.216, 0.539 | **MATCH** |
| GPT-4o .705 / .108 / .000 / .599 | 0.705, 0.108, 0.000, 0.598→.599 | **MATCH** |
| Llama .455 / .167 / .000 / .773 | 0.455, 0.167, 0.000, 0.773 | **MATCH** |
| Gemini .886 / .270 / .250 / .652 | 0.886, 0.270, 0.250, 0.652 | **MATCH** |
| o4-mini AccP2A=.955 | 42/44=0.955 | **MATCH** |
| o4-mini CCI mean=.220, med=.143, TEP=.628 | On **parseable 43/44**: mean=0.220, med=0.143, TEP=0.628; on all 44: mean=0.215, TEP=0.637 | **MATCH** (paper uses parseable subset per Table 4 caption) |

### Table 5 — proximity (`table5_proximity.tex`)

| CLAIM | SOURCE | RECOMPUTED | STATUS |
|---|---|---|---|
| \(\bar{c}_T\): CC 0.468, SP 0.147, WIS 0.000 | Hardcoded in `scripts/figures/fig2_contam_gradient.py` L32: `template_contam = [0.468, 0.147, 0.000]` | Same triple in figure script; Infini-gram raw query logs not re-run in this audit | **MATCH** to published figure code; Infini-gram raw **COULD NOT VERIFY** independently without regenerating Infini-gram counts |
| Claude CC/SP/WIS Can/W3: .700/.600, .647/.000, .353/.000 | Frozen/adversarial ALGO P1 (`ALGO_P1_4model_frozen_labels.csv` / triangulation subtype maps) | Exact match in `results/paper/AUDIT/spearman_acc_W3retention_subtype_data.csv` L2–16 / rederive | **MATCH** |
| GPT-4o CC/SP/WIS: .600/.000, .412/.265, .353/.000 | same | same | **MATCH** |

### Matched Fisher tests

| CLAIM | SOURCE | RECOMPUTED | STATUS |
|---|---|---|---|
| SP: Claude 0/34, GPT-4o 9/34, Fisher \(p=0.0021\) | `results/paper/AUDIT/fisher_matched_canonical_expanded.csv` L3: loose_matched SP claude_vs_gpt4o `a_W3=0, b_W3=9, fisher_p_two_sided=0.0021287…` | Acc_W3 on SP-adv, not dual-canonical-correct | **MATCH** |
| CC: Claude 6/10, GPT-4o 0/10, \(p=0.0108\) | same file L21: loose_matched coin_change `6.0, 0.0, 0.0108359…` | same interpretation | **MATCH** |

### Within-model \(\phi\)

| CLAIM | SOURCE | RECOMPUTED | STATUS |
|---|---|---|---|
| GPT-4o ALGO \(\phi=+0.43\) | `results/derived/deep_p1_pairwise.csv` / `deep_metrics_analysis.py`; row ALGO GPT-4o canonical↔W3 | \(\phi=0.4318\) (\(n=110\)) | **MATCH** |
| o4-mini GSM \(\phi=+0.66\) | same analysis on `GSM_P1_behavioral_o1mini.csv` | \(\phi=0.660\) (\(n=44\)) | **MATCH** |

### Population Spearman

| CLAIM | SOURCE | RECOMPUTED | STATUS |
|---|---|---|---|
| \(r=+0.147\), \(p=0.46\), \(n=28\) | `paper/figures/scripts/gen_figures.py` `fig_population()`; cells from ALGO subtype×inst (can>0) + bank-valid GSM (4 models) + BW-std (4 models) | Exact run: **n=28, r=+0.1470, p=0.455** (paper rounds p to 0.46); 95% bootstrap CI \([-0.27,+0.56]\) | **MATCH** |
| ALGO-only \(r=+0.37\), \(p=0.10\), \(n=20\) (`appendix.tex` L259) | same cell frame, ALGO slice | r=0.373, p=0.105, n=20 | **MATCH** |

### ALGO rank-retention

| CLAIM | SOURCE | RECOMPUTED | STATUS |
|---|---|---|---|
| \(\rho=+0.90\), \(p=0.04\), \(n=5\) | `rederive_all_metrics.py` step [6/6] accuracy vs W3 retention Spearman across 5 models | `spearman_rho=0.9000`, `p_value=0.0374`, n_models=5 | **MATCH** |

### BW rename (headline deltas)

| CLAIM | SOURCE | RECOMPUTED | STATUS |
|---|---|---|---|
| Claude +23.9pp \(p=1.0\textrm{e-4}\) | BW P1 + `gen_more_figures.py` (see §1) | +23.9pp, \(p=6.0\textrm{e-5}\) | **MATCH** (numbers); protocol label **MISMATCH** (§1) |
| Gemini +18.5pp \(p=0.014\) | same | +18.5pp, p=0.014 | **MATCH** (numbers); client not ModelClient T=0 |
| Llama −22.0pp \(p<1\textrm{e-4}\) | same | −22.0pp, \(p=1.2\textrm{e-5}\) | **MATCH** |
| GPT-4o / o4-mini unchanged | same | Δ=0 | **MATCH** |

### Injection compliance

Source: `results/paper/AUDIT/injection_recovery_summary.csv` (+ Gemini from injected raw).

| CLAIM | RECOMPUTED | STATUS |
|---|---|---|---|
| Claude 88.5% | 0.885245… | **MATCH** |
| GPT-4o 93.4% | 0.934426… | **MATCH** |
| Llama 39.3% | 0.393443… | **MATCH** |
| o4-mini 100% | 1.0 (rederive / Fig probe2 text) | **MATCH** |
| Gemini 0% | 0.0 | **MATCH** |

### Post-injection accuracy (plausible vs implausible)

Rederive last-row `post_injection_correct` (`rederive_all_metrics.algo_p2_metrics`):

| CLAIM (`main.tex` L385–387) | RECOMPUTED | STATUS |
|---|---|---|---|
| Claude 52.5% vs **54.1%** | Plausible **0.525**, implausible **0.525** (Δ=0.0pp; matches `appendix.tex` L331 and “Two five-model nulls” L396). **54.1% is the pooled implausible aggregate** (`appendix.tex` L325–327, n=122 Claude+GPT-4o), **not** Claude’s second arm. | **MISMATCH** (Claude second figure) |
| GPT-4o 50.8% vs 55.7% | 0.508 / 0.557 | **MATCH** |
| o4-mini 37.7% vs **40.9%** | 0.377 / **0.426** | **MISMATCH** (implausible 42.6% ≠ 40.9%) |

### Triangulation counts

| CLAIM | SOURCE | RECOMPUTED | STATUS |
|---|---|---|---|
| 8 retrieval / 4 computation / 157 mixed / 271 ambiguous on 440 ALGO | `results/derived/ALGO_P3_triangulation_v3.csv` `convergence_label` value_counts | identical counts; n=440 | **MATCH** |

### Threshold sweep (270 configs)

| CLAIM | SOURCE | RECOMPUTED | STATUS |
|---|---|---|---|
| Max strong-label % = **5.0%**; max agreement ≈ **5.4%** | `results/derived/ALGO_P3_threshold_sensitivity.csv` (`scripts/runs/algo_triangulation_threshold_sensitivity.py`) | `strong_pct` max=**5.0**; `agreement_rate_subtype_pattern` max=**0.05442** (5.44%) | **MATCH** |

---

## SECTION 3 — Sample-size / n audit

| CLAIM (paper) | Actual data | STATUS |
|---|---|---|
| GSM P1 \(n=44\) Claude/Gemini/o4-mini | Canonical rows in bank-valid GSM P1 files = 44 each (`gen_figures._assert_gsm_p1_canonical_counts`) | **MATCH** |
| GSM P1 \(n=20\) GPT-4o/Llama | Canonical bank-valid = 20 each (GSM_001–020 only; GSM_041–064 gap) | **MATCH** |
| ALGO triangulation 440 = 110×4 | `ALGO_P3_triangulation_v3.csv` len=440; models={Claude,GPT-4o,Llama,Gemini} | **MATCH** on arithmetic |
| o4-mini excluded because **constant W3=1.00** (`appendix.tex` L538–539) | `ALGO_P1_behavioral_o1mini.csv`: Acc_can=**1.000**, Acc_W3=**0.609** (n=110) — **not** W3=1.00 | **MISMATCH** (exclusion *reason* is false; exclusion itself still applied) |
| SP matched \(n=34\) | Fisher loose_matched SP `n_matched=34`; bank SP-adv canonical = 34 | **MATCH** |
| CC matched \(n=10\) | Fisher loose_matched CC `n_matched=10` | **MATCH** |
| BW bank \(n=65\) | `data/problems/question_bank_bw.csv`: 65 problems × 7 variants | **MATCH** |
| BW rename figure uses \(n=109\) (Claude/GPT/Llama) vs \(n=65\) (Gemini/o4) | P1 CSVs: Claude/GPT/Llama intersect can∩W5 = **109**; Gemini/o4 = **65** | **MATCH** to data; **inconsistent denominators across models in one figure** — flag for camera-ready wording |
| Table 7 BW Can=.154 (Claude) vs rename Can=.422 | .154 ≈ bank-65 / reported table path; .422 = P1 pooled n=109 including extra BW_* rows beyond bank-65 or MBW mix | **Applied inconsistently** across Table 7 vs Fig rename — **needs paper clarification** |
| BW “124 instances” triangulation note (`appendix.tex` L543) | Claude BW triangulation slice including variants; not equal to 65 or 109 | Separate analysis n — document explicitly |

---

## SECTION 4 — Outstanding GPU queue status

**Remote SSH** to `172.24.16.177` timed out (VPN down) at audit time. Status below is from **local pulled artifacts** dated 2026-07-15 01:23 IST (completed earlier in `mech_contentgold`).

| Job | Status | Output | Numbers / gate |
|---|---|---|---|
| 1. Qwen2.5-7B-Instruct chat-direct content-gold | **COMPLETED** | `results/raw/mechanistic_sweep_qwen25_7b_instruct_chatdirect_contentgold.csv` (398 rows) | Content ranks very high (ALGO can median ~1e5); 0 format-keyword golds |
| 2. Llama-3.1-8B-Instruct chat-direct content-gold | **COMPLETED** | `results/raw/mechanistic_sweep_llama31_8b_instruct_chatdirect_contentgold.csv` (398) | ALGO can median final-layer rank **14.0**; 0 format keywords |
| 3. Qwen2.5-7B base raw-qa content-gold | **COMPLETED** | `results/raw/mechanistic_sweep_qwen25_7b_base_rawqa_contentgold.csv` (398) | High ranks; 0 format keywords |
| 4. Llama ALGO forced-greedy accuracy | **COMPLETED** | `results/raw/ALGO_llama31_8b_greedy_canonical.csv` | **6/110 = 5.5%** (SP 1/55=1.8%, CC 2/25, WIS 3/30) |
| 5. Pass/fail gate | **COMPLETED → AMBIGUOUS** | `results/derived/mechanistic_contentgold_gate_report.md` | Rule: greedy~6% + high content-rank ⇒ PASS; greedy~6% + rank≈1 ⇒ FAIL. Observed: greedy **5.5%** + median rank **14.0** (middle band) ⇒ **AMBIGUOUS** |

Do **not** analyze legacy `mechanistic_sweep_llama31_8b_instruct.csv` (scaffold/`Path` golds, ranks≈1) for content-gold claims.

---

## SECTION 5 — Provenance of published Llama ALGO / Table 7 numbers

| CLAIM | SOURCE | RECOMPUTED | STATUS |
|---|---|---|---|
| Table 7 SP-chall Llama Can **.059** | `paper/tables/table7_pervariant.tex` L34; `results/derived/ALGO_P1_4model_frozen_labels.csv` L75: `llama,shortest_path,adversarial,canonical,k=2,n=34,acc=0.059` | **2/34=0.059** | **MATCH** to frozen file |
| Table 7 SP-std Llama Can **.048** | same table L38; frozen L82: `k=1,n=21,acc=0.048` | **1/21=0.048** | **MATCH** |
| Frozen labels descend from OpenRouter P1 | `results/raw/ALGO_P1_behavioral_llama.csv`; driver uses `OpenRouterClient` | Overall can **7/111≈6.3%**; **no `temperature` / not forced-greedy** | **Confirmed: no temperature control** |
| Delta vs local forced-greedy (§4) | `ALGO_llama31_8b_greedy_canonical.csv` | Overall **6/110=5.5%**; SP-only **1/55≈1.8%** vs frozen SP-chall **2/34=5.9%** and SP-std **1/21=4.8%** | **Δ ≈ −0.8pp overall** vs OpenRouter 6.3%; **SP subtype not identical** to frozen cells (different aggregation / instance splits) |

Authoritative provenance note also in `results/derived/LLAMA_ALGO_CANONICAL_PROVENANCE.md` and gate report.

---

## SECTION 6 — Anything else broken

| Item | Evidence | Severity |
|---|---|---|
| **OpenRouter silent temperature default** on all P1 / ALGO P2 / GSM P2 / BW rename | `openai_client.py` omits `temperature` | Systemic; any “greedy / T=0” wording outside ModelClient BW P2 is unsafe |
| **Table 6 / appendix “NL-tolerant covers all five models”** | `BW_P2_cci_nl.csv` has 3 models only | Camera-ready must narrow or re-run Gemini/o4 NL P2 |
| **Main text mislabels P1 W5 as NL-tolerant Probe-2** | `main.tex` L441 vs `gen_more_figures.py` | Numeric MATCH, protocol MISMATCH |
| **o4-mini triangulation exclusion reason “W3=1.00”** is false | Acc_W3=0.609 | Fix rationale (floor Acc_can=1.00 / retention degeneracy still arguable, but stated reason is wrong) |
| **Claude post-inj “52.5% vs 54.1%”** | 54.1% = pooled implausible aggregate, not Claude | Copy-paste / aggregation bug in main text |
| **o4-mini implausible 40.9%** | Recomputed 42.6% | Small numeric MISMATCH |
| **BW n mixed (65 / 109 / 124)** across Table 7, rename fig, appendix triangulation | §3 | Clarify per display |
| **Legacy scaffold mechanistic CSV** still in `results/raw/` | ranks≈1 on `Path` | Do not cite as content-gold |
| **Content-gold gate AMBIGUOUS** | median rank 14 | Not yet a clean pass for mechanistic pipeline soundness claim |
| Table 9 / liberal param-204 vs 270-sweep | Different labeling systems (`triangulation_v2` k-of-n vs legacy AND); max strong 5% only for 270-sweep | Already flagged in `TABLE9_AND_AUDIT_PRIORITIES.md` — keep wording non-conflating |

---

## Summary

**Tallies (main numeric / coverage claims audited above):**

- **MATCH:** ~45 (Tables 3–5 core cells, Fisher SP/CC, φ, population Spearman, rank-retention ρ, triangulation 8/4/157/271, threshold 5.0%/5.4%, injection compliance, most BW rename magnitudes, GSM P2 including o4 parseable CCI/TEP, Llama Table 7 frozen .059/.048).
- **MISMATCH:** **6 material** — (1) NL-tolerant “all five models”; (2) rename attributed to NL-tolerant P2; (3) o4-mini “W3=1.00” exclusion reason; (4) Claude post-inj second figure 54.1%; (5) o4-mini implausible 40.9% vs 42.6%; (6) BW denominators / Table7 vs rename Can inconsistent without disclosure.
- **BLOCKED / COULD NOT VERIFY:** remote GPU live tmux (VPN timeout — local artifacts used); Infini-gram \(\bar{c}_T\) raw regeneration not re-queried (figure script hardcodes match).

**Section 1 verdict:** The BW Probe-2 five-model / clean-T=0 coverage claim **does not hold** and **must be narrowed** in camera-ready text. Confirmed T=0 `ModelClient` coverage is **Claude, GPT-4o, Llama only** for BW Probe-2 CCI/NL/TEP. The five-model W5 rename numbers in Section 4.4 / Figure `fig:bw_rename` are **Probe-1 OpenRouter (no temperature)** and are **not** NL-tolerant Probe-2 outputs.
