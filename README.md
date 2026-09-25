# Retrieval vs computation

Equal benchmark accuracy can be reached by procedures that behave differently under surface change. This repository is a measurement programme for that distinction: six controlled surface variants, three problem families (arithmetic, planning, algorithmic optimisation), and the model roster below. The instrument records whether a verified answer survives a named transform, whether a declared plan matches isolated execution, and whether a mid-solve injection is accepted without changing the final answer.

## Status

Status, Sept 2026: behavioural layer shipped (Phase 0, gate G0 passed). Per-instance labels are not issued yet; they wait on calibration gate G1, which needs a GPU. Open-weight Track T is running on a T4.

## Papers

| Artefact | Description |
|----------|-------------|
| [`paper/main.tex`](paper/main.tex) / [`paper/main.pdf`](paper/main.pdf) | Paper I draft: What Survives a Rename (NeurIPS 2026 E&D format) |
| [`paper/venue/caisc2026/`](paper/venue/caisc2026/) | Accepted predecessor: *Same Score, Different Strategy* (CAISc 2026) |

**Code:** [github.com/Adya6714/retrieval-vs-computation](https://github.com/Adya6714/retrieval-vs-computation).  
**Teaching walkthrough:** [BOOK.md](BOOK.md).  
**Site:** [GitHub Pages](https://adya6714.github.io/retrieval-vs-computation/).

---

## Models

Roster generated from [`configs/models.yaml`](configs/models.yaml) by `scripts/consolidate/make_model_table.py`. Paper analyses use five primary behavioural models plus DeepSeek-R1-distill-Llama-70B on planning only (see `results/derived/COVERAGE_MASTER.csv`).

<!-- MODELS:START -->
| Model | ID | Via | Behavioural | Mechanistic | Priority | Role |
|---|---|---|:---:|:---:|---|---|
| Claude Sonnet 4 | `anthropic/claude-sonnet-4` | OpenRouter | yes | no | required | Anchor. PDAS=1.000 finding. |
| GPT-4o | `openai/gpt-4o` | OpenRouter | yes | no | required | Frontier closed-source comparison. |
| Llama 3.1 8B | `meta-llama/llama-3.1-8b-instruct` | OpenRouter | yes | no | required | Capability floor. Open-source reproducibility. |
| DeepSeek-R1-Distill-Llama-70B | `deepseek/deepseek-r1-distill-llama-70b` | OpenRouter | yes | no | optional | Reasoning-specialized (cost-efficient distill). |
| Qwen2.5 72B | `qwen/qwen-2.5-72b-instruct` | OpenRouter | yes | no | required | Large open-source, different training distribution. |
| Qwen2.5 7B (local HF) | `Qwen/Qwen2.5-7B-Instruct` | local HF | no | yes | optional | mechanistic probes only; behavioral sweeps use qwen2.5-72b via OpenRouter |
<!-- MODELS:END -->

---

## Headline findings

Values rounded from `results/derived/` as listed in [Reproduce](#reproduce).

- **Numeric regeneration is nearly free; cover-story change is expensive.** Gemini GSM canonical **.909** → W6 **.958**, canonical **.909** → W3 **.523** (`probe1_per_model_variant.csv`).
- **Formal notation is often costlier than renaming.** o4-mini GSM canonical **.841** → W4 **.682** (same file); W3 on that cell is **.841**.
- **Planning direction inversion raises accuracy.** On the Blocksworld ranker used for concordance, Claude W5 accuracy is **.869** against a W3 of **.375** (`C7_concordance_filtered.csv`). The manuscript table reports Claude canonical **.172** → W5 **.873**.
- **No item-level locus of fragility** among items scored on all five primary models: **0/110** ALGO, **0/20** GSM, **14/64** BW (`P1_failure_patterns.csv`, `fail_all_five_paper_models`).
- **Transform difficulty is a family property.** Kendall \(W\) within family: ALGO **.419** (\(p=.049\)), BW **.679** (\(p<.001\)), GSM **.660** (\(p=.042\)). Across families within Claude: \(W=.165\), \(p=.824\) (`P1_variant_ordering.csv`).
- **Compliance is not correctness under injection.** o4-mini accepts **100%** of ALGO Phase-2B injections (`rebuild/NUMBERS.csv` / frozen P2.3 cell; \(n=61\)). Pooled post-injection accuracy on the injected logs is **.390** (`scientific_file_profiles.csv`, `ALGO_P2_phase2_injected.csv`), near the uninjected baseline in the paper.

---

## Programme

| Stage | Status | What it needs | Decision criterion |
|-------|--------|----------------|--------------------|
| **1. Behavioural instrument** — W1–W6 surface battery, plan–execution coupling, injection, public-corpus proximity | **Complete** | Frozen banks + append-only raw logs (this repo) | A named transform produces a drop (or rise) that is not an oracle artefact; item-level shared-hard counts and Kendall \(W\) are recoverable from `results/derived/`. **Refute** if W1–W6 order is noise (\(W\) not distinguishable from the within-row permutation null) or if a single item set fails for every model outside obfuscated planning. |
| **2. Mechanism** — residual-stream / rank of the answer token under rename vs numeric change | **Pilot** | Unquantised open-weight model that solves enough renamed items for a within-model correlation | Rename-induced drop in gold-token accessibility predicts per-item W3 survival. **Refute** if the correlation is absent at adequate power, or if numeric (W6) and lexical (W3) conditions produce the same rank shift. Current open-weight pilots solve too few renamed items (\(1/60\), \(0/61\)) to run that test. |
| **3. Training conditions** — known membership, surface diversity vs dose | **Designed** | Open-corpus model (e.g. OLMo/Dolma) plus controlled fine-tune arms | Behavioural indicators recover held-out membership (pre-registered AUC). Diversity of surfaces, not template dose, raises W3 retention. **Refute** if labels do not track exposure, or if dose and diversity are interchangeable. |
| **4. Developmental** — checkpoint sweep | **Planned** | Intermediate checkpoints of one open training run | Surface invariance appears after canonical accuracy, with a measurable lag. **Refute** if invariance and accuracy move together at every checkpoint. |

---

## Where everything lives

| Path | What it is |
|------|------------|
| `data/problems/` | Question banks: `question_bank_{gsm,bw,algo}.csv` |
| `probes/` | Shared library (variants, verifiers, CCI/TEP, clients) |
| `scripts/` | Sweeps (`*_SCR_*`), figures (`*_FIG_*`), generation, consolidate |
| `results/raw/` | Append-only per-instance logs (resume on problem × variant × model) |
| `results/derived/` | Aggregates cited in the paper (recomputable from raw + banks) |
| `paper/` | Manuscript, tables, figures, PDF; CAISc archive under `paper/venue/caisc2026/` |
| `site/` | Interactive site (GitHub Pages) |
| `research-vault/` | Programme vault (`THE_PLAN.md`, notes, handoffs) |
| **Plan** | [`research-vault/RvC-Vault/THE_PLAN_AMENDMENT_A1.md`](research-vault/RvC-Vault/THE_PLAN_AMENDMENT_A1.md) (Track T amendment) · [`docs/audit/REPO_AUDIT_2026-09-25.md`](docs/audit/REPO_AUDIT_2026-09-25.md) (verified hygiene / Paper I blockers) |

**Banks.** GSM (arithmetic; GSM-Symbolic templates), BW (Blocksworld / mystery; PlanBench PDDL), ALGO (coin change, shortest path, weighted interval scheduling). Shared schema includes `problem_id`, `variant_type`, `variant_subtype`, `w3_kind`, `problem_text`, `correct_answer`, `problem_family`, `problem_subtype`.

**Variants.** W1 paraphrase, W2 reformat, W3 cover-story isomorph (W3b; nonce W3a in progress), W4 formal notation, W5 direction reversal (init/goal swap on BW), W6 procedural regeneration (new numbers / new instance, gold re-solved).

**Probes.** P1 behavioural sweep; P2 plan vs execution (CCI) and false-state injection (TEP); P3 Infini-gram proximity and per-instance triangulation.

---

## Reproduce

No API key is required to recompute aggregates from committed raw logs.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.
python scripts/runs/rederive_all_metrics.py
python scripts/consolidate/p1_variant_ordering.py
python scripts/consolidate/p1_failure_patterns.py
python scripts/consolidate/make_model_table.py
```

| Finding | Command that writes the table | Derived file | Cell |
|---------|-------------------------------|--------------|------|
| Gemini GSM .909 / .958 / .523 | `python scripts/runs/rederive_all_metrics.py` | `results/derived/probe1_per_model_variant.csv` | `GSM,Gemini,{canonical,W6,W3}` → `0.9090909`, `0.9583333`, `0.5227273` |
| o4-mini GSM .841 → .682 (W4) | same | same | `GSM,o4-mini,{canonical,W4}` → `0.8409091`, `0.6818182` |
| Claude BW direction (W5 .869) | `python scripts/consolidate/p1_variant_ordering.py` (C7 filter companion) | `results/derived/C7_concordance_filtered.csv` | ranker `BW,anthropic/claude-sonnet-4`, `acc_W5=0.8687`, `acc_W3=0.375` |
| Shared-hard 0/110, 0/20, 14/64 | `python scripts/consolidate/p1_failure_patterns.py` | `results/derived/P1_failure_patterns.csv` | `fail_all_five_paper_models` |
| Kendall \(W\) .419 / .679 / .660; Claude \(W=.165\) | `python scripts/consolidate/p1_variant_ordering.py` | `results/derived/P1_variant_ordering.csv` | `within_family_across_models`; `within_model_across_families` Claude |
| Injection acceptance 1.0; post-injection .390 | P2 metric pass inside `rederive_all_metrics.py` | `results/derived/scientific_file_profiles.csv` (`post_injection_correct` on `ALGO_P2_phase2_injected.csv`); frozen cell `P2.3.ALGO.o4-mini.compliance.compliant` in `rebuild/NUMBERS.csv` | \(n=61\) sessions |

Print the GSM headline row:

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("results/derived/probe1_per_model_variant.csv")
print(df[(df.probe=="GSM") & (df.model=="Gemini")])
print(df[(df.probe=="GSM") & (df.model=="o4-mini")])
PY
```

**Fresh model calls** (API): `OPENROUTER_API_KEY` in `.env`; family sweeps `scripts/{ALGO,GSM,BW}_P1_SCR_run_behavioral_sweep.py` with `--resume`. Bank regeneration needs `make bootstrap` (PlanBench, GSM-Symbolic, Fast Downward).

**Tests:** `PYTHONPATH=. pytest tests/ -v`

---

## Layout

```
rvc/
├── data/problems/
├── probes/
├── scripts/
├── results/raw/
├── results/derived/
├── paper/main.tex
├── paper/main.pdf
├── paper/venue/caisc2026/
├── BOOK.md
├── research-vault/
└── tests/
```
