# BM-05 — Tooling and Costs (what to install, what it costs)

## Free, install now
- **Obsidian** — the vault. Plugins: Dataview (query notes), Templater (forms), later Zotero Integration (papers) and Smart Connections (semantic search).
- **Git + GitHub (private repo)** — version control and backup for the vault and code.
- **Python** — pandas, numpy, scipy, matplotlib, jupyter. The analysis workhorse.
- **R + mirt/sirt** OR **Python + StepMix / py-irt** — for the mixture-IRT work (DS-01). R's mirt is the most mature.
- **TransformerLens** or **nnsight** (Python) — for looking inside open models (D2, DS-03). Only needed when you reach GPU experiments.
- **Fast Downward** — planning verifier for BW/direction experiments (already in the repo toolchain).

## Cheap, later
- **Obsidian Sync** (~$4–5/mo) — phone access; optional (git works free).
- **Model APIs** (OpenAI/Anthropic/OpenRouter) — pay per call. Ingestion + small sweeps: under ~$10/mo. Big behavioral sweeps: tens of dollars each. See [[LB-03_Evaluation_Strategy_Catalog]] for per-experiment estimates.
- **Rented GPUs** (marketplace A100/H100) — only for open-model experiments. D1-lite ≈ $30–60; patching + RSA ≈ $60–300; OLMo sweep ≈ $60–200. Verify current hourly rates; use spot/interruptible to save.

## The honest total
- The research BRAIN itself: ~$0 software + <$10/mo ingestion + your time.
- YEAR ONE of the whole flagship agenda: roughly $300–1,000 with careful model-tier choices; under ~$2K if you use frontier models liberally. Excludes from-scratch training (D1-full, $1–5K) and human studies (optional, $2–4K).
- Start-now cost: $0. The first two novel analyses (HP-13, HP-14) need no paid anything.

## A note on picking models for experiments
- For "look inside" work you NEED open weights: Llama-3.1-8B (your behavioral pool), OLMo (checkpoints + public corpus), Qwen, Pythia.
- For behavioral sweeps you can mix closed (Claude, GPT-4o, Gemini) and open, as the program already does.
- Track every dollar in a `costs.csv` in the repo; update [[LB-03_Evaluation_Strategy_Catalog]] quarterly as prices move.
