# Step 16 run status — 2026-05-30

## Summary

**GSM P1 gap fill attempted; all API calls failed with 402.**

| Model | Processed | Skipped (resume) | Script errors | Valid new rows |
|-------|-----------|------------------|---------------|----------------|
| GPT-4o | 168 | 120 | 0 | **0** (168 × `ERROR: 402 Payment Required`) |
| Llama | 168 | 120 | 0 | **0** (168 × `ERROR: 402 Payment Required`) |

Log: `results/raw/new_model_sweep_logs/step16_gsm_p1_gpt4o_llama.log`

## Why 402 despite “$34.79 remaining”?

We checked the wrong endpoint before the run.

| Metric | Value | Meaning |
|--------|-------|---------|
| `auth/key` → `limit_remaining` | **$34.79** | Per-key **spending cap** headroom ($100 cap − $65.21 key usage) |
| `credits` → wallet | **−$0.66** | **Account balance** = total purchased − total usage |

OpenRouter bills against the **account wallet**. Yours is slightly **overdrawn** (~$0.66), so every chat completion returns 402 even though the key still has cap room.

**Fix:** Add ~$1+ at https://openrouter.ai/settings/credits (or any top-up that brings wallet > 0).

Preflight now uses `/api/v1/credits` via `scripts/runs/api_budget.py` → `fetch_openrouter_balances()`.

## GSM P2 o4-mini

No API needed — `GSM_P2_phase1_o1mini.csv` remains 44/44 complete from prior sweep.

## Re-run command

```bash
# Verify wallet > 0 first:
python3 -c "from scripts.runs.api_budget import BudgetLogger; print(BudgetLogger.fetch_openrouter_balances())"

python3 scripts/BW_P1_SCR_run_behavioral_sweep.py \
  --model openai/gpt-4o \
  --family arithmetic_reasoning \
  --question-bank-path data/problems/question_bank_gsm.csv \
  --output results/raw/GSM_P1_behavioral_gpt4o.csv \
  --resume

python3 scripts/BW_P1_SCR_run_behavioral_sweep.py \
  --model meta-llama/llama-3.1-8b-instruct \
  --family arithmetic_reasoning \
  --question-bank-path data/problems/question_bank_gsm.csv \
  --output results/raw/GSM_P1_behavioral_llama.csv \
  --resume
```

Then: `python scripts/runs/rederive_all_metrics.py` · `python scripts/runs/pre_api_master_audit.py`
