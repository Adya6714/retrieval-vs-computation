# API usage

Manual log only. After each run session, add one line: **done / total**.

**Current status:** ⛔ **API unavailable** (wallet −$0.66, 402 on all calls). No new runs until topped up. Work continues on analysis + documentation only.

## Budget (2026-05-30)

| Metric | Value | Source |
|--------|-------|--------|
| Per-key cap headroom | $34.79 | `GET /api/v1/auth/key` → `limit_remaining` |
| **Account wallet (spendable)** | **−$0.66** | `GET /api/v1/credits` → `total_credits − total_usage` |

**Important:** Preflight must check **wallet**, not key cap. Key can show headroom while account is overdrawn → 402 on every call.

Check: `python3 -c "from scripts.runs.api_budget import BudgetLogger; print(BudgetLogger.fetch_openrouter_balances())"`

---

## Runs

| Date | What | Progress | Notes |
|------|------|----------|-------|
| 2026-05-30 | Step 16 GSM P1 GPT-4o | 168 / 168 attempted | All 402 — wallet overdrawn, not key cap |
| 2026-05-30 | Step 16 GSM P1 Llama | 168 / 168 attempted | All 402 — same root cause |

Log: `results/raw/new_model_sweep_logs/step16_gsm_p1_gpt4o_llama.log`

---

## Done this cycle

| What | Progress | Date |
|------|----------|------|
| GSM P2 o4-mini (prior sweep) | 44 / 44 | 2026-05-24 |
| Step 16 GSM P1 gap fill | **blocked — wallet −$0.66; top up ~$1+** | 2026-05-30 |
