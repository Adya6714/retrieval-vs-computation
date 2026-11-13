# GSM Evaluation Flow

> **v1 archive.** For new runs use [`docs/v11/README.md`](../v11/README.md) and `rvc11/`.
> Full v1 pipeline map: [MASTER_EVALUATION_PIPELINES.md §3](./MASTER_EVALUATION_PIPELINES.md#3-gsm-arithmetic).

## Quick pointers (v1)

| Piece | Path |
|-------|------|
| Bank | `data/problems/question_bank_gsm.csv` |
| P1 sweep | `scripts/GSM_P1_SCR_run_behavioral_sweep.py` |
| P2 / P3 | see MASTER §3 |
| Frozen numbers | `rebuild/NUMBERS.csv` |

v1.1 bank rows for GSM live in `data/v11/bank.jsonl` (built by `python -m rvc11 build-bank`).
