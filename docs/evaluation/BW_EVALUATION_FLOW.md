# Blocksworld (BW) Evaluation Flow

> **v1 archive.** For new runs use [`docs/v11/README.md`](../v11/README.md) and `rvc11/`.
> Full v1 pipeline map: [MASTER_EVALUATION_PIPELINES.md §2](./MASTER_EVALUATION_PIPELINES.md#2-blocksworld-bw).

## Quick pointers (v1)

| Piece | Path |
|-------|------|
| Bank | `data/problems/question_bank_bw.csv` |
| P1 sweep | `scripts/BW_P1_SCR_run_behavioral_sweep.py` |
| P2 / P3 | `scripts/BW_P2_*`, `scripts/BW_P3_*` — see MASTER §2 |
| Frozen numbers | `rebuild/NUMBERS.csv` |

v1.1 planning rows live in `data/v11/bank.jsonl`. P2 false-state in `rvc11` is GSM-only so far; BW P2 is not ported yet.
