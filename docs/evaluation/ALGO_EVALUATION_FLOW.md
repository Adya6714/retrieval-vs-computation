# Algorithmic (ALGO) Evaluation Flow

> **v1 archive.** For new runs use [`docs/v11/README.md`](../v11/README.md) and `rvc11/`.
> Full v1 pipeline map: [MASTER_EVALUATION_PIPELINES.md §4](./MASTER_EVALUATION_PIPELINES.md#4-algorithmic-algo).

## Quick pointers (v1)

| Piece | Path |
|-------|------|
| Bank | `data/problems/question_bank_algo.csv` (`coin_change`, `shortest_path`, `wis`) |
| P1 sweep | `scripts/ALGO_P1_SCR_run_behavioral_sweep.py` |
| P2 / P3 | `scripts/ALGO_P2_*`, `scripts/ALGO_P3_*` — see MASTER §4 |
| Frozen numbers | `rebuild/NUMBERS.csv` |

v1.1 ALGO rows live in `data/v11/bank.jsonl`. ALGO P2 is not ported into `rvc11` yet.
