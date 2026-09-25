# C1 — ALGO `problem_id` width report (no rename applied)

Date: 2026-09-25

## What exists

| Source | Unique IDs | 2-digit (`CC_01`) | 3-digit (`CC_031`) |
|---|---:|---:|---:|
| `data/problems/question_bank_algo.csv` | 110 | 10 (`CC_01`…`CC_10`) | 100 |
| `results/derived/ALGO_P2_per_instance_cci.csv` | 61 | 10 | 51 |
| P1 rescored ALGO files | 110 | 10 | 100 |

`CC_01` and `CC_001` do **not** both exist. Normalising `CC_01` → `CC_001` would not collide with any current bank ID.

## Join sites (ALGO `problem_id`)

Scripts that merge/join on ALGO `problem_id` (non-exhaustive but covers consolidate + P2):

- `scripts/consolidate/algo_mech_behavior_link.py` — inner merge on `(model, problem_id)`
- `scripts/consolidate/algo_canonical_w6_matched_compare.py` — index by `problem_id`
- `scripts/consolidate/cluster_bootstrap_algo.py` — merge bank subtype onto P1
- `scripts/consolidate/qwen_algo_w3_offline_score.py` — `isin(adversarial ids)`
- P2 CCI/TEP derived tables join to bank/P1 by raw string equality

## Collision / silent-drop behaviour

| Join mode | Result on current data |
|---|---|
| Raw string (`CC_01` ↔ `CC_01`) | **61/61** CCI ids hit bank canonical |
| Pad only CCI to 3 digits vs bank raw | **51/61** — drops `CC_001`…`CC_010` (silent) |
| Pad only bank vs CCI raw | **51/61** — same 10 CC ids drop |
| Pad both sides | **61/61** |

So mixed widths do not break today’s raw-string joins, but any one-sided `zfill(3)` / `f"{n:03d}"` silently drops the ten 2-digit CC problems from CCI analyses.

## Proposal (do not apply until approved)

**Canonical form:** `{CC|SP|WIS}_{NNN}` with zero-padded 3-digit numeric suffix (`CC_01` → `CC_001`).

Suggested follow-up (separate commit, after approval):

1. Rewrite bank + all raw/derived ALGO `problem_id` strings via a single `normalize_algo_id()` helper.
2. Add `tests/test_algo_id_joins.py` asserting P2 CCI ids ⊆ bank ids under the helper.
3. Ban new 2-digit writes in `stage1_generate_algo.py`.
