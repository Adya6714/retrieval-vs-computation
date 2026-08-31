# D. Declared-vs-executed algorithm agreement

Phase 1: `stated_algorithm`. Phase 2A: `reasoning_type == algorithm_invocation` on uninjected steps (`ALGO_P2_phase2_normal*.csv`).
Match = the invocation step names an algorithm in the same family as the Phase-1 declaration (Dijkstra / DP / greedy / …).
2×2: agreement × Phase-2A `final_answer_correct` (last step), Fisher two-sided.

| model | n | any invocation | match | agreement rate | 2×2 (agree✓, agree✗, dis✓, dis✗) | Fisher p |
|---|---:|---:|---:|---:|---|---:|
| Claude | 110 | 6 | 5 | 0.045 | 1/4/54/51 | 0.3634 |
| GPT-4o | 110 | 0 | 0 | 0.000 | 0/0/55/55 | 1.0000 |
| Llama | 110 | 4 | 2 | 0.018 | 0/2/24/84 | 1.0000 |
| Gemini | 110 | 3 | 3 | 0.027 | 0/3/34/73 | 0.5510 |
| o4-mini | 0 | 0 | 0 | NA | 0/0/0/0 | NA |

**Flags:** o4-mini agreement **not computable** — No ALGO Phase-1 declaration file for this model. There is no `ALGO_P2_phase1_o1mini.csv`. o4-mini *does* have Phase-2A steps in `ALGO_P2_phase2_normal.csv` (including a few `algorithm_invocation` rows), but nothing to match against.
GPT-4o has Phase-1 declarations but **zero** `algorithm_invocation` steps in Phase 2A, so agreement rate is 0 by construction.
Invocation is rare overall (paper Table 8 / appendix cases); most sessions never name an algorithm at execution time.
