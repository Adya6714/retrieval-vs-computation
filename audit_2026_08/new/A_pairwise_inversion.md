# A. All-pairs rename inversion

Frozen adversarial pool: **34 SP + 10 CC + 17 WIS = 61** problems (paper §4.3 says n=64; released frozen list is 61).
Fisher exact is the 2×2 of (W3-correct, W3-wrong) counts for model A vs model B. Bootstrap 95% CI is a paired resample of `problem_id` (10,000) on accuracy difference (A−B).

**Rows:** 60 (3 subtypes × 10 pairs × 2 definitions).
**Canonically-matched pairs with two-sided Fisher p<0.05:** 8.

| subtype | A | B | n | A W3 | B W3 | p | acc diff [CI] |
|---|---|---|---:|---:|---:|---:|---|
| SP | Claude | o4-mini | 22 | 0 | 20 | 2.62e-10 | -0.909 [-1.000, -0.773] |
| SP | Gemini | o4-mini | 23 | 9 | 20 | 0.0018 | -0.478 [-0.696, -0.261] |
| CC | GPT-4o | o4-mini | 6 | 0 | 6 | 0.0022 | -1.000 [-1.000, -1.000] |
| SP | Claude | Gemini | 18 | 0 | 8 | 0.0029 | -0.444 [-0.667, -0.222] |
| SP | GPT-4o | o4-mini | 14 | 7 | 14 | 0.0058 | -0.500 [-0.786, -0.214] |
| CC | GPT-4o | Gemini | 4 | 0 | 4 | 0.0286 | -1.000 [-1.000, -1.000] |
| SP | Claude | GPT-4o | 11 | 0 | 5 | 0.0351 | -0.455 [-0.727, -0.182] |
| CC | Claude | GPT-4o | 5 | 4 | 0 | 0.0476 | 0.800 [0.400, 1.000] |

Claude vs GPT-4o (paper’s reported inversion pair):
- paired: n=34, Claude W3=0, GPT-4o W3=9, p=0.0021, diff=-0.265 [-0.412, -0.118]
- canonically-matched: n=11, Claude W3=0, GPT-4o W3=5, p=0.0351, diff=-0.455 [-0.727, -0.182]

**Flags:** none — all 10 pairs × 3 subtypes × 2 definitions computed from ALGO P1 `verified` after dropping `mock` and `ERROR:` rows. Pairwise n is the intersection of problems both models actually logged.
