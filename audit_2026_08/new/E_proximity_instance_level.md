# E. Instance-level proximity vs VRI

Repeats §4.3 (proximity vs VRI, raw Pearson and residualized on per-problem canonical accuracy) with **`instance_contamination_score`** instead of template proximity.
VRI = mean(W1, W2, W4) − W3, per problem, 0/1 correctness.

Pool: frozen adversarial **n=61** (34 SP + 10 CC + 17 WIS). Paper text says n=64; that count is not recoverable as a unique-ID list from released files.

| model | n | instance r (p) | instance partial r (p) | template r (p) | instance bootstrap 95% CI |
|---|---:|---|---|---|---|
| Claude | 61 | 0.44 (0.0003) | 0.42 (0.0008) | -0.37 (0.0037) | [0.27, 0.59] |
| GPT-4o | 61 | 0.36 (0.0040) | 0.38 (0.0025) | 0.28 (0.0298) | [0.18, 0.55] |
| Llama | 61 | 0.11 (0.3815) | 0.12 (0.3410) | 0.27 (0.0371) | [-0.14, 0.41] |
| Gemini | 61 | 0.12 (0.3541) | 0.12 (0.3604) | -0.05 (0.7209) | [-0.23, 0.44] |
| o4-mini | 61 | -0.17 (0.1869) | -0.17 (0.1869) | -0.05 (0.7302) | [-0.37, 0.10] |

Paper §4.3 headlines (claimed template, n=64): Claude r=+0.44, GPT-4o r=+0.37, Llama/Gemini ~0.12, o4-mini r=−0.094; partial Claude +0.41 / GPT-4o +0.39.
Those published r values match **instance** scores on the frozen 61, not template scores. Template r on the same 61 is much weaker (see `template_pearson_r` in the CSV). The figure script `contam_vri_pearson()` already correlated `instance_contamination_score` vs VRI.

**Flags:** n is 61 not 64. All five models computed. Instance scores are 0 for many problems (floor mass).
