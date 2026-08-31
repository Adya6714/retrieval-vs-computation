# Solidify report

Three analyses from existing data. Script: `rebuild/solidify/compute_solidify.py`. Frozen filters unchanged.

## T1 — Complete-case triangulation

n_complete = **169** (of 440). Default labels on that subset: retrieval=8, computation=4, mixed=157, ambiguous=0.

| | n | retrieval | computation | mixed | ambiguous | confident-label rate |
|---|---:|---:|---:|---:|---:|---:|
| full panel (missing **or** disagree) | 440 | 8 | 4 | 157 | 271 | 0.0273 |
| complete-case (disagree only) | 169 | 8 | 4 | 157 | 0 | 0.0710 |

270-sweep **maximum** confident-label rate: **0.0523** (full) vs **0.1361** (complete-case).

Details: `T1_complete_case.md`.

## T2 — DS-02 intrusion

o4-mini ALGO: 5/43 = 0.116 [0.051, 0.245].
Fisher vs other models on ALGO:

- o4-mini vs Claude: OR=6.45, p=0.026
- o4-mini vs GPT-4o: OR=5.86, p=0.035
- o4-mini vs Llama: OR=∞, p=0.002
- o4-mini vs Gemini: OR=2.57, p=0.273

Details: `T2_intrusion.md`.

## T3 — DS-14 double dissociation

Canonically-matched opposite-sign subtype pairs: 1. Strict crossovers: **1**. Suggestive: **0**.

Details: `T3_double_dissociation.md`.

