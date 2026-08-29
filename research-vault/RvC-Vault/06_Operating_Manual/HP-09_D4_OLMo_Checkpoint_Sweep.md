# HP-09 — Developmental Sweep on OLMo Checkpoints + Dolma Exposure
addresses: [[D04_Developmental_Checkpoints]] H4 · phase: 2 · needs: 1 GPU (7B inference), storage for checkpoints, repo

PROMPT:
Goal: track when surface-invariant reasoning emerges over training, with exposure counted in the actual corpus.
Steps:
0. Feasibility check first: list available OLMo 3 7B intermediate checkpoints on HuggingFace (stage/step granularity) and OLMo 2 7B checkpoint spacing; pick ~10 checkpoints log-spaced over tokens-seen; record exact revisions. If OLMo 3 stage-1 granularity is too coarse, fall back to OLMo 2 7B + Pythia-6.9B ladder.
1. Harness: fixed 4-shot prompt (base models; deviation from paper's zero-shot scope is intentional and documented), T=0, family verifiers unchanged. Items: GSM 44 + ALGO 110 canonicals + W3 + W6 variants only (budget control).
2. Per checkpoint: canonical acc, R_W3, R_W6, and Commitment Depth (cd.py from HP-08) on a 30-item subsample.
3. Exposure: count template-string and solution-string occurrences in Dolma (use the released infini-gram index if available for the corpus; else targeted substring scan over the relevant Dolma subsets); per-item accumulated-exposure curve by checkpoint token count.
4. Analysis: emergence shape (gradual vs phase change) for invariance vs accuracy; lag between canonical-acc emergence and W3-retention emergence; per-item correlation of exposure with retention; overlay midtrain-stage boundaries from the OLMo 3 report.
Output: sweep CSVs, exposure counts CSV, 3 figures (developmental curves, CD trace, exposure scatter), D4_REPORT.md.
Validate: report items flooring at each checkpoint and condition retention on canonical-correct; pin every checkpoint revision hash; exposure counting method documented well enough to rerun.
