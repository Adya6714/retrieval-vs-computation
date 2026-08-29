# BI-01 Training-Time Diagnostic (flag: speculative)

Claim under construction: Commitment Depth ([[D08_Commitment_Depth]]) computed on a held-out probe set during training could flag memorization-style learning (commitment moving early on exposed templates) before behavioral evals move.

Required before this is claimable: (1) HP-06 shows CD separates seen/unseen; (2) HP-09 shows CD traces are stable/interpretable across checkpoints; (3) cost analysis — logit-lens over a probe set is cheap (one forward pass per item per checkpoint), which is the selling point vs full eval sweeps.

Audience if validated: pretraining and finetuning teams; the pitch is a cheap dashboard scalar, not a new eval. Do not name specific companies in drafts.
