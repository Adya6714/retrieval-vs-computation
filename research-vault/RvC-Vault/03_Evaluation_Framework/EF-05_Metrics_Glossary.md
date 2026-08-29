# EF-05 Metrics Glossary

| Metric | Definition | Notes |
|---|---|---|
| VAR(v) | mean(behavioral_correct) per variant, model | probe1_per_model_variant.csv |
| R_W3 | Acc_W3 / Acc_can | undefined at Acc_can=0 |
| CSS | fraction of {W1,W2,W3,W4,W6} matching canonical, per problem | W5 excluded |
| VRI | mean(Acc_W1,W2,W4) − Acc_W3 (paper App. C) | repo css.py uses mean(W2,W4) − W3 — reconcile (HP-04) |
| RCS | W5 correctness | answer changes under W5 |
| CCI | (1/k) sum 1[|v1−v2|<=0.01]; k=0 zero-imputed in paired tests | fingerprint, not score |
| TEP | fraction of post-injection steps differing from uninjected run | high propagation, answers survive |
| Compliance | compliant / partial / refusal / format-ignored at injection | Gemini format-ignores 100% |
| Proximity | Infini-gram overlap, template + instance scores in [0,1] | proxy for closed models |
| Convergence label | 3-signal agreement rule | EF-04; threshold-sensitive |
| Commitment Depth (proposed) | earliest layer where gold token enters top-k of unembedding projection | r≈+0.82 with final rank in existing 7B sweep; [[D08_Commitment_Depth]] |

Statistics stack: Wilson 95% CIs; paired Wilcoxon primary + t secondary; Pearson primary with Spearman check; Holm–Bonferroni on two pre-specified confirmatory families; 10k bootstrap. Upgrade queued in EF-06: pooled mixed-effects logistic model.
