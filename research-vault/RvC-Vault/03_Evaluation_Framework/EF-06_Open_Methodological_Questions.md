# EF-06 Open Methodological Questions

Ranked by how hard a hostile reviewer can hit them.

1. **Label validity has no external criterion.** Fix: D1-lite calibration then threshold freeze (EF-04). The program's #1 vulnerability and #1 opportunity.
2. **Mechanistic evidence is about the wrong manipulation.** Appendix F contrasts canonical vs W6; the headline is W3. Until D2 runs canonical-vs-W3 patching, the mechanistic appendix supports a neighboring claim, not the paper's claim.
3. **Exposure–difficulty confound (WIS).** Matched-difficulty WIS bank (HP-05) is the only clean discharge; residualized correlations are supporting, not sufficient.
4. **CCI format confound.** Prompt-matched declaration/execution ablation; report CCI with a parser-coverage covariate.
5. **Per-cell tests instead of a pooled model.** Fit mixed-effects logistic: correct ~ variant × model × family + (1 + variant | problem). Population estimates, proper pairing, replaces dozens of Fishers. Bonus: 2PL IRT with an exposure covariate attacks #3 by separating item difficulty from exposure. Analysis-only; fold into HP-04.
6. **One reasoning-trained model.** Add DeepSeek-R1 or Qwen3 at full coverage when API budget allows.
7. **Zero-shot-only scope.** Few-shot partially restores accuracy under template perturbation (GSM-Symbolic), so W3 numbers are zero-shot estimates, not capability bounds — keep saying so.
8. **Verifier/parse edge cases.** ALGO parse_status review queue exists; sample-audit 50 rows per family per model before next submission.
