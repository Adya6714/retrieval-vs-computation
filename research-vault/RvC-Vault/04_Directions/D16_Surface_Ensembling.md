# D16 Surface ensembling at test time

status: Track T, T6 · execution: [[HP-20_Surface_Ensembling]] · hypothesis: H12 · serves: practitioner translation ([[BI-04_Engineering_Relevance]])

**Claim tested.** Because fragility is model-item specific (F4), majority vote across answer-preserving surface renderings of the same problem beats sample-based self-consistency at matched call count.

**Free retrospective arm.** Paper I already holds per-variant answers for five API models (canonical, W1, W2, W3, W4 have fixed gold). A surface vote over those answers costs nothing to compute. Only the self-consistency comparison arm needs new calls.

**Design.**
- Surface vote: answers on {canonical, W1, W2, W3b, W4}; majority, ties broken by canonical.
- Self-consistency: 5 samples at T=0.7 on canonical; majority.
- Matched calls: 5 each.
- Open-weight replication on T4 with the D13 models (both arms, no API).

**Metrics.** Accuracy of each ensemble; gain over single canonical call; per-family breakdown.

**Prediction.** Surface vote ≥ self-consistency on ALGO and GSM; no gain on BW (shared failures on obfuscated items, F4).

**Kill criterion.** Fails on ≥ 2 of 3 models: report null; do not tune the vote rule.

**Cost.** Retrospective arm $0; open-weight arm T4 hours.
