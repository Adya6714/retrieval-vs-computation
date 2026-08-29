# BM-07 — The Deep Strategies in Plain English (DS series)

Each DS note is a deeper, more novel move. Here they are without jargon, plus why each is a bigger deal than the basic perturb-and-observe approach. Full detail in 09_Deep_Strategies.

**DS-01 Latent-strategy model (mixture IRT).** Instead of hand-writing rules for "this counts as retrieval," fit a STATISTICAL model that discovers strategy groups from the data and gives each answer a probability of being retrieval vs computation. Why bigger: turns arbitrary thresholds into a principled model; your threshold-wobble weakness vanishes. Human testing solved this in 1990; nobody aimed it at LLMs. Cost $0. → [[DS-01_Latent_Strategy_Measurement_Model]]

**DS-02 Intrusion errors (read the wrong answers).** When the model fails a renamed problem, check WHAT it said. If it gives the ORIGINAL problem's answer, that's memory bleeding through — a fingerprint of retrieval no accuracy score captures. Why bigger: it's about error CONTENT, not error rate. Cost $0, uses data you already have. → [[DS-02_Intrusion_Error_Analysis]]

**DS-03 Look inside (representation similarity).** Compare the model's INTERNAL state on the original vs renamed problem. If the internals stay similar, it "sees through" the rename; if they scatter, it doesn't — and this works even when the final answer is wrong. Why bigger: measures understanding below the level of right/wrong answers, and gives a smooth curve. Needs open models, ~1–2 GPU-days. → [[DS-03_Representational_Invariance_RSA]]

**DS-04 Variety, not volume (exposure-variance law).** Everyone asks "how MANY times did the model see it." The deeper question from learning science: was it the SAME wording 1000 times, or 100 different wordings? Test whether VARIETY (not count) is what builds robustness. Why bigger: a positive result is a LAW of how to build training data ("diversify, don't just repeat or dedupe") that companies would act on. → [[DS-04_Exposure_Variance_Laws]]

**DS-05 Does it know it's remembering?** After you control what the model saw, ask it "have you seen this before?" and check if it can tell. Why bigger: this is about the model knowing the SOURCE of its answer (memory vs reasoning), which is different from knowing if it's correct — and it's a safety-relevant result if models CAN'T tell. Nearly free. → [[DS-05_Machine_Source_Monitoring]]

**DS-06 Predict, don't just describe.** Fit curves to how fragility grows with change; then PREDICT a new model's curve before measuring it. Why bigger: prediction is what separates a science from a list of observations. → [[DS-06_Fragility_Laws_Prediction]]

**DS-07 Whose fault is fragility?** Statistically split fragility into "this PROBLEM is fragile for everyone" vs "this MODEL is fragile." Why bigger: if it's mostly the problem, then fragility is a property of the text-world models learn from, not a model defect — a contrarian, important claim. Cost $0. → [[DS-07_Variance_Decomposition]]

**DS-08 Let the instrument choose its own next question.** Once DS-01 exists, the test can pick the single most informative perturbation per item, like a smart adaptive exam. Why bigger: makes a real-world audit ~10x cheaper. → [[DS-08_Adaptive_Diagnosis]]

**DS-09 Make it bigger, not just different.** Scale problem COMPLEXITY (longer chains, more pieces) as a controlled dial. Retrieval breaks fast; real computation degrades gracefully. Why bigger: connects to the major open question of length/compositional generalization, as a mechanism question. → [[DS-09_Compositional_Stress]]

**DS-10 Are the cracks shared across models?** Do the exact items that break model A also break model B? If yes, there's a common cause — shared training text. Why bigger: it's a claim about the whole AI ecosystem, not one model. Cost low. → [[DS-10_Cross_Model_Transfer_Attack]]

**DS-11 Run the arrow backwards.** Normally: known training -> predicted behavior. Instead: observed behavior -> INFER an unknown (closed) model's training-data structure. Why bigger: it's like reading a star's composition from its light — turns your tool into an instrument for studying models you can't open. Year-2 summit goal. → [[DS-11_Data_Attribution_Inverse]]

**DS-12 The whole thesis.** All of the above assembled: a validated instrument + calibration + laws + ecology + inverse inference = a measurement science for reasoning. → [[DS-12_Grand_Synthesis]]

## How to think about starting these
Order by cost and dependency: DS-02 and DS-01 now (free, and they teach you the most). DS-03 with your first GPU work. DS-04 as the flagship causal experiment after D1-lite. The rest stage behind those. See [[BM-04_Your_First_Week]].
