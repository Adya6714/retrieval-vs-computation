# BM-01 — Glossary For Beginners (plain English)

**LLM** — large language model; predicts the next word; trained on huge text. GPT-4o, Claude, Llama are LLMs.
**Benchmark** — a fixed test with known answers used to score models. Tells you WHETHER a model scores, not WHY.
**Canonical (item)** — the original, unmodified problem. Everything is compared against it.
**Variant / perturbation** — a modified version of a problem that keeps the same correct answer (renamed names, reworded, reformatted). We use W1–W6.
**W3 (entity rename)** — our key variant: replace names with nonsense words ("John" → "Zorbnak"). If the answer breaks, the model was probably leaning on the specific words, i.e. memorizing.
**Probe** — a small experiment that reveals ONE thing about how the model works. We have three main probes.
**Retrieval vs computation** — retrieval = recognized this from training and recalled the answer; computation = actually reasoned it out. The whole program is about telling these apart.
**Exposure / contamination** — how much this problem (or ones like it) appeared in training data. High exposure = probably memorized.
**Mechanistic interpretability** — opening the model to look at its internal numbers (activations) instead of only its output. Like an MRI vs an interview.
**Activation / residual stream** — the internal vector of numbers flowing through the network as it processes text. What mechanistic methods read.
**Layer** — LLMs process text through stacked layers; early layers handle surface, later layers handle meaning (roughly). "Where in the layers" is often the interesting question.
**Logit / logit-lens** — logits are the model's raw scores for each possible next word. Logit-lens = projecting an internal layer to see which word it's "leaning toward" at that point.
**Patching (activation patching)** — copy an internal activation from one run into another run to test cause-and-effect: "if I paste the easy version's internals here, does the answer come back?"
**Fine-tuning / LoRA** — extra training on a small dataset to adapt a model. LoRA is a cheap fine-tuning method (trains small add-on weights). We use it to MAKE a model memorize known items on purpose (D1).
**Checkpoint** — a saved snapshot of a model partway through training. A series of checkpoints lets you watch skills develop (D4).
**Ground truth** — a case where you KNOW the true answer (e.g., you chose what the model was trained on), used to check your instrument. Calibration needs this.
**Calibration** — adjusting/checking your instrument against ground truth until it reports truth correctly. The program's #1 missing piece.
**Validity** — does the test measure what it claims? **Reliability** — would you get the same result on a rerun? Both come from psychometrics.
**Psychometrics** — the science of building tests that reliably and validly measure a hidden trait (IQ, etc.). We borrow its rulebook.
**IRT (Item Response Theory)** — separates "how hard is the item" from "how able is the taker." **Mixture IRT** — allows different hidden groups (e.g., different strategies) with different item behavior; membership inferred from responses. Basis of [[DS-01_Latent_Strategy_Measurement_Model]].
**MTMM (multitrait-multimethod)** — a validity check: do different methods aimed at the same trait agree (convergent) and not bleed into other traits (discriminant)?
**Intrusion error** — a wrong answer that is actually the RIGHT answer to a DIFFERENT (usually the original) problem. A fingerprint of memory interference. Basis of [[DS-02_Intrusion_Error_Analysis]].
**RSA / CKA** — methods to compare two sets of internal representations for similarity. Basis of [[DS-03_Representational_Invariance_RSA]].
**Dose-response / dose** — from pharmacology: vary the "dose" (here, amount of change or exposure) and watch the response curve. Basis of [[D03_Continuous_Transfer_Distance]] and [[DS-04_Exposure_Variance_Laws]].
**CoT (chain of thought)** — the model's step-by-step written reasoning. **Faithfulness** — whether that written reasoning reflects what the model actually did.
**API call** — one request to a model. Costs money per call. Our whole program so far ≈ 20,000 calls.
**GPU-day / A100 / H100** — GPU time for running open models locally; rented by the hour. A100/H100 are common data-center GPUs. "1 GPU-day" ≈ one card for ~24h.
**Cursor** — an AI coding assistant that runs code. In this lab it does the EXECUTION (numbers, tables, plots); it must not invent numbers.
**HP prompt** — a ready-made instruction block (in 06_Operating_Manual) you paste into Cursor to run one task safely.
**Pre-registration** — writing down your method and decision rules BEFORE running, so you can't fool yourself after seeing results.
