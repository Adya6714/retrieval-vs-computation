# BX-03 — Memory-Science Probes (reading the signature of memory)

## Origin
Ebbinghaus (SAVINGS: relearning something is faster than first learning, even when you cannot recall it — proof of a latent trace). Tulving (RECOGNITION vs RECALL dissociation; ENCODING SPECIFICITY). Meyer & Schvaneveldt (semantic PRIMING). Underwood/Postman (proactive & retroactive INTERFERENCE — old memories corrupt new ones and vice versa). Johnson (SOURCE MONITORING: knowing WHERE a memory came from). These read the FINGERPRINT of memory rather than testing presence with a quiz.

## The deep principle
Memory leaves specific, measurable signatures: it is faster the second time (savings), it leaks into wrong answers (intrusions/interference), it is cued by surface overlap (priming), and its SOURCE can be tracked separately from its content. Each signature is a different, hard-to-fake channel of evidence — exactly what a convergent instrument needs.

## LLM translation (several are new and cheap)
- **Savings:** fine-tune a model to a fixed accuracy on SEEN vs matched UNSEEN problems; SEEN should reach criterion in fewer steps even after we degrade it. Latent-trace evidence distinct from accuracy. (Extends [[D01_Controlled_Exposure_Validation]].)
- **Intrusion / interference:** when W3 fails, does the CANONICAL answer intrude? That is proactive interference — a direct retrieval fingerprint. This is [[DS-02_Intrusion_Error_Analysis]], now correctly named as memory science.
- **Priming:** does prepending a surface-similar-but-irrelevant solved problem speed up or bias the target answer more for retrieval-class items? Priming magnitude as a per-item signal.
- **Recognition vs recall:** models often "recognize" (pick the right answer in multiple choice) far better than they "recall" (generate it). A large recognition-recall gap is a memory-vs-computation signature. Cheap, black-box.
- **Source monitoring:** ask, after solving, whether the model has seen this before, and score self-report against known exposure. This is [[DS-05_Machine_Source_Monitoring]] — the source-monitoring construct exactly.

## What it proves for the objective
Gives four MORE orthogonal, theory-grounded signals for the convergent instrument (savings, priming, recognition-recall gap, source-monitoring d′), each with a century of validation behind its interpretation. Generates [[DS-16_Memory_Signature_Suite]].

## Papers to be inspired by
- Xie et al. K&K ([[P17_Xie_KK_Memorization_2024]]) — fine-tune-and-perturb, closest existing analog to savings/controlled exposure.
- Semantic-entropy / uncertainty work — adjacent to recognition-recall but about correctness, not source.
- Membership-inference literature (Min-K%, [[P11_Shi_MinK_2024]]) — the "was it in training" question from the security side; memory-science framing reinterprets it as source monitoring.
