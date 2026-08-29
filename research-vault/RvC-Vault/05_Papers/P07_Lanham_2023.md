# P07 Lanham et al. 2023 (Measuring CoT Faithfulness) — verified-source
framing: intervene on CoT (truncate, corrupt, paraphrase) to measure answer dependence.
perturbs: the model's own CoT tokens within-trace.
measures: answer-change rates; larger models condition less on stated reasoning.
granularity: per-item within-trace.
uncontrolled: session isolation absent.
bears_on: [[EF-02_Probe2_Plan_Execution]]; the "answer already determined" reading of low faithfulness directly motivates H5 → [[D02_Causal_Patching]] injection-step patching.
practitioner_translation: informs eval methodology at labs.
