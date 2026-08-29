# P09 Razeghi et al. 2022 (Pretraining term frequencies) — verified-source
framing: does term frequency in pretraining predict few-shot numerical accuracy? (yes)
perturbs: nothing; observational correlation on GPT-J with known corpus.
measures: accuracy vs frequency of operands.
granularity: per-instance frequency, group-level accuracy curves.
uncontrolled: frequency ≠ causal exposure; difficulty covaries.
bears_on: [[EF-03_Probe3_Exposure]] (the founding observation), [[D04_Developmental_Checkpoints]] (their method + checkpoints = our D4).
practitioner_translation: informal ("models are better on common numbers"); no tooling.
