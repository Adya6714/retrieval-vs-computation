# P14 MATCHA (Jiang et al., 2025) — verified-source (positioned in CAISc Table 1)
framing: decoupling hypothesis on GSM8K via within-trace answer-conditioned perturbations, LLM judge.
perturbs: within-trace, answer-conditioned.
measures: robust-answer / fragile-logic decoupling.
granularity: per-instance-ish, judge-mediated.
uncontrolled: LLM-judge subjectivity; no exposure probe; no session isolation.
bears_on: [[EF-02_Probe2_Plan_Execution]] (nearest concurrent work; our determinism + isolation are the differentiators).
