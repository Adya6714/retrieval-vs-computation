# P06 Turpin et al. 2023 (Unfaithful CoT) — verified-source
framing: CoT explanations can misrepresent the causes of predictions.
perturbs: biasing features in-context.
measures: behavioral; explanation-vs-cause mismatch.
granularity: per-item within-trace.
uncontrolled: within-context only — model retains its stated reasoning in context.
bears_on: [[EF-02_Probe2_Plan_Execution]] (our CCI removes the in-context declaration entirely).
practitioner_translation: widely cited in eval practice ("don't trust CoT"); qualitative.
