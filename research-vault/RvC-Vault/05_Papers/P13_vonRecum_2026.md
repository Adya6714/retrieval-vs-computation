# P13 von Recum et al. ICLR 2026 (CoT interventions on reasoning models) — verified-source
framing: are reasoning LLMs robust to interventions on their own CoT? Seven intervention types at fixed timesteps.
perturbs: the model's own CoT tokens, within-trace.
measures: recovery rates by size/timing.
granularity: per-intervention.
uncontrolled: within-trace (model has its own reasoning in context).
bears_on: [[EF-02_Probe2_Plan_Execution]] — complementary: they measure self-correction with context; our 2B measures compliance with none. Cite as the pairing.
