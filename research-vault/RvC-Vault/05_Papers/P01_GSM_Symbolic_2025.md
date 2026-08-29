# P01 GSM-Symbolic (Mirzadeh et al., ICLR 2025) — status: verified-source (cited in our drafts)
framing: does GSM8K accuracy reflect reasoning or template familiarity?
perturbs: template variables (names, numbers), clause insertion (GSM-NoOp), on arithmetic only.
measures: behavioral accuracy; group-mean drops and variance across instantiations.
granularity: group-level.
uncontrolled: no execution or exposure probe; no per-instance labels; single domain.
bears_on: [[D01_Controlled_Exposure_Validation]] (few-shot restoration caveat), [[D03_Continuous_Transfer_Distance]] (their name/number split anticipates our W3/W6 asymmetry), [[EF-01_Probe1_Surface_Invariance]].
practitioner_translation: widely cited in industry commentary on "LLMs can't reason"; no validated audit tooling shipped — translation is rhetorical, not operational.
new_angle: their variance-across-instantiations statistic is a cheap add to our W6 reporting.
