# P11 Shi et al. ICLR 2024 (Detecting pretraining data, Min-K%) — verified-source
framing: membership inference from token-probability statistics.
perturbs: nothing; logprob-based.
measures: Min-K% score.
granularity: per-instance.
uncontrolled: needs logprobs (limited on closed APIs); verbatim-biased.
bears_on: [[EF-03_Probe3_Exposure]] — Min-K% on Llama/OLMo is a direct upgrade to Infini-gram for open models; add to D4 pipeline.
