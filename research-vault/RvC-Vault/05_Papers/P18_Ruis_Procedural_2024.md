# P18 Procedural Knowledge in Pretraining Drives Reasoning (Ruis et al., ICLR 2025, arXiv:2411.12580) — verified-source (abstract+blog, 2026-07-07)
framing: what pretraining data do models rely on when reasoning? EK-FAC influence functions over 5M docs / 2.5B tokens for 7B & 35B models.
perturbs: nothing; attribution.
measures: per-document influence on reasoning vs factual queries.
granularity: per-document × per-query.
key finding: reasoning queries draw on shared *procedural* documents (formulae, code demonstrating methods); answers to reasoning queries rarely appear in top-influential docs — unlike factual queries.
uncontrolled: correlational attribution; 2.5B-token window; two models one family.
bears_on: [[D01_Controlled_Exposure_Validation]] — dictates that D1-mid must manipulate *procedural-document* exposure and *answer-string* exposure as separate arms; [[EF-03_Probe3_Exposure]] — explains why n-gram proximity (verbatim-biased) can miss the exposure that matters; [[D04_Developmental_Checkpoints]].
practitioner_translation: "curate procedural data to boost reasoning" — plausible, not validated as guidance.
