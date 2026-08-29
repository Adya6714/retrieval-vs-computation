# P32 Memorization or Interpolation? Detecting LLM Memorization through Input Perturbation Analysis (arXiv:2505.03019) — verified-source (abstract, 2026-07-07)
framing: Perturbation Sensitivity Hypothesis — true memorization vs successful interpolation distinguished by sensitivity to input perturbation; black-box, unknown training data.
measures: perturbation-sensitivity score.
granularity: per-instance-ish.
uncontrolled: single perturbation signal; no execution/exposure probes; no wrong-answer content analysis; no ground-truth calibration.
bears_on: [[DS-02_Intrusion_Error_Analysis]] (nearest neighbor on the detection axis — differentiator is error CONTENT + calibration), [[D01_Controlled_Exposure_Validation]].
