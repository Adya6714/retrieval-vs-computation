# BI-02 Pre-Deployment Structural Audit (flag: speculative)

Product shape (design requirements only, per [[D10_Structural_Audit_Suite]]): input = a task distribution from the customer; output = per-task fragility curve (D3 severity knob), direction-gap score (D6), sampling-consistency delta (D7, black-box-only mode), and a calibrated retrieval-risk flag (post-D1 thresholds). Budget target <100 calls per audited task.

Missing before external claims: D1 calibration; one case study linking audit signals to a real deployment failure mode (candidate: renamed-variable robustness of a code agent). 

Position vs existing: perturbation benchmarks exist ([[P03_RUPBench_2024]] etc.); an audit with *calibrated meaning* per instance does not. Calibration is the moat.
