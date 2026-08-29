# D07 — Sampling-Consistency Probe (new, proposed this pass)
status: pilot-gated · execution: [[HP-12_D7_Sampling_Pilot]]

**Idea.** A fourth behavioral signal, orthogonal-by-construction to the current T=0 suite: at T≈0.7, sample k=10 responses per (item, variant); measure answer-distribution entropy and trace diversity. Hypothesis: retrieval-like items show a sharp canonical basin (near-zero entropy) that *shatters* under W3 (high entropy / mode collapse to wrong answers), while computation-like items show similar moderate entropy across variants. The delta-entropy(canonical→W3) becomes a per-instance signal.

**Why bother.** (a) It is the only probe in the suite a practitioner can run on any black-box API without weights, verifiers, or corpora — the deployable core of [[D10_Structural_Audit_Suite]]. (b) It gives the MTMM matrix ([[EF-04_Convergence_Labels_MTMM]]) a genuinely different method. Related but distinct: semantic-entropy hallucination detection (single-input uncertainty); this measures uncertainty *response to controlled perturbation*.

**Kill criterion (pre-registered).** Pilot on 30 ALGO items × 2 models. If delta-entropy correlates with CSS above |r|=0.8, it adds nothing — drop it. If it dissociates on the 26 universally-fragile items vs matched robust items, promote to full probe.

**Cost.** k× multiplier on a small subset; trivial engineering (temperature flag + entropy script).
