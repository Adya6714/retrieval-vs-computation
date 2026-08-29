# D10 — Pre-Deployment Structural Audit Suite (industry translation)
status: speculative until D1 calibration · lives with [[07_Broader_Impact/BI-02_Predeployment_Audit]]

**Vision.** Package Probe 1 (graded distance from D3), the direction probe (D6), and the sampling-consistency probe (D7) as an automatable pre-deployment test: feed a task distribution, get per-task fragility curves and a "structurally unsafe" flag for agents whose logic breaks under superficial change.

**Discipline.** Per program policy: this is flagged speculative. Two things must exist before pitching it as validated: (a) D1-lite calibration showing the signals track ground truth, (b) evidence the signals predict *deployment-relevant* failures (a small case study: does W3 fragility on a code-agent task predict failure on renamed-variable production code?). Until then, this note collects design requirements only: black-box-only mode (D7 core), cost budget per audited task (<100 calls), severity knob (D3 distance), and report format practitioners can read.

**What already exists elsewhere.** Perturbation benchmarks exist (RUPBench etc.); none couples graded distance + direction + consistency into a per-task audit with calibrated meaning. The calibration is the moat — which is another reason D1 leads.
