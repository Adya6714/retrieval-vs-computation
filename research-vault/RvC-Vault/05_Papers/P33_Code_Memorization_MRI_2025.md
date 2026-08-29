# P33 Memorize or Generalize? Evaluating LLM Code Generation with Code Rewriting (arXiv:2503.02296) — verified-source (abstract, 2026-07-07)
framing: Memorization Risk Index fires only when the model outputs similar code AND fails the perturbed task — capturing harmful memorization vs benign reuse.
measures: MRI (output similarity × perturbed-task failure).
granularity: per-instance, code domain.
findings: memorization does not grow with scale (sometimes shrinks); SFT raises memorization; PPO balances better.
bears_on: [[DS-02_Intrusion_Error_Analysis]] (the "similar output + fail" logic, ported from code-similarity to answer-intrusion for reasoning), [[D09_Robustness_Finetune_Transfer]] (their SFT/RL result).
