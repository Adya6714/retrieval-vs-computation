# D02 — Causal Patching: Rename Fragility as Binding Failure
status: Tier 1 · execution: [[HP-07_D2_Patching_Llama]]

**Claims tested.** H2 (W3 collapse in retrieval-consistent items is mediated by canonical answer/binding representations that persist under rename) and H5 (injected wrong state fails to derail because the answer direction is already fixed).

**Core design.** Model: Llama-3.1-8B (in the behavioral pool — ties the loops the paper defers). Pairs: canonical/W3 items where behavior dissociates (canonical correct, W3 wrong), plus matched controls (both correct). Interventions:
- *Entity-position patching:* patch residual stream at renamed-entity token positions with canonical-run activations, layer by layer. If accuracy/logit-diff on the correct answer is restored → fragility is entity-binding failure, not lost computation. Direction B (canonical run + nonce activations) tests sufficiency of the disruption.
- *Late-layer patching* at the divergence band the Qwen pilot located (last 3–4 layers) to separate "commitment overwrite" from "input re-binding."
- *Injection-step patching (H5):* in Probe-2B style prompts, patch pre-injection residuals into post-injection runs; if final answers don't move, the answer direction predates the injection — mechanistically resolving the compliance-vs-correctness dissociation.

**Metrics.** Logit difference on constrained answer tokens (avoids multi-token generation headaches), answer-flip rate, and Commitment Depth shift ([[D08_Commitment_Depth]]). Follow patching best practices ([[P23_Zhang_Nanda_Patching_2024]]): both patch directions, noise vs counterfactual corruption both reported.

**Why Tier 1.** Repairs the manipulation mismatch (current mechanistic evidence is canonical-vs-W6, headline is W3 — [[EF-06_Open_Methodological_Questions]] #2) and converts "supportive" into "interventional." No API cost; one A100-class GPU.

**Frontier (2026-07-07).** Binding-ID and lookback/rebinding circuit literature ([[P21_Feng_Binding_2023]], [[P22_Prakash_Lookback_Rebinding]]) plus arithmetic causal mediation ([[P24_Stolfo_Arithmetic_CMA_2023]]) give mature methods; none has been aimed at rename-fragility on reasoning tasks or connected to a behavioral triangulation suite. Attribution-graph tooling (Anthropic circuit tracing, 2025; status abstract-only in [[P-00_Ingestion_Queue]]) is an optional depth extension.

**Risks.** Effects diffuse across heads → use path patching / head-set patching and report attenuation honestly. Multi-token answers → constrain to first numeric token or use rank metrics. Nonce tokens multi-token under Llama BPE → verify tokenization equivalence per item first (script step in HP-07).

evidence: [[P21_Feng_Binding_2023]] [[P22_Prakash_Lookback_Rebinding]] [[P23_Zhang_Nanda_Patching_2024]] [[P24_Stolfo_Arithmetic_CMA_2023]]
