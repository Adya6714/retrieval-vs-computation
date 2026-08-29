# HP-07 — Causal Patching: Canonical↔W3 on Llama-3.1-8B
addresses: [[D02_Causal_Patching]] H2/H5 · phase: 1 flagship · needs: 1× A100-40GB, TransformerLens or nnsight, repo

PROMPT:
Goal: interventional test of whether W3 rename fragility is entity-binding failure, and whether injected wrong state fails to derail because the answer is already fixed.
Steps:
1. Pair selection: from GSM and ALGO raws, take (item, Llama) pairs with canonical correct ∧ W3 wrong (dissociating) and canonical correct ∧ W3 correct (control), ≥20 each. Tokenization pre-check: for each pair, verify canonical and nonce entity spans align to comparable token positions; drop or realign pairs that don't (report attrition).
2. Metric: logit difference for the gold answer's first token vs the model's wrong-answer first token, at the answer position. Report answer-flip rate secondarily.
3. Interventions (both directions each, per Zhang & Nanda ICLR 2024): (a) patch canonical-run residuals into the W3 run at renamed-entity token positions, layer sweep 0–31; (b) reverse; (c) late-layer band patch (last 4 layers) at the answer position; (d) H5 arm: Probe-2B-style prompts with injected wrong intermediate state — patch pre-injection residuals into post-injection runs at the injection span; if gold logit-diff is unchanged, the answer direction predates injection.
4. Analysis: restoration curves by layer for dissociating vs control pairs; interaction = the headline. Commitment Depth overlay (HP-08): does restoration peak near CD±2?
Output: patching scripts, results parquet, 3 figures (layer-restoration curves, position heatmap, H5 bar), D2_REPORT.md.
Validate: control pairs must show near-zero patch effects (sanity); report both patch directions and both corruption framings; any single-site claim must survive a head-set/path-patching check or be reported as diffuse.
