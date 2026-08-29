# HP-12 — Sampling-Consistency Pilot (kill-criterion gated)
addresses: [[D07_Sampling_Consistency_Probe]] · phase: 3 pilot · needs: small API budget (~1.2k calls)

PROMPT:
Goal: decide whether delta answer-entropy is a fourth orthogonal signal or redundant with CSS.
Steps:
1. Items: 30 ALGO problems = 10 universally-fragile (from the 26-item W3-collapse set), 10 robust (W3-retained by ≥4 models), 10 mid. Models: claude-sonnet-4, gpt-4o.
2. Per (item, variant∈{canonical, W3}): k=10 samples at T=0.7; extract final answers with the existing parser; compute normalized answer entropy + majority-answer accuracy.
3. Signals: ΔH = H(W3) − H(canonical); also H(canonical) alone.
4. Pre-registered decision rule (write before running): if |corr(ΔH, per-item CSS)| > 0.8 → redundant, drop the probe; if ΔH separates fragile from robust groups at p<.05 with |corr|≤0.8 → promote to full probe and add as an MTMM method column.
Output: raw samples CSV, D7_PILOT.md with the decision recorded either way.
Validate: decision rule committed to git before the first API call; parser failure rate reported (entropy over unparsed outputs is meaningless — exclude and count).
