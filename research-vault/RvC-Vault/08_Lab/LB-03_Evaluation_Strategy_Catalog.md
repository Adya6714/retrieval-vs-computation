# LB-03 — Evaluation Strategy Catalog, Requirements, Costs

Cost assumptions (state them, verify quarterly): API calls average ~2K tokens round trip at T=0 CoT; marketplace GPU rates early-2026 ≈ A100-80GB $1.3–2.5/hr, H100 $2–3.5/hr, spot lower; your full three-probe program to date ≈ 20K calls = the reference "program unit." All figures are ranges, not quotes.

**S1. Surface-perturbation invariance (P1 — built).** Needs: variant generators + family verifier + parser per task family; T=0 harness. Cost: new family at 60 items × 7 variants × 5 models ≈ 2.1K calls ≈ $5–60 by model tier. Validity threats: tokenization, difficulty drift (verifier + cross-tokenizer contrast mitigate).

**S2. Graded-distance ladders (D3).** Adds: pre-registered distance metric, ladder generator, per-rung verifier recheck. Cost: 40 items × 10 rungs × 5 models ≈ 2K calls + generation ≈ $10–80. Threats: metric validity (pre-register), referent ambiguity at partial rename (verifier).

**S3. Cross-session process probes (P2 — built).** Adds: session-isolation harness, step parsers, injection library. Cost ≈ 2–3× S1 per family (multi-step). Threats: format confound (prompt-matched ablation), parser coverage (report as covariate).

**S4. Exposure probes (P3 — built + upgrades).** Infini-gram public API ≈ free. Min-K%/logprob membership: open models only, 1 GPU-day ≈ $30–60. Dolma counting for OLMo: free if the public index covers it, else targeted scans (storage ~1–2TB if local — avoid; stream). Threats: proxy-corpus mismatch (the reason D4 exists), paraphrase blindness (Ruis).

**S5. Ground-truth calibration (D1 family).** D1-lite LoRA on 8B: 1× A100-80GB, ~1 GPU-day ≈ **$30–60**. D1-mid continued-pretrain 1B on 5–10B tokens: ~100–250 A100-hrs ≈ **$150–600**. D1-full from-scratch 160M–1B on 10–30B tokens: **$1K–5K** — only after D1-mid shows dose-response. Threats: fine-tune≠pretraining objection (present as calibration lower bound), catastrophic forgetting (monitor unseen accuracy).

**S6. Causal patching + Commitment Depth (D2/D8).** 1× A100-80GB (48GB workable), TransformerLens/nnsight (free), 2–5 GPU-days ≈ **$60–300** total. Threats: diffuse effects (path/head-set patching), multi-token answers (logit-diff on first token), single-model scope (replicate on a second open model +$50–100).

**S7. Developmental checkpoint sweeps (D4).** ~10 checkpoints × ~500 item-variants, 7B inference: 2–4 GPU-days ≈ **$60–200**; checkpoint downloads ~150–200GB disk. Threats: early-checkpoint floors (retention ratios), harness drift (freeze the few-shot harness).

**S8. Sampling-consistency probes (D7).** k=10 at T≈0.7 on subsets: pilot ≈ 1.2K calls ≈ **$3–30**. Threats: parser failures dominate entropy (exclude+count), redundancy with CSS (pre-registered kill rule).

**S9. Psychometric analyses (MTMM, mixed-effects, IRT, reliability).** Compute ≈ **$0** (statsmodels/lme4/py-irt on existing CSVs). Requirements: long-format hygiene (Phase 0). Highest insight-per-dollar in the catalog; also the least occupied (C7). Reliability study (statement #5 in LB-02): 3 reruns of P1 on one family ≈ 3× S1 cost — do once, cite forever.

**S10. Cross-lingual points (D5).** LLM translation + back-translation checks ≈ $10–40; optional human spot-check via one bilingual friend ≈ free–$100. Threats: tokenizer confound (the 2×2 design is the mitigation).

**S11. Direction probes (D6).** Fast Downward (free) for plan verification both directions; API ≈ S1-scale subset ≈ $10–50.

**S12. Longitudinal model tracking (cheap moat).** Freeze a 100-item cross-family probe set; rerun on every major model release; publish the time series. ≈ **$20–60 per release**, compounding into a dataset nobody else maintains. Start now; costs nothing to start.

(Optional, far future) **S13. Human-comparison studies.** Prolific ≈ $12–15/participant-hour; 100-participant study ≈ **$2–4K** + ethics review. Only if a claim strictly requires human baselines.

## Budget envelope
Phase 0 ≈ $50–150 (HP-01 gap-fill + analyses). Phase 1 flagship ≈ $100–450 (S5-lite + S6). Phase 2 ≈ $60–200 (S7). Phase 3 ≈ $50–250 (S2+S8+S10+S11). **Year-one total ≈ $300–1,000 at careful tiering; <$2K with generous frontier-model usage.** Excludes S5-full and S13. Track actuals in a `costs.csv` in the repo; update this note quarterly.
