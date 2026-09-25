# THE PLAN, Amendment A1 (2026-09-25): open-weight Track T while Phase 1 is GPU-blocked

status: proposed · amends: [[THE_PLAN]] v1.0 · recorded in: [[01_Program_State]] E · decision: [[02_Decision_Memo]] addendum 2026-09-25
THE_PLAN still wins every conflict. This amendment adds work; it removes no gate and moves no claim.

## Why an amendment
- Phase 1 needs 1× A100-80GB and API credits. Neither is available.
- Available compute: free Colab T4 (16 GB, FP16, Turing, no native FP8 or BF16).
- The behavioural layer (Phase 0) is shipped. Its main external weakness is that every model is closed-weight, so no result can be tied to precision, training stage, or data.
- Track T runs the existing instrument on small open-weight models under controlled interventions. It needs no gate because it makes no exposure claim and issues no per-instance labels.

## Scope rules for Track T
- Allowed: probe-level results (Acc, retention, φ, flip rates, CCI) on open weights under controlled interventions.
- Not allowed: per-instance retrieval/computation labels, exposure claims, any claim that belongs to G1 or later.
- Floor rule unchanged: retention reported only where Acc_can ≥ .30. Small models clear it on GSM and coin_change; not on BW. Track T uses GSM + ALGO coin_change (+ shortest_path where the floor is cleared).
- Pre-register each direction's primary contrast in `PREREGISTRATION.md` (new section "Track T") before its first run.

## Track T directions (all T4-feasible)
| ID | Direction | Hypothesis | Serves | Cost |
|---|---|---|---|---|
| T0 | [[EF-07_W3_Construct_Audit]] option S via [[HP-23_W3a_Nonce_Bank]]: W3a nonce vs W3b isomorph | H13 | CC-1, H2, Paper I fix | < $10 API or T4 |
| T1 | [[D11_Inference_Stack_Noise_Floor]] | H8 | instrument validity, all per-instance claims | T4 hours |
| T2 | [[D13_Continued_Pretraining_Transfer]] (Qwen2.5 base vs Coder vs Math) | H10 | CC-1, AC-3 precursor | T4 hours |
| T3 | [[D12_Precision_Compression_Invariance]] (FP16, INT8, NF4, GPTQ, AWQ; per-band) | H9 | CC-1, AC-1 localisation precursor | T4 hours |
| T4 | [[D14_Distillation_Invariance]] (R1-Distill-Qwen-1.5B vs Qwen2.5-Math-1.5B) | H11 | CC-1, AC-3 precursor | T4 hours |
| T5 | [[D06_Direction_Asymmetry]] backward-prompting discriminator (append) | D06 | F3 mechanism | < 100 calls |
| T6 | [[D16_Surface_Ensembling]] | H12 | practitioner result | T4 hours |
| T7 | [[D17_Commitment_Depth_T4]] | H14 | AC-1 precursor | T4 hours |
| T8 | [[D05_Cross_Linguistic]] W7 via [[HP-25_W7_Language_Transform]] | H15 | CC-1, GlobalSouthAI angle | T4 hours or small API |

Phase 3 addition (gated behind G2, not Track T): [[D15_RLVR_Surface_Diversity]] tests H7b, the RL form of AC-3. A plumbing-only dry run on Qwen2.5-0.5B is allowed; no result from it is reported.

## New hypotheses (registered here, status in [[01_Program_State]] C)
- H8 Inference-stack nondeterminism (batch size, attention backend, accumulation precision) flips a measurable fraction of per-item verdicts at T=0; per-instance labels are only meaningful above that floor.
- H9 Precision reduction degrades surface invariance (R_W3, φ) before it degrades canonical accuracy.
- H10 Code-specialised continued pretraining removes or inverts the W4 formal-notation penalty without changing W3 cost.
- H11 Distillation transfers canonical accuracy more fully than it transfers invariance.
- H12 Majority vote across answer-preserving surfaces beats sample self-consistency at matched call count, because fragility is model-item specific (F4).
- H13 Cost of cover-story isomorph (W3b) exceeds cost of nonce rename (W3a); if W3a is cheap, F1 is a domain-transfer effect, not an entity-binding effect.
- H14 Commitment depth (first layer where the gold token is top-1) shifts later under W3 than canonical, on models strong enough to solve renamed items.
- H15 A model computing in a language-agnostic latent shows R_W7 near 1; a model that does not shows W7 collapsing like W3.
- H7b (sub-hypothesis of H7) Under RLVR with matched steps, surface-diverse prompts yield higher held-out invariance than canonical-only prompts at equal canonical accuracy.

## Kill criteria (pre-registered)
- T1: if flip rate across stack configurations exceeds 10% of items for a model, all per-instance statements for that model are suspended until a batch-invariant configuration is fixed.
- T2 to T4: if no model in a contrast clears Acc_can ≥ .30 on ≥ 20 items, the contrast is reported as underpowered and not interpreted.
- T6: if surface ensembling does not beat self-consistency at matched calls on at least 2 of 3 models, report the null; do not tune the vote rule.

## Paper mapping
- Paper I resubmission: T0 (required fix), T5 (F3 mechanism), T1 (noise floor section).
- New short paper ("Same Score, Different Precision"): T1 + T2 + T3 + T4. Target: an efficient-ML or evaluation workshop, then an E&D-track main-conference submission if effects hold.
- Paper II gains T3's per-band map as prior evidence for band-constrained repair (AC-2).
