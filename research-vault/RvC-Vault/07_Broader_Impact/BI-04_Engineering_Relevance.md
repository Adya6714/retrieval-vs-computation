# BI-04 Relevance to inference, compression, training and alignment engineering

status: practitioner translation, validation flags per row · directions: [[D11_Inference_Stack_Noise_Floor]] to [[D16_Surface_Ensembling]]

**One sentence.** The instrument is a validity check: it measures a property (surface invariance) that accuracy and perplexity do not, so it can detect damage from an engineering change that standard evals miss.

**What it is not.** Not a method for faster kernels, lower FLOPs, lower memory, or better throughput. Do not claim otherwise.

| Engineering area | What the instrument adds | Direction | Validation |
|---|---|---|---|
| Inference stack, kernels, determinism | Per-item verdict flip rate across batch size, attention backend, accumulation precision; a noise floor for any per-instance eval | D11 | untested |
| Quantization (INT8, INT4, NF4, GPTQ, AWQ; FP8 on Ada/Hopper) | Invariance loss at matched accuracy; per-band sensitivity map for mixed-precision allocation | D12 | untested |
| KV-cache compression, eviction, low-bit KV | State-tracking stress via Probe 2 execution transcripts | D12 (deferred, needs L4) | untested |
| Continued pretraining, domain transfer (general or multilingual to code/math) | Which surface the procedure is keyed to after transfer, not just accuracy | D13, D05 | untested |
| Distillation | Whether the student inherits invariance or only accuracy; intrusion signature | D14 | untested |
| RLVR / GRPO | Surface-diverse verifiable environments; diversity-vs-dose test of AC-3 | D15 | gated (G2) |
| Prompting and test-time compute | Surface ensembling; backward-planning prompts; avoid formal notation for non-code models | D16, HP-22, F2 | F2 shipped; others untested |
| Alignment, CoT monitoring | F6: false premises adopted and visibly propagated yet final answers recover, so the visible trace is not the whole causal path | F6 | shipped (behavioural) |
| Honesty, self-knowledge | Source monitoring: can a model report "recalled" vs "computed"? | CC-2 / DS-05 | gated (G1) |
| Eval integrity, safety cases | Contamination-robust check that a capability score reflects computation | Phase 5 audit | planned |

**Constraint shared by every row.** Needs open weights except F6 and prompting. Closed-model results cannot support precision, training, or exposure claims.
