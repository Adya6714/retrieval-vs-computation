# D12 Precision and compression invariance

status: Track T, T3 · execution: [[HP-18_Precision_Sweep]] · hypothesis: H9 · serves: CC-1, AC-1 and AC-2 precursors
evidence: [[P-00_Ingestion_Queue]] (LLM.int8 outlier features; KV-cache compression line; all quarantined)

**Claim tested.** Weight quantization degrades surface invariance (R_W3, R_W4, φ between canonical and transformed correctness) before it degrades canonical accuracy. Working title for the paper: "Same Score, Different Precision".

**Why plausible (hypothesis, not evidence).** F5 says robustness is a capacity distinct from accuracy. Quantization error concentrates in outlier channels and low-magnitude directions. If transfer to a new surface depends on weaker, less redundant features than solving the familiar canonical form, it should break first.

**Design.**
- Models: the three D13 models (Qwen2.5-1.5B Instruct, Coder-Instruct, Math-Instruct) so the precision axis crosses the training axis at no extra item cost.
- Precision arms on T4: FP16 reference; INT8 (bitsandbytes LLM.int8); NF4 (bitsandbytes 4-bit); GPTQ-Int4 and AWQ-Int4 (published pre-quantized Qwen checkpoints; verify availability per model).
- Per-band arm: round-to-nearest INT4 fake-quantization (group size 128) applied to one contiguous band of decoder blocks at a time (e.g. 4 bands of 7 blocks for a 28-block model), rest FP16. Output: an invariance-sensitivity map per band.
- Items: GSM + coin_change, variants canonical, W1, W3 (W3a and W3b once [[EF-07_W3_Construct_Audit]] option S lands), W4, W6.
- Run only on a stack configuration that D11 shows to be flip-stable.

**Metrics.** ΔAcc_can, ΔR_W3, ΔR_W4 vs FP16; per-item φ(FP16 correct, quantized correct) on canonical and on W3; item flip set overlap across precisions.

**Primary contrast (pre-register).** ΔR_W3 − (ΔAcc_can / Acc_can) with cluster bootstrap CI. H9 supported if the CI excludes 0 for ≥ 2 of 3 models at NF4 or INT4.

**FP8 and KV cache (deferred).** Native FP8 (E4M3/E5M2) needs Ada or Hopper (L4, 4090, H100). T4 only emulates it. With Colab Pro L4: FP8 weights and FP8 KV cache in vLLM; KV eviction (H2O, SnapKV, StreamingLLM) tested on Probe 2 Session B transcripts, which are the longest contexts and the ones that require state tracking.

**What this does not claim.** Nothing about kernel speed or FLOPs. D12 is a validity check for compression methods, not a compression method.

**Cost.** T4 hours; pre-quantized checkpoints avoid calibration runs.
