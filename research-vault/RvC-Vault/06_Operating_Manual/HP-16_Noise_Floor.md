# HP-16 Inference-stack noise floor (T1)
addresses: [[D11_Inference_Stack_Noise_Floor]] · phase: Track T · needs: Colab T4, HF token not required for Qwen

PROMPT:
Goal: measure how often per-item verdicts change when only the inference stack changes, at greedy decoding.
Rules: raw outputs are truth (append-only, `--resume`); do not interpret results; log $0 to costs.csv with the Colab session note.
Steps:
1. Create `scripts/trackT/T1_noise_floor.py` with args `--model`, `--family {GSM,ALGO}`, `--configs all`, `--out results/raw/T1_noise_floor_{model_slug}.csv`, `--resume`, `--dry-run` (MockClient).
2. Items: every row of `data/problems/question_bank_gsm.csv` and the `coin_change` rows of `question_bank_algo.csv`. Use the same prompt template as the API sweeps (read it from the existing P1 sweep script; do not write a new one).
3. Configurations (full factorial, 12): batch_size ∈ {1, 8, 16} with left padding and pad = eos; attn_implementation ∈ {"eager", "sdpa"}; torch_dtype ∈ {float16, float32}. Greedy (`do_sample=False`), max_new_tokens fixed per family and recorded.
4. For each item × config write: problem_id, variant_type, model, batch_size, attn_impl, dtype, response_sha16, raw_response, parsed_answer, verified (call the family verifier from `probes/`), gen_tokens, truncated.
5. Derived script `scripts/trackT/T1_noise_floor_metrics.py` → `results/derived/T1_noise_floor_metrics.csv`: per model × variant: n_items, output_divergence_rate vs reference (batch 1, eager, float32), verdict_flip_rate, flip_rate_given_canonical_correct, list of flipped item IDs.
6. Models: Qwen/Qwen2.5-1.5B-Instruct first; then Qwen/Qwen2.5-0.5B-Instruct.
Output: raw CSVs, derived CSV, `docs/trackT/T1_REPORT.md` containing tables only (no interpretation).
Validate: reference config rerun twice must be hash-identical; if not, stop and report. Gold-in-gold-out check on verifiers before any model call. Record GPU name, CUDA, torch and transformers versions in the report header.
