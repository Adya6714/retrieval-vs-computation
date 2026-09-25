# HP-19 Distillation contrast (T4)
addresses: [[D14_Distillation_Invariance]] · phase: Track T · needs: Colab T4

PROMPT:
Goal: compare a distilled reasoning student with its base and with an SFT sibling on the surface battery, plus intrusion coding.
Rules: same as HP-16.
Steps:
1. Models: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, Qwen/Qwen2.5-Math-1.5B (base; use a plain completion prompt identical in content), Qwen/Qwen2.5-Math-1.5B-Instruct.
2. Before running, open the DeepSeek-R1 report and confirm the stated base model of the 1.5B distill; write the citation and page in the report header. If the base differs, stop and report.
3. Items: GSM + coin_change, all variants. max_new_tokens = 4096 for the distilled model, recorded; log truncation per row. Parse the final answer after the think block.
4. Run the DS-02 intrusion coding script (`scripts/consolidate/c1_intrusion_error_analysis.py`) on W3 errors for each arm.
Output: `results/raw/T4_P1_{model_slug}.csv`, `results/derived/T4_metrics.csv`, `results/derived/T4_intrusion_rates.csv`, `docs/trackT/T4_REPORT.md` (tables only).
Validate: truncation rate per arm reported; any arm with >10% truncation flagged.
