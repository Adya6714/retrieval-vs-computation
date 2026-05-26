#!/usr/bin/env bash
set -eu
cd /Users/adya/Desktop/rvc
LOG=results/raw/new_model_sweep_logs/t3_gemini_p2.log
exec > >(tee -a "$LOG") 2>&1
echo "=== T3 Gemini GSM P2 start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 scripts/GSM_P2_SCR_run_probe2.py \
  --model google/gemini-2.5-flash \
  --output results/raw/GSM_P2_phase1_gemini.csv \
  --resume
echo "=== T3 Gemini ALGO P2 phase1 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 scripts/ALGO_P2_SCR_run_phase1.py \
  --bank data/problems/question_bank_algo.csv \
  --model google/gemini-2.5-flash \
  --output results/raw/ALGO_P2_phase1_gemini.csv \
  --resume
echo "=== T3 Gemini ALGO P2 phase2 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 scripts/ALGO_P2_SCR_run_phase2.py \
  --bank data/problems/question_bank_algo.csv \
  --model google/gemini-2.5-flash \
  --output results/raw/ALGO_P2_phase2_normal_gemini.csv \
  --instance-type adversarial \
  --resume
echo "=== T3 DONE exit=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
