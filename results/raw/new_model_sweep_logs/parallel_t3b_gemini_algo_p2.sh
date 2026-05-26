#!/usr/bin/env bash
set -eu
cd /Users/adya/Desktop/rvc
LOG=results/raw/new_model_sweep_logs/t3b_gemini_algo_p2.log
exec > >(tee -a "$LOG") 2>&1
GEMINI=google/gemini-2.5-flash
echo "=== T3b ALGO P2 phase2 NORMAL $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 scripts/ALGO_P2_SCR_run_phase2.py \
  --bank data/problems/question_bank_algo.csv \
  --condition normal \
  --instance-type adversarial \
  --models "$GEMINI" \
  --output results/raw/ALGO_P2_phase2_normal_gemini.csv \
  --resume
echo "=== T3b ALGO P2 phase2 INJECTED $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 scripts/ALGO_P2_SCR_run_phase2.py \
  --bank data/problems/question_bank_algo.csv \
  --condition injected \
  --instance-type adversarial \
  --models "$GEMINI" \
  --output results/raw/ALGO_P2_phase2_injected_gemini.csv \
  --resume
echo "=== T3b DONE exit=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
