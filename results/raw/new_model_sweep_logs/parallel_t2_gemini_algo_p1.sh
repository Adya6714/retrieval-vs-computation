#!/usr/bin/env bash
set -eu
cd <REPO_ROOT>
LOG=results/raw/new_model_sweep_logs/t2_gemini_algo_p1.log
exec > >(tee -a "$LOG") 2>&1
echo "=== T2 Gemini ALGO P1 start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 scripts/ALGO_P1_SCR_run_behavioral_sweep.py \
  --bank data/problems/question_bank_algo.csv \
  --model google/gemini-2.5-flash \
  --output results/raw/ALGO_P1_behavioral_gemini.csv \
  --resume
echo "=== T2 DONE exit=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
