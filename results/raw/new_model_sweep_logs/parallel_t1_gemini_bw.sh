#!/usr/bin/env bash
set -eu
cd <REPO_ROOT>
LOG=results/raw/new_model_sweep_logs/t1_gemini_bw.log
exec > >(tee -a "$LOG") 2>&1
echo "=== T1 Gemini BW P1 start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 scripts/BW_P1_SCR_run_behavioral_sweep.py \
  --model google/gemini-2.5-flash \
  --family planning_suite \
  --question-bank-path data/problems/question_bank_bw.csv \
  --output results/raw/BW_P1_behavioral_gemini.csv \
  --resume
echo "=== T1 DONE exit=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
