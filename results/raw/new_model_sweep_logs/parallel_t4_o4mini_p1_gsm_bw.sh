#!/usr/bin/env bash
set -eu
cd <REPO_ROOT>
LOG=results/raw/new_model_sweep_logs/t4_o4mini_p1_gsm_bw.log
exec > >(tee -a "$LOG") 2>&1
MODEL=openai/o4-mini
# Reuse o1mini filenames for downstream scripts; model column will be openai/o4-mini
echo "=== T4 o4-mini GSM P1 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 scripts/BW_P1_SCR_run_behavioral_sweep.py \
  --model "$MODEL" \
  --family arithmetic_reasoning \
  --question-bank-path data/problems/question_bank_gsm.csv \
  --output results/raw/GSM_P1_behavioral_o1mini.csv \
  --resume
echo "=== T4 o4-mini BW P1 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 scripts/BW_P1_SCR_run_behavioral_sweep.py \
  --model "$MODEL" \
  --family planning_suite \
  --question-bank-path data/problems/question_bank_bw.csv \
  --output results/raw/BW_P1_behavioral_o1mini.csv \
  --resume
echo "=== T4 DONE exit=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
