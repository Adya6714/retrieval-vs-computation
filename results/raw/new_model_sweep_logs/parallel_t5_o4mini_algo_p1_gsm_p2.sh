#!/usr/bin/env bash
set -eu
cd <REPO_ROOT>
LOG=results/raw/new_model_sweep_logs/t5_o4mini_algo_p1_gsm_p2.log
exec > >(tee -a "$LOG") 2>&1
MODEL=openai/o4-mini
echo "=== T5 o4-mini ALGO P1 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 scripts/ALGO_P1_SCR_run_behavioral_sweep.py \
  --bank data/problems/question_bank_algo.csv \
  --model "$MODEL" \
  --output results/raw/ALGO_P1_behavioral_o1mini.csv \
  --resume
echo "=== T5 o4-mini GSM P2 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 scripts/GSM_P2_SCR_run_probe2.py \
  --model "$MODEL" \
  --output results/raw/GSM_P2_phase1_o1mini.csv \
  --resume
echo "=== T5 DONE exit=$? $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
