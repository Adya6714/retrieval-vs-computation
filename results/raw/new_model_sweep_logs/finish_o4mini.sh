#!/usr/bin/env bash
# Finish remaining o4-mini sweeps (resume retries ERROR rows only).
set -eu
cd <REPO_ROOT>
export OPENROUTER_MAX_TOKENS=4096
LOG=results/raw/new_model_sweep_logs/finish_o4mini.log
exec > >(tee -a "$LOG") 2>&1
MODEL=openai/o4-mini

run() {
  echo "=== $1 $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  shift
  "$@" || echo "WARN: $1 failed exit=$?"
}

# GSM P1 — fill ERROR rows (~11 left)
run "GSM P1" python3 scripts/BW_P1_SCR_run_behavioral_sweep.py \
  --model "$MODEL" \
  --family arithmetic_reasoning \
  --question-bank-path data/problems/question_bank_gsm.csv \
  --output results/raw/GSM_P1_behavioral_o1mini.csv \
  --resume

# BW P1 — mostly ERROR rows
run "BW P1" python3 scripts/BW_P1_SCR_run_behavioral_sweep.py \
  --model "$MODEL" \
  --family planning_suite \
  --question-bank-path data/problems/question_bank_bw.csv \
  --output results/raw/BW_P1_behavioral_o1mini.csv \
  --resume

# ALGO P1 — fill ERROR rows
run "ALGO P1" python3 scripts/ALGO_P1_SCR_run_behavioral_sweep.py \
  --bank data/problems/question_bank_algo.csv \
  --model "$MODEL" \
  --output results/raw/ALGO_P1_behavioral_o1mini.csv \
  --resume

# GSM P2
run "GSM P2" python3 scripts/GSM_P2_SCR_run_probe2.py \
  --model "$MODEL" \
  --output results/raw/GSM_P2_phase1_o1mini.csv \
  --resume

echo "=== FINISH_O4MINI DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
python3 results/paper/complete_new_models_pipeline.py
