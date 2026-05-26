#!/usr/bin/env bash
# Orchestrate new-model sweeps with logging. Run from repo root.
set -euo pipefail
REPO="/Users/adya/Desktop/rvc"
cd "$REPO"
LOGDIR="$REPO/results/raw/new_model_sweep_logs"
mkdir -p "$LOGDIR"

GEMINI="google/gemini-2.5-flash"
O1MINI="openai/o1-mini"
TS="$(date -u +%Y%m%dT%H%M%SZ)"

run_step() {
  local name="$1"
  shift
  local log="$LOGDIR/${TS}_${name}.log"
  echo "===== START $name @ $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee "$log"
  "$@" 2>&1 | tee -a "$log"
  echo "===== END $name exit=$? @ $(date -u +%Y-%m-%dT%H:%M:%SZ) =====" | tee -a "$log"
}

# --- Gemini P1 BW ---
run_step gemini_bw_p1 python3 scripts/BW_P1_SCR_run_behavioral_sweep.py \
  --model "$GEMINI" \
  --family planning_suite \
  --question-bank-path data/problems/question_bank_bw.csv \
  --output results/raw/BW_P1_behavioral_gemini.csv \
  --resume

# --- Gemini P1 ALGO ---
run_step gemini_algo_p1 python3 scripts/ALGO_P1_SCR_run_behavioral_sweep.py \
  --bank data/problems/question_bank_algo.csv \
  --model "$GEMINI" \
  --output results/raw/ALGO_P1_behavioral_gemini.csv \
  --resume

# --- Gemini GSM P2 ---
run_step gemini_gsm_p2 python3 scripts/GSM_P2_SCR_run_probe2.py \
  --model "$GEMINI" \
  --output results/raw/GSM_P2_phase1_gemini.csv \
  --resume

# --- Gemini ALGO P2 phase1 (all canonical; phase2 filters adversarial) ---
run_step gemini_algo_p2_p1 python3 scripts/ALGO_P2_SCR_run_phase1.py \
  --bank data/problems/question_bank_algo.csv \
  --model "$GEMINI" \
  --output results/raw/ALGO_P2_phase1_gemini.csv \
  --resume

run_step gemini_algo_p2_p2 python3 scripts/ALGO_P2_SCR_run_phase2.py \
  --bank data/problems/question_bank_algo.csv \
  --model "$GEMINI" \
  --output results/raw/ALGO_P2_phase2_normal_gemini.csv \
  --instance-type adversarial \
  --resume

# --- o1-mini P1 ---
run_step o1mini_gsm_p1 python3 scripts/BW_P1_SCR_run_behavioral_sweep.py \
  --model "$O1MINI" \
  --family arithmetic_reasoning \
  --question-bank-path data/problems/question_bank_gsm.csv \
  --output results/raw/GSM_P1_behavioral_o1mini.csv \
  --resume

run_step o1mini_bw_p1 python3 scripts/BW_P1_SCR_run_behavioral_sweep.py \
  --model "$O1MINI" \
  --family planning_suite \
  --question-bank-path data/problems/question_bank_bw.csv \
  --output results/raw/BW_P1_behavioral_o1mini.csv \
  --resume

run_step o1mini_algo_p1 python3 scripts/ALGO_P1_SCR_run_behavioral_sweep.py \
  --bank data/problems/question_bank_algo.csv \
  --model "$O1MINI" \
  --output results/raw/ALGO_P1_behavioral_o1mini.csv \
  --resume

# --- o1-mini GSM P2 only ---
run_step o1mini_gsm_p2 python3 scripts/GSM_P2_SCR_run_probe2.py \
  --model "$O1MINI" \
  --output results/raw/GSM_P2_phase1_o1mini.csv \
  --resume

echo "ALL SWEEPS COMPLETE @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
