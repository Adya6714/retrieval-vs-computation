#!/usr/bin/env bash
# Content-gold mechanistic queue + Llama greedy↔rank pass/fail gate.
#
# Order is intentional (do not reorder):
#   1) Qwen-Instruct chat-direct content-gold
#   2) Llama-Instruct chat-direct content-gold
#   3) Qwen-base raw-qa content-gold
#   4) Llama ALGO greedy decode (do_sample=False)
#   5) Gate: greedy ~6% + high content-gold ranks = PASS; ranks≈1 = FAIL
#
# Usage:
#   bash scripts/runs/launch_mechanistic_contentgold.sh          # all + gate
#   bash scripts/runs/launch_mechanistic_contentgold.sh qwen-instr
#   bash scripts/runs/launch_mechanistic_contentgold.sh llama
#   bash scripts/runs/launch_mechanistic_contentgold.sh qwen-base
#   bash scripts/runs/launch_mechanistic_contentgold.sh gate     # greedy + gate only

set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs results/raw results/derived

COMMON=(
  --families gsm algo bw
  --variants canonical w6
  --gold-token-mode content
  --device cuda
)

run_one() {
  local tag="$1"; shift
  local log="logs/${tag}_$(date +%F_%H%M%S).log"
  echo "=== START $tag ===" | tee -a "$log"
  python3 scripts/run_mechanistic_sweep_7b.py "$@" 2>&1 | tee -a "$log"
  echo "=== DONE $tag ===" | tee -a "$log"
}

run_qwen_instr() {
  run_one mechanistic_qwen25_7b_instruct_chatdirect_contentgold \
    --model Qwen/Qwen2.5-7B-Instruct \
    --prompt-mode chat-direct \
    --dtype float16 \
    --output results/raw/mechanistic_sweep_qwen25_7b_instruct_chatdirect_contentgold.csv \
    "${COMMON[@]}"
}

run_llama() {
  # HF_TOKEN must be set + Llama license accepted
  run_one mechanistic_llama31_8b_instruct_chatdirect_contentgold \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --prompt-mode chat-direct \
    --dtype bfloat16 \
    --output results/raw/mechanistic_sweep_llama31_8b_instruct_chatdirect_contentgold.csv \
    "${COMMON[@]}"
}

run_qwen_base() {
  run_one mechanistic_qwen25_7b_base_rawqa_contentgold \
    --model Qwen/Qwen2.5-7B \
    --prompt-mode raw-qa \
    --dtype float16 \
    --output results/raw/mechanistic_sweep_qwen25_7b_base_rawqa_contentgold.csv \
    "${COMMON[@]}"
}

run_llama_greedy() {
  local log="logs/algo_llama31_greedy_canonical_$(date +%F_%H%M%S).log"
  echo "=== START llama ALGO forced-greedy ===" | tee -a "$log"
  python3 scripts/algo_llama_greedy_accuracy.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --output results/raw/ALGO_llama31_8b_greedy_canonical.csv \
    --resume \
    2>&1 | tee -a "$log"
  echo "=== DONE llama ALGO forced-greedy ===" | tee -a "$log"
}

run_gate() {
  local log="logs/mechanistic_contentgold_gate_$(date +%F_%H%M%S).log"
  echo "=== START contentgold gate ===" | tee -a "$log"
  python3 scripts/runs/mechanistic_contentgold_gate.py \
    --mech results/raw/mechanistic_sweep_llama31_8b_instruct_chatdirect_contentgold.csv \
    --greedy results/raw/ALGO_llama31_8b_greedy_canonical.csv \
    --report results/derived/mechanistic_contentgold_gate_report.md \
    2>&1 | tee -a "$log"
  echo "=== DONE contentgold gate ===" | tee -a "$log"
}

case "${1:-all}" in
  qwen-instr) run_qwen_instr ;;
  llama)      run_llama ;;
  qwen-base)  run_qwen_base ;;
  greedy)     run_llama_greedy ;;
  gate)
    run_llama_greedy
    run_gate
    ;;
  all)
    run_qwen_instr
    run_llama
    run_qwen_base
    run_llama_greedy
    run_gate
    ;;
  *)
    echo "usage: $0 [all|qwen-instr|llama|qwen-base|greedy|gate]" >&2
    exit 2
    ;;
esac
