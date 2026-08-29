#!/usr/bin/env bash
# Tier-1 Llama greedy behavioral re-run (BW rename + GSM/ALGO W3).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
export TMPDIR="${TMPDIR:-$HOME/tmp}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR" results/raw logs

MODE="${1:-all}"  # all | bw | gsm | algo | determinism

run() {
  local fams=("$@")
  python3 scripts/llama_greedy_behavioral_rerun.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --families "${fams[@]}" \
    --resume \
    2>&1 | tee -a "logs/llama_greedy_rerun_$(date +%F).log"
}

case "$MODE" in
  all) run bw gsm algo ;;
  bw)  run bw ;;
  gsm) run gsm ;;
  algo) run algo ;;
  determinism)
    python3 scripts/llama_greedy_behavioral_rerun.py --determinism-only --dtype bfloat16
    ;;
  *)
    echo "Usage: $0 {all|bw|gsm|algo|determinism}"
    exit 1
    ;;
esac

echo "=== outputs ==="
ls -lh results/raw/llama_greedy_rerun_*.csv results/raw/llama_determinism_check.csv 2>/dev/null || true
