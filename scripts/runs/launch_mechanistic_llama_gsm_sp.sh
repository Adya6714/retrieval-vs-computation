#!/usr/bin/env bash
# Launch Appendix-H Llama GSM+SP mechanistic follow-up on a GPU box.
# Prefer GSM first (smaller); then SP. Resume-safe.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"   # all | gsm | sp | analyze
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
OUT="results/raw/mechanistic_llama_gsm_sp_raw.csv"
mkdir -p results/raw results/derived logs

run_fam() {
  local fam="$1"
  echo "=== Llama mechanistic: family=$fam ==="
  python3 scripts/run_mechanistic_llama_gsm_sp.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --families "$fam" \
    --variants canonical W3 W6 \
    --output "$OUT" \
    --resume \
    2>&1 | tee -a "logs/mechanistic_llama_gsm_sp_${fam}_$(date +%F).log"
}

case "$MODE" in
  gsm) run_fam gsm ;;
  sp)  run_fam sp ;;
  all)
    run_fam gsm
    run_fam sp
    ;;
  analyze)
    ;;
  *)
    echo "Usage: $0 {all|gsm|sp|analyze}"
    exit 1
    ;;
esac

echo "=== Analyzing ==="
python3 scripts/analyze_mechanistic_llama_gsm_sp.py --raw "$OUT"
echo "Done."
