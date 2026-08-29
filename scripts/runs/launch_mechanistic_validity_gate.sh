#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export HF_TOKEN="${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
mkdir -p upload logs results/raw
python3 scripts/run_mechanistic_validity_gate.py \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dtype bfloat16 \
  --out-dir upload \
  2>&1 | tee "logs/mechanistic_validity_gate_$(date +%F_%H%M).log"
