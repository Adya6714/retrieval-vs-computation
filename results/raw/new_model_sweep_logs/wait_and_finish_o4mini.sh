#!/usr/bin/env bash
# Poll OpenRouter until key limit resets, then run remaining o4-mini sweeps.
set -eu
cd <REPO_ROOT>
export OPENROUTER_MAX_TOKENS=4096
LOG=results/raw/new_model_sweep_logs/wait_and_finish.log
exec >>"$LOG" 2>&1

test_api() {
  python3 - <<'PY'
import os, sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(".env"))
from probes.behavioral.openai_client import OpenRouterClient
r = OpenRouterClient(model="openai/o4-mini").complete("ping", "Reply OK only")
sys.exit(0 if r.get("response", "").startswith("ERROR:") is False and "OK" in r.get("response", "").upper() else 1)
PY
}

echo "=== wait loop start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
for i in $(seq 1 120); do
  if test_api; then
    echo "API ready after attempt $i $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    # Gemini ALGO P1 — retry ERROR rows only
    python3 scripts/ALGO_P1_SCR_run_behavioral_sweep.py \
      --bank data/problems/question_bank_algo.csv \
      --model google/gemini-2.5-flash \
      --output results/raw/ALGO_P1_behavioral_gemini.csv \
      --resume || true
    bash results/raw/new_model_sweep_logs/finish_o4mini.sh
    python3 results/paper/complete_new_models_pipeline.py
    python3 results/paper/run_all_new_model_metrics.py
    exit 0
  fi
  echo "attempt $i: limit/credits block, sleep 300s..."
  sleep 300
done
echo "Gave up after 120 attempts"
exit 1
