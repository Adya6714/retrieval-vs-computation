#!/usr/bin/env bash
# Run after FortiClient VPN is Connected.
# Greedy-decode Llama-3.1-8B-Instruct on ALGO canonical; score with verify_algo.
set -euo pipefail
HOST="${GPU_SSH:-adya_srivastava_2023@172.24.16.177}"
ROOT="${REMOTE_RVC_DIR:-~/retrieval-vs-computation}"

echo "=== scp greedy script ==="
scp "$(dirname "$0")/algo_llama_greedy_accuracy.py" \
  "$HOST:$ROOT/scripts/"

echo "=== remote run (tmux) ==="
ssh -t "$HOST" bash -lc "
  cd $ROOT && source .venv/bin/activate &&
  pip -q install networkx &&
  mkdir -p logs results/raw &&
  tmux new -s algo_greedy -d \"python3 scripts/algo_llama_greedy_accuracy.py \\
      --model meta-llama/Llama-3.1-8B-Instruct \\
      --output results/raw/ALGO_llama31_8b_greedy_canonical.csv \\
      --resume \\
      2>&1 | tee logs/algo_llama_greedy_\\\$(date +%F).log\" &&
  tmux ls &&
  echo 'Attached log: tmux attach -t algo_greedy'
"
