#!/usr/bin/env bash
# Detached O16 Part A runner (macOS/Linux; survives terminal hangup).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
mkdir -p logs results/derived
export INFINIGRAM_FAST=1
export INFINIGRAM_API_URL="${O16_INFINIGRAM_API_URL:-https://api.infini-gram.io/}"
THROTTLE="${1:-0.5}"
LOG=logs/o16_corpus_search.log
PIDFILE=logs/o16_corpus_search.pid

if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "already running pid=$(cat "$PIDFILE")"
  exit 0
fi

# nohup+disown: portable substitute for setsid on macOS
nohup python scripts/consolidate/o16_corpus_ground_truth.py --throttle "$THROTTLE" \
  >>"$LOG" 2>&1 </dev/null &
echo $! > "$PIDFILE"
disown "$(cat "$PIDFILE")" 2>/dev/null || true
sleep 1
if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "started pid=$(cat "$PIDFILE") log=$LOG throttle=$THROTTLE"
else
  echo "FAILED to start; see $LOG" >&2
  exit 1
fi
