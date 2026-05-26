#!/usr/bin/env bash
# Quick progress checker — run anytime: bash results/raw/new_model_sweep_logs/check_progress.sh
cd /Users/adya/Desktop/rvc
python3 - <<'PY'
from pathlib import Path
import pandas as pd
RAW = Path("results/raw")
targets = {
    "GSM_P1_behavioral_gemini.csv": 288,
    "BW_P1_behavioral_gemini.csv": 455,
    "ALGO_P1_behavioral_gemini.csv": 640,
    "GSM_P2_phase1_gemini.csv": 44,
    "ALGO_P2_phase1_gemini.csv": 110,
    "ALGO_P2_phase2_normal_gemini.csv": None,
    "GSM_P1_behavioral_o1mini.csv": 288,
    "BW_P1_behavioral_o1mini.csv": 455,
    "ALGO_P1_behavioral_o1mini.csv": 640,
    "GSM_P2_phase1_o1mini.csv": 44,
}
print(f"{'rows':>12}  {'target':>6}  file")
for f, exp in targets.items():
    p = RAW / f
    if p.exists():
        n = len(pd.read_csv(p))
        t = str(exp) if exp else "?"
        mark = " OK" if exp and n >= exp else ""
        print(f"{n:>12}  {t:>6}  {f}{mark}")
    else:
        print(f"{'—':>12}  {str(exp or '?'):>6}  {f}")
PY
echo ""
echo "Active python sweeps:"
ps aux | grep python3 | grep -E "sweep|probe2|phase" | grep -v grep || echo "  (none)"
