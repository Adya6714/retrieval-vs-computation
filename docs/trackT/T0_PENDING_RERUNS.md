# T0 pending reruns (C2 / HP-23 step 2)

Date: 2026-09-25

## Why listed instead of executed under `--resume`

`OPENROUTER_API_KEY` is present in the environment. Default sweep `--resume` **skips** existing `(problem_id, variant_type, model)` triples in `results/raw/BW_P1_behavioral.csv`. Raw CSVs are append-only, so those triples cannot be deleted to force a re-score. A live re-score therefore needs an explicit force path (sidecar output) — not plain `--resume`.

## Affected bank rows (substring / article-collision fix)

18 BW/MBW W3 rows regenerated in `data/problems/question_bank_bw.csv` (of which 12 had `You are Alice robot arm`).

IDs:

- `BW_496`
- `BW_498`
- `BW_499`
- `BW_500`
- `BW_501`
- `BW_503`
- `BW_504`
- `BW_510`
- `BW_513`
- `BW_514`
- `BW_515`
- `BW_E_001`
- `BW_E_004`
- `MBW_496`
- `MBW_497`
- `MBW_498`
- `MBW_499`
- `MBW_500`

## Raw cells to re-score

For each ID × model below (W3 only): **72** cells currently in `results/raw/BW_P1_behavioral.csv` were collected against the corrupted prompt text.

Models:

- `anthropic/claude-sonnet-4`
- `openai/gpt-4o`
- `meta-llama/llama-3.1-8b-instruct`
- `deepseek/deepseek-r1-distill-llama-70b`

## Suggested force command (sidecar; does not rewrite historical raw)

```bash
# 1) Build a one-shot bank slice of the fixed W3 rows
python - <<'PY'
import csv
from pathlib import Path
ids = {
    "BW_496","BW_498","BW_499","BW_500","BW_501","BW_503","BW_504","BW_510",
    "BW_513","BW_514","BW_515","BW_E_001","BW_E_004",
    "MBW_496","MBW_497","MBW_498","MBW_499","MBW_500",
}
src = Path("data/problems/question_bank_bw.csv")
out = Path("data/problems/_tmp_w3b_fix_slice.csv")
with src.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
    fields = list(rows[0].keys())
slice_rows = [r for r in rows if r["problem_id"] in ids and r["variant_type"] == "W3"]
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(slice_rows)
print("wrote", out, "n=", len(slice_rows))
PY

# 2) Sweep into a new raw file (append-only sibling)
for m in \
  anthropic/claude-sonnet-4 \
  openai/gpt-4o \
  meta-llama/llama-3.1-8b-instruct \
  deepseek/deepseek-r1-distill-llama-70b
do
  PYTHONPATH=. python scripts/BW_P1_SCR_run_behavioral_sweep.py \
    --question-bank-path data/problems/_tmp_w3b_fix_slice.csv \
    --model "$m" --no-resume \
    --output results/raw/BW_P1_behavioral_T0_w3b_fix.csv
done
```

After the sidecar lands, rescore and merge policy are a separate Track T decision.
