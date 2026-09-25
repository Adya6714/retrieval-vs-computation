#!/usr/bin/env python3
"""T2 pre-registered logistic mixed model (HP-17 step 6).

HP Track T entry point. Supports --resume and --dry-run.
Do not invent numbers; raw CSVs are append-only truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def model_slug(model: str) -> str:
    return model.replace("/", "_").replace(" ", "_")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def append_rows(path: Path, fieldnames: list[str], rows: list[dict], resume_key=None) -> int:
    """Append rows; if resume_key, skip keys already present."""
    ensure_parent(path)
    done = set()
    if path.exists() and path.stat().st_size > 0 and resume_key is not None:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                done.add(resume_key(row))
    write_header = not path.exists() or path.stat().st_size == 0
    n = 0
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for row in rows:
            if resume_key is not None and resume_key(row) in done:
                continue
            w.writerow({k: row.get(k, "") for k in fieldnames})
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-glob", default="results/raw/T2_P1_*.csv")
    ap.add_argument("--out", default="results/derived/T2_mixed_model.csv")
    ap.add_argument("--report", default="docs/trackT/T2_REPORT.md")
    args = ap.parse_args()
    out = REPO_ROOT / args.out
    report = REPO_ROOT / args.report
    ensure_parent(out)
    ensure_parent(report)
    paths = sorted(REPO_ROOT.glob(args.raw_glob))
    cols = ["term", "coef", "ci_low", "ci_high", "n", "note"]
    if not paths:
        with out.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=cols).writeheader()
        report.write_text("# T2_REPORT\n\nNo T2 raws yet.\n")
        print(f"[T2 mixed] empty {out}")
        return
    # Placeholder table — fit on Colab when Acc_can floors clear; do not invent coefficients.
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow({
            "term": "Coder:W4",
            "coef": "",
            "ci_low": "",
            "ci_high": "",
            "n": "",
            "note": "fit pending non-dry-run raws; see PREREGISTRATION Track T",
        })
    report.write_text(
        "# T2_REPORT\n\n"
        f"Raws: {', '.join(p.name for p in paths)}\n\n"
        "Mixed-model coefficients pending full Acc_can ≥ .30 cells.\n"
    )
    print(f"[T2 mixed] wrote {out}")


if __name__ == "__main__":
    main()
