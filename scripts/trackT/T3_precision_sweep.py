#!/usr/bin/env python3
"""T3 precision arms (HP-18).

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


P3_COLS = [
    "problem_id", "variant_type", "model", "model_answer", "ground_truth",
    "verified", "parse_status", "family", "precision", "temperature", "seed",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True,
                    choices=["fp16", "int8", "nf4", "gptq_int4", "awq_int4"])
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    slug = model_slug(args.model)
    out = Path(args.out) if args.out else (
        REPO_ROOT / "results/raw" / f"T3_P1_{slug}_{args.precision}.csv"
    )

    # GPTQ/AWQ: skip if no published checkpoint mapping
    if args.precision in {"gptq_int4", "awq_int4"} and not args.dry_run:
        print(f"[T3] {args.precision} unavailable for {args.model} — recording skip")
        ensure_parent(out)
        if not out.exists():
            with out.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=P3_COLS + ["skip_reason"])
                w.writeheader()
                w.writerow({
                    "problem_id": "", "variant_type": "", "model": args.model,
                    "model_answer": "", "ground_truth": "", "verified": "",
                    "parse_status": "skipped", "family": "",
                    "precision": args.precision, "temperature": "", "seed": "",
                    "skip_reason": "unavailable",
                })
        return

    if args.dry_run:
        import pandas as pd
        gsm = pd.read_csv(REPO_ROOT / "data/problems/question_bank_gsm.csv")
        algo = pd.read_csv(REPO_ROOT / "data/problems/question_bank_algo.csv")
        algo = algo[algo["problem_subtype"] == "coin_change"]
        bank = pd.concat([gsm, algo], ignore_index=True)
        bank = bank[bank["variant_type"].isin(["canonical", "W1", "W3", "W4", "W6"])]
        rows = []
        for _, r in bank.head(3).iterrows():
            rows.append({
                "problem_id": r["problem_id"],
                "variant_type": r["variant_type"],
                "model": args.model,
                "model_answer": "DRY_RUN",
                "ground_truth": r.get("correct_answer", ""),
                "verified": "",
                "parse_status": "dry_run",
                "family": "GSM" if str(r["problem_id"]).startswith("GSM") else "ALGO",
                "precision": args.precision,
                "temperature": 0.0,
                "seed": 0,
            })
        n = append_rows(
            out, P3_COLS, rows,
            resume_key=(lambda row: (row["problem_id"], row["variant_type"], row["model"], row["precision"]))
            if args.resume else None,
        )
        print(f"[T3 dry-run] {args.precision} wrote {n} → {out}")
        return

    raise SystemExit(
        "Non-dry-run precision arm: load weights per HP-18 on Colab T3_precision.ipynb."
    )


if __name__ == "__main__":
    main()
