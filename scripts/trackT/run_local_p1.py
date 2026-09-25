#!/usr/bin/env python3
"""Local HF Probe 1 runner for Track T (HP-17 / T2).

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


P1_COLS = [
    "problem_id", "variant_type", "model", "model_answer", "ground_truth",
    "verified", "parse_status", "family", "temperature", "seed",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--families", default="GSM,ALGO")
    ap.add_argument("--subtypes", default="coin_change,shortest_path")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (
        REPO_ROOT / "results/raw" / f"T2_P1_{model_slug(args.model)}.csv"
    )
    import pandas as pd
    frames = []
    for fam in [f.strip() for f in args.families.split(",") if f.strip()]:
        if fam == "GSM":
            df = pd.read_csv(REPO_ROOT / "data/problems/question_bank_gsm.csv")
            df["family"] = "GSM"
        elif fam == "ALGO":
            df = pd.read_csv(REPO_ROOT / "data/problems/question_bank_algo.csv")
            subs = {s.strip() for s in args.subtypes.split(",") if s.strip()}
            df = df[df["problem_subtype"].isin(subs)]
            df["family"] = "ALGO"
        else:
            raise SystemExit(f"unsupported family {fam}")
        frames.append(df)
    bank = pd.concat(frames, ignore_index=True)

    if args.dry_run:
        rows = []
        for _, r in bank.head(4).iterrows():
            rows.append({
                "problem_id": r["problem_id"],
                "variant_type": r.get("variant_type", "canonical"),
                "model": args.model,
                "model_answer": "DRY_RUN",
                "ground_truth": r.get("correct_answer", ""),
                "verified": "",
                "parse_status": "dry_run",
                "family": r["family"],
                "temperature": 0.0,
                "seed": 0,
            })
        n = append_rows(
            out, P1_COLS, rows,
            resume_key=(lambda row: (row["problem_id"], row["variant_type"], row["model"]))
            if args.resume else None,
        )
        print(f"[T2 dry-run] wrote {n} rows → {out}")
        return

    raise SystemExit(
        "Non-dry-run: load HF causal LM at FP16 with T1-stable config and score "
        "via probes/ (HP-17). Use Colab T2_qwen_family.ipynb with DRY_RUN=False."
    )


if __name__ == "__main__":
    main()
