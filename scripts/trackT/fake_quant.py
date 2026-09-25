#!/usr/bin/env python3
"""T3 per-band fake INT4 (HP-18 step 2).

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
    ap.add_argument("--model", required=True)
    ap.add_argument("--band", type=int, required=True)
    ap.add_argument("--n-bands", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    slug = model_slug(args.model)
    out = REPO_ROOT / "results/raw" / f"T3_P1_{slug}_band{args.band}.csv"
    cols = [
        "problem_id", "variant_type", "model", "model_answer", "ground_truth",
        "verified", "parse_status", "family", "precision", "band", "n_bands",
        "group_size",
    ]
    if args.dry_run:
        row = {
            "problem_id": "DRY",
            "variant_type": "canonical",
            "model": args.model,
            "model_answer": "DRY_RUN",
            "ground_truth": "",
            "verified": "",
            "parse_status": "dry_run",
            "family": "",
            "precision": "fake_int4",
            "band": args.band,
            "n_bands": args.n_bands,
            "group_size": args.group_size,
        }
        n = append_rows(
            out, cols, [row],
            resume_key=(lambda r: (r["problem_id"], r["band"], r["model"]))
            if args.resume else None,
        )
        print(f"[fake_quant dry-run] band {args.band} wrote {n} → {out}")
        return

    raise SystemExit(
        "Non-dry-run: apply round-to-nearest symmetric INT4 (group 128) to Linear "
        "weights of the selected decoder band (HP-18), then score GSM+coin_change."
    )


if __name__ == "__main__":
    main()
