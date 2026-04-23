#!/usr/bin/env python3
"""Propagate canonical correct_answer to W2 and W4 rows in question_bank.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_CSV = Path("data/problems/question_bank.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy canonical correct_answer to W2/W4 rows by problem_id"
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing")
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    with args.csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        raise ValueError("CSV has no header.")

    canonical_answers: dict[str, str] = {}
    for row in rows:
        pid = str(row.get("problem_id", "")).strip()
        vtype = str(row.get("variant_type", "")).strip().lower()
        if pid and vtype == "canonical":
            canonical_answers[pid] = str(row.get("correct_answer", ""))

    updated = 0
    for row in rows:
        pid = str(row.get("problem_id", "")).strip()
        vtype = str(row.get("variant_type", "")).strip().upper()
        if vtype not in {"W2", "W4"}:
            continue
        if pid not in canonical_answers:
            continue
        target_answer = canonical_answers[pid]
        current_answer = str(row.get("correct_answer", ""))
        if current_answer == target_answer:
            continue
        row["correct_answer"] = target_answer
        updated += 1
        print(f"Updated W2/W4 for problem_id {pid}")

    print(f"Updated {updated} rows.")

    if args.dry_run:
        return

    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

