#!/usr/bin/env python3
"""Migrate Dataset Week 1 (wide) into unified question_bank.csv (long)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from probes.common.io import QUESTION_BANK_COLUMNS, QUESTION_BANK_PATH

VARIANT_COLUMNS = ["w2", "w3", "w4", "w5", "w6"]
VARIANT_LABELS = {
    "w2": "W2",
    "w3": "W3",
    "w4": "W4",
    "w5": "W5",
    "w6": "W6",
}


def _clean(value: str | None) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _canonical_row(raw: Dict[str, str]) -> Dict[str, str]:
    return {
        "problem_id": _clean(raw.get("problem_id")),
        "variant_type": "canonical",
        "problem_text": _clean(raw.get("problem_text")),
        "correct_answer": _clean(raw.get("correct_answer")),
        "problem_family": _clean(raw.get("problem_family")),
        "problem_subtype": _clean(raw.get("problem_subtype")).lower(),
        "difficulty": _clean(raw.get("difficulty")),
        "contamination_pole": _clean(raw.get("contamination_pole")),
        "source": _clean(raw.get("source")),
        "verifier_function": _clean(raw.get("verifier_function")),
        "difficulty_params": _clean(raw.get("difficulty_params")),
        "notes": _clean(raw.get("notes")),
    }


def _variant_row(canonical: Dict[str, str], variant_text: str, variant_type: str) -> Dict[str, str]:
    row = dict(canonical)
    row["variant_type"] = variant_type
    row["problem_text"] = variant_text
    # Keep W5 correct answers as-is from canonical for now; generation script can override.
    return row


def _validate_strict_columns(path: Path, fieldnames: List[str]) -> None:
    expected = set(QUESTION_BANK_COLUMNS)
    got = set(fieldnames)
    if expected != got:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        raise ValueError(
            f"{path} schema mismatch. Missing={missing} Extra={extra}. "
            "Question bank must exactly match strict schema."
        )


def migrate(input_csv: Path, output_csv: Path) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if not output_csv.exists():
        raise FileNotFoundError(
            f"Output question bank not found: {output_csv}. "
            "Create/initialize it first."
        )

    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
    _validate_strict_columns(output_csv, fieldnames)

    with input_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        input_rows = list(reader)

    migrated_rows: List[Dict[str, str]] = []
    for raw in input_rows:
        canonical = _canonical_row(raw)
        problem_id = canonical["problem_id"]
        if not problem_id:
            continue
        if not canonical["problem_text"]:
            continue

        migrated_rows.append(canonical)

        for wide_col in VARIANT_COLUMNS:
            text = _clean(raw.get(wide_col))
            if not text:
                continue
            migrated_rows.append(_variant_row(canonical, text, VARIANT_LABELS[wide_col]))

    with output_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QUESTION_BANK_COLUMNS)
        # writer.writeheader()
        writer.writerows(migrated_rows)

    canonical_count = sum(1 for row in migrated_rows if row["variant_type"] == "canonical")
    variant_count = len(migrated_rows) - canonical_count
    print(f"Migrated {len(migrated_rows)} rows to {output_csv}")
    print(f"Canonical rows: {canonical_count}")
    print(f"Variant rows: {variant_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Week 1 wide CSV to strict question bank")
    parser.add_argument("--input", type=Path, default=Path("Dataset Week 1.csv"))
    parser.add_argument("--output", type=Path, default=Path(QUESTION_BANK_PATH))
    args = parser.parse_args()
    migrate(args.input, args.output)


if __name__ == "__main__":
    main()

