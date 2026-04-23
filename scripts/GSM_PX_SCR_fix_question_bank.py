#!/usr/bin/env python3
"""Apply targeted fixes to the GSM question bank CSV."""

from __future__ import annotations

import csv
import re
from pathlib import Path


LEADING_NUMBER_RE = re.compile(r"^-?[\d,]+(?:\.\d+)?")

PRIMARY_CSV_PATH = Path("data/problems/gsm_question_bank.csv")
FALLBACK_CSV_PATH = Path("data/problems/question_bank_gsm.csv")

GSM_003_W2_TEXT = """| Parameter | Value |
|---|---|
| Purple cards | 30 |
| Yellow cards | 50% more than purple cards |
| Blue cards | Equal to purple + yellow |
| Objective | Total cards |"""

GSM_018_W3_TEXT = (
    "A competitive exam consists of 80 questions, of which 25% are biology "
    "questions and 15% are geometry questions. The remaining questions are "
    "split equally between economics and geography. Jordan is expected to "
    "correctly answer 10% of the biology questions, 25% of the geometry "
    "questions, and 1/3 of both the economics and geography questions. How "
    "many points in total is Jordan certain to score?"
)

GSM_005_W4_TEXT = """Let p = 6, e = 1, d = 60, n = 6.
w = (p + e) * d * n
x = w / 7
y = w / 7
total_min = w + x + y
total_hr = total_min / 60

Find total_hr."""


def extract_leading_number(value: str, *, strip_commas: bool) -> str:
    """Return the leading numeric token from value, or value if absent."""
    text = (value or "").strip()
    match = LEADING_NUMBER_RE.match(text)
    if not match:
        return text
    numeric = match.group(0)
    return numeric.replace(",", "") if strip_commas else numeric


def resolve_csv_path() -> Path:
    if PRIMARY_CSV_PATH.exists():
        return PRIMARY_CSV_PATH
    if FALLBACK_CSV_PATH.exists():
        return FALLBACK_CSV_PATH
    raise FileNotFoundError(
        f"Could not find {PRIMARY_CSV_PATH} or {FALLBACK_CSV_PATH}."
    )


def main() -> None:
    csv_path = resolve_csv_path()

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        raise ValueError(f"{csv_path} is missing headers.")

    summary = {
        "w3_units_stripped": 0,
        "gsm_003_w2_problem_text_fixed": 0,
        "gsm_003_w2_answer_fixed": 0,
        "gsm_008_w6_or_w5_1408_to_1024_fixed": 0,
        "gsm_018_w3_problem_text_fixed": 0,
        "gsm_018_w3_answer_fixed": 0,
        "gsm_005_w4_problem_text_fixed": 0,
        "global_correct_answer_normalized": 0,
    }

    # 1) Strip units from all w3 correct_answer fields.
    for row in rows:
        if row.get("variant_type", "").strip() != "w3":
            continue
        old_value = row.get("correct_answer", "")
        new_value = extract_leading_number(old_value, strip_commas=False)
        if new_value != old_value:
            row["correct_answer"] = new_value
            summary["w3_units_stripped"] += 1

    # 2) Fix GSM_003 w2 problem_text and answer.
    for row in rows:
        if (
            row.get("problem_id", "").strip() == "GSM_003"
            and row.get("variant_type", "").strip() == "w2"
        ):
            if row.get("problem_text", "") != GSM_003_W2_TEXT:
                row["problem_text"] = GSM_003_W2_TEXT
                summary["gsm_003_w2_problem_text_fixed"] += 1
            if row.get("correct_answer", "") != "150":
                row["correct_answer"] = "150"
                summary["gsm_003_w2_answer_fixed"] += 1

    # 3) Fix GSM_008 w6 (or w5 legacy): replace 1408 with 1024 in problem_text.
    for row in rows:
        if row.get("problem_id", "").strip() != "GSM_008":
            continue
        if row.get("variant_type", "").strip() not in {"w6", "w5"}:
            continue
        old_text = row.get("problem_text", "")
        new_text = old_text.replace("1408", "1024")
        if new_text != old_text:
            row["problem_text"] = new_text
            summary["gsm_008_w6_or_w5_1408_to_1024_fixed"] += 1

    # 4) Fix GSM_018 w3 renamed version text and keep answer 21.
    for row in rows:
        if (
            row.get("problem_id", "").strip() == "GSM_018"
            and row.get("variant_type", "").strip() == "w3"
        ):
            if row.get("problem_text", "") != GSM_018_W3_TEXT:
                row["problem_text"] = GSM_018_W3_TEXT
                summary["gsm_018_w3_problem_text_fixed"] += 1
            if row.get("correct_answer", "") != "21":
                row["correct_answer"] = "21"
                summary["gsm_018_w3_answer_fixed"] += 1

    # 5) Fix GSM_005 w4 to the specified variable-setup text.
    for row in rows:
        if (
            row.get("problem_id", "").strip() == "GSM_005"
            and row.get("variant_type", "").strip() == "w4"
        ):
            if row.get("problem_text", "") != GSM_005_W4_TEXT:
                row["problem_text"] = GSM_005_W4_TEXT
                summary["gsm_005_w4_problem_text_fixed"] += 1

    # 6) Normalize all correct_answer values to pure numeric strings.
    for row in rows:
        old_value = row.get("correct_answer", "")
        new_value = extract_leading_number(old_value, strip_commas=True)
        if new_value != old_value:
            row["correct_answer"] = new_value
            summary["global_correct_answer_normalized"] += 1

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Applied fixes to: {csv_path}")
    print("Fix summary:")
    print(f"- w3 unit-stripping updates: {summary['w3_units_stripped']}")
    print(
        "- GSM_003 w2 text fixes: "
        f"{summary['gsm_003_w2_problem_text_fixed']}"
    )
    print(
        "- GSM_003 w2 answer fixes: "
        f"{summary['gsm_003_w2_answer_fixed']}"
    )
    print(
        "- GSM_008 w6/w5 1408->1024 replacements: "
        f"{summary['gsm_008_w6_or_w5_1408_to_1024_fixed']}"
    )
    print(
        "- GSM_018 w3 text fixes: "
        f"{summary['gsm_018_w3_problem_text_fixed']}"
    )
    print(
        "- GSM_018 w3 answer fixes: "
        f"{summary['gsm_018_w3_answer_fixed']}"
    )
    print(
        "- GSM_005 w4 text fixes: "
        f"{summary['gsm_005_w4_problem_text_fixed']}"
    )
    print(
        "- Global correct_answer normalizations: "
        f"{summary['global_correct_answer_normalized']}"
    )


if __name__ == "__main__":
    main()
