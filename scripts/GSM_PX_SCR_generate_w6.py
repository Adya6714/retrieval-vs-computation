#!/usr/bin/env python3
"""Generate W6 procedural variants for GSM question bank."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


TEMPLATE_RE = re.compile(r"template_id=(\d+)")
ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")

BANK_PATH = Path("data/problems/gsm_question_bank.csv")
FALLBACK_BANK_PATH = Path("data/problems/question_bank_gsm.csv")
GENERATED_JSONL_PATHS = [
    Path("~/Desktop/ml-gsm-symbolic/generated_data/GSM_symbolic.jsonl").expanduser(),
    Path("~/Desktop/ml-gsm-symbolic/generated_data/GSM_p1.jsonl").expanduser(),
    Path("~/Desktop/ml-gsm-symbolic/generated_data/GSM_p2.jsonl").expanduser(),
]


def get_template_key(payload: dict) -> str | None:
    template = payload.get("template_id", payload.get("id"))
    if template is None:
        return None
    return str(template)


def get_instance_key(payload: dict) -> str | None:
    instance = payload.get("instance")
    if instance is None:
        return None
    return str(instance)


def extract_correct_answer(answer_text: str) -> str | None:
    matches = ANSWER_RE.findall(answer_text or "")
    if not matches:
        return None
    return matches[-1].strip()


def load_generated_index() -> dict[tuple[str, str], dict]:
    index: dict[tuple[str, str], dict] = {}
    for jsonl_path in GENERATED_JSONL_PATHS:
        if not jsonl_path.exists():
            raise FileNotFoundError(f"Missing generated data file: {jsonl_path}")
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {jsonl_path} at line {line_number}: {exc}"
                    ) from exc
                template = get_template_key(payload)
                instance = get_instance_key(payload)
                if template is None or instance is None:
                    continue
                index[(template, instance)] = payload
    return index


def parse_template_id_from_source(source: str) -> str | None:
    match = TEMPLATE_RE.search(source or "")
    return match.group(1) if match else None


def source_prefix_from_canonical(source: str) -> str:
    if not source:
        return "generated/ml-gsm-symbolic"
    split_token = " | template_id="
    if split_token in source:
        return source.split(split_token, 1)[0].strip()
    return source.strip()


def main() -> None:
    read_path = BANK_PATH
    if not read_path.exists():
        if FALLBACK_BANK_PATH.exists():
            print(
                "Note: "
                f"{BANK_PATH} not found; bootstrapping from {FALLBACK_BANK_PATH}."
            )
            read_path = FALLBACK_BANK_PATH
        else:
            raise FileNotFoundError(
                f"Could not find {BANK_PATH} or {FALLBACK_BANK_PATH}."
            )

    generated_index = load_generated_index()

    with read_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        all_rows = list(reader)

    if not fieldnames:
        raise ValueError(f"{read_path} is missing headers.")

    canonical_rows = [
        row for row in all_rows if row.get("variant_type", "").strip() == "canonical"
    ]
    non_w6_rows = [row for row in all_rows if row.get("variant_type", "").strip() != "w6"]
    removed_w6_count = len(all_rows) - len(non_w6_rows)

    generated_rows: list[dict] = []
    missing_templates: list[str] = []
    missing_answers: list[str] = []

    for canonical in canonical_rows:
        problem_id = canonical.get("problem_id", "").strip()
        template_id = parse_template_id_from_source(canonical.get("source", ""))
        if template_id is None:
            missing_templates.append(f"{problem_id}:missing_template_id_in_source")
            continue

        generated = generated_index.get((template_id, "1"))
        if generated is None:
            missing_templates.append(f"{problem_id}:template_id={template_id}")
            continue

        answer = extract_correct_answer(generated.get("answer", ""))
        if answer is None:
            missing_answers.append(f"{problem_id}:template_id={template_id}")
            continue

        source_prefix = source_prefix_from_canonical(canonical.get("source", ""))
        new_row = dict(canonical)
        new_row["variant_type"] = "w6"
        new_row["problem_text"] = (generated.get("question") or "").strip()
        new_row["correct_answer"] = answer
        new_row["source"] = (
            f"{source_prefix} | template_id={template_id} | instance=1 | "
            "variant=w6_procedural"
        )
        generated_rows.append(new_row)

    final_rows = non_w6_rows + generated_rows

    BANK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BANK_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Saved: {BANK_PATH}")
    print(f"Removed existing w6 rows: {removed_w6_count}")
    print(f"Generated w6 rows: {len(generated_rows)}")
    print(f"Missing template instance=1: {len(missing_templates)}")
    for item in missing_templates:
        print(f"  - {item}")
    print(f"Missing #### answer line: {len(missing_answers)}")
    for item in missing_answers:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
