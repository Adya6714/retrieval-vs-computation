from __future__ import annotations

import csv
from pathlib import Path
import re

import pandas as pd


BANK_COLUMNS = [
    "problem_id",
    "variant_type",
    "problem_text",
    "correct_answer",
    "problem_family",
    "problem_subtype",
    "difficulty",
    "contamination_pole",
    "source",
    "verifier_function",
    "difficulty_params",
    "notes",
    "status",
    "selection_reason",
]


def read_existing_bank(bank_path: Path) -> pd.DataFrame:
    if not bank_path.exists():
        return pd.DataFrame(columns=BANK_COLUMNS)
    return pd.read_csv(bank_path, dtype=str).fillna("")


def used_source_keys(df: pd.DataFrame) -> set[str]:
    keys: set[str] = set()
    for raw in df.get("source", []):
        text = str(raw).strip()
        if not text:
            continue
        keys.add(text)
        path_m = re.search(r"path=([^|]+)", text)
        if path_m:
            path_val = path_m.group(1).strip()
            keys.add(path_val)
            keys.add(Path(path_val).name)
        file_m = re.search(r"filename=([^|]+)", text)
        if file_m:
            filename = file_m.group(1).strip()
            keys.add(filename)
    return keys


def max_id_number(df: pd.DataFrame, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$", flags=re.IGNORECASE)
    max_num = 0
    for raw in df.get("problem_id", []):
        m = pattern.match(str(raw).strip())
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num


def next_problem_id(prefix: str, current_max: int) -> str:
    return f"{prefix}_{current_max + 1:03d}"


def write_rows(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BANK_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in BANK_COLUMNS})
