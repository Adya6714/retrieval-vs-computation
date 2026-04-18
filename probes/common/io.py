"""
This module handles all CSV I/O for the pipeline, standardizing data loading 
and appending results while validating required schemas.
"""

import csv
from pathlib import Path
import pandas as pd

QUESTION_BANK_PATH = "data/problems/question_bank.csv"
QUESTION_BANK_COLUMNS = [
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
]


def load_question_bank(path: str = QUESTION_BANK_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    missing = set(QUESTION_BANK_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"File {path} is missing required columns: {', '.join(sorted(missing))}")
    return df


def load_problems(path: str) -> pd.DataFrame:
    # Backward-compatible alias retained while moving to a single question bank.
    df = load_question_bank(path)
    # Canonical problems are represented by variant_type == "canonical".
    if "variant_type" in df.columns:
        df = df[df["variant_type"].astype(str).str.strip().str.lower() == "canonical"].copy()
    return df


def load_variants(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {
        "problem_id",
        "variant_id",
        "variant_type",
        "variant_text",
        "correct_answer",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"File {path} is missing required columns: {', '.join(missing)}")
    return df


def append_result(path: str, row: dict) -> None:
    p = Path(path)
    write_header = not p.exists() or p.stat().st_size == 0
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_results(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)
