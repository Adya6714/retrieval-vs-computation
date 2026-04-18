"""Run contamination triage over Probe 1 instances."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from probes.contamination.score import score_problem

INPUT_PATH = Path("data/problems/probe1_instances.csv")
OUTPUT_PATH = Path("results/contamination_triage.csv")

OUTPUT_COLUMNS = [
    "problem_id",
    "problem_family",
    "problem_text",
    "correct_answer",
    "max_ngram_length",
    "max_ngram_count",
    "contamination_score",
]


def _existing_problem_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {str(row.get("problem_id", "")).strip() for row in reader if row.get("problem_id")}


def run_triage(
    limit: int | None = None,
    family: str | None = None,
    resume: bool = True,
    input_path: Path = INPUT_PATH, 
    output_path: Path = OUTPUT_PATH
) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = _existing_problem_ids(output_path) if resume else set()
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with input_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    if family is not None:
        rows = [row for row in rows if row.get("problem_family") == family]

    if limit is not None:
        rows = rows[:limit]

    with output_path.open("a", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)

        if write_header:
            writer.writeheader()
            outfile.flush()

        for row in rows:
            problem_id = str(row.get("problem_id", "")).strip()
            if problem_id and problem_id in processed_ids:
                continue

            problem_text = str(row.get("problem_text", ""))
            score = score_problem(problem_text)

            output_row = {
                "problem_id": problem_id,
                "problem_family": row.get("problem_family", ""),
                "problem_text": problem_text,
                "correct_answer": row.get("correct_answer", ""),
                "max_ngram_length": score["max_ngram_length"],
                "max_ngram_count": score["max_ngram_count"],
                "contamination_score": score["contamination_score"],
            }
            writer.writerow(output_row)
            outfile.flush()

            if problem_id:
                processed_ids.add(problem_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run contamination triage")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N problems then stop")
    parser.add_argument("--family", type=str, default=None, help="Process only problems where problem_family matches this value")
    
    parser.add_argument("--resume", action="store_true", default=True, help="Skip problem_ids already present in the output CSV (default)")
    parser.add_argument("--no-resume", action="store_false", dest="resume", help="Do not skip problem_ids already present")
    
    args = parser.parse_args()

    run_triage(limit=args.limit, family=args.family, resume=args.resume)
