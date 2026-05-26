"""Run contamination triage over Probe 1 instances."""

from __future__ import annotations

import csv
from pathlib import Path

from probes.contamination.score import score_problem

INPUT_PATH = Path("data/probe1_instances.csv")
OUTPUT_PATH = Path("results/contamination_triage.csv")

OUTPUT_COLUMNS = [
    "problem_id",
    "problem_family",
    "problem_text",
    "correct_answer",
    "max_ngram_length",
    "max_ngram_count",
    "contamination_score",
    "behavioral_correct",
]


def _existing_problem_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {str(row.get("problem_id", "")).strip() for row in reader if row.get("problem_id")}


def run_triage(input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = _existing_problem_ids(output_path)
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with input_path.open("r", newline="", encoding="utf-8") as infile, output_path.open(
        "a", newline="", encoding="utf-8"
    ) as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)

        if write_header:
            writer.writeheader()
            outfile.flush()

        for row in reader:
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
                "behavioral_correct": row.get("behavioral_correct", ""),
            }
            writer.writerow(output_row)
            outfile.flush()

            if problem_id:
                processed_ids.add(problem_id)


if __name__ == "__main__":
    run_triage()
