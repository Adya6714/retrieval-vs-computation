"""Run contamination triage over Probe 1 instances."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

from tqdm import tqdm
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

load_dotenv()

from probes.contamination import infinigram_client as _ig
from probes.contamination.score import score_problem
from probes.common.io import QUESTION_BANK_PATH

INPUT_PATH = Path(QUESTION_BANK_PATH)
OUTPUT_PATH = Path("results/BW_P3_RES_contamination_triage.csv")

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
    output_path: Path = OUTPUT_PATH,
    max_ngram: int | None = None,
) -> None:
    print(
        f"Infini-gram endpoint {_ig.API_URL!r} index={_ig.INDEX_NAME!r} "
        f"ssl_verify={_ig.SSL_VERIFY}",
        flush=True,
    )
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = _existing_problem_ids(output_path) if resume else set()
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    with input_path.open("r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)

    if rows:
        required_columns = {
            "problem_id",
            "variant_type",
            "problem_text",
            "correct_answer",
            "problem_family",
        }
        missing_cols = required_columns - set(rows[0].keys())
        if missing_cols:
            raise ValueError(
                f"Question bank missing required columns: {sorted(missing_cols)}"
            )

    # Triage should run over canonical rows only.
    rows = [
        row
        for row in rows
        if str(row.get("variant_type", "")).strip().lower() == "canonical"
    ]

    if family is not None:
        rows = [
            row for row in rows
            if row.get("problem_subtype", "").strip().lower() == family.lower()
            or row.get("problem_family", "").strip().lower() == family.lower()
        ]

    if limit is not None:
        rows = rows[:limit]

    with output_path.open("a", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)

        if write_header:
            writer.writeheader()
            outfile.flush()

        print(f"Processing {len(rows)} problems...")
        for row in tqdm(rows, desc="Contamination Triage"):
            problem_id = str(row.get("problem_id", "")).strip()
            if problem_id and problem_id in processed_ids:
                continue

            problem_text = str(row.get("problem_text", "")).strip().strip('"')
            score = score_problem(
                problem_text,
                family=str(row.get("problem_family", "")),
                max_ngram=max_ngram,
            )

            output_row = {
                "problem_id": problem_id,
                "problem_family": row.get("problem_family", row.get("problem_subtype", "")),
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
    
    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run contamination triage")
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only first N problems then stop"
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Process only problems where problem_family matches this value",
    )
    parser.add_argument(
        "--question-bank-path",
        type=str,
        default=str(INPUT_PATH),
        help="Path to question bank CSV",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_PATH),
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-ngram",
        type=int,
        default=None,
        help="Override maximum n-gram length used in contamination scoring",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Skip problem_ids already present in the output CSV (default)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Do not skip problem_ids already present",
    )

    args = parser.parse_args()

    run_triage(
        limit=args.limit,
        family=args.family,
        resume=args.resume,
        input_path=Path(args.question_bank_path),
        output_path=Path(args.output),
        max_ngram=args.max_ngram,
    )
