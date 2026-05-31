"""Run contamination triage for ALGO (algorithmic) canonical rows."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from tqdm import tqdm
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except (TimeoutError, OSError):
    pass

DEFAULT_BANK = Path("data/problems/question_bank_algo.csv")
DEFAULT_OUTPUT = Path("results/raw/ALGO_P3_contamination.csv")

OUTPUT_COLUMNS = [
    "problem_id",
    "problem_family",
    "problem_subtype",
    "problem_text",
    "correct_answer",
    "max_ngram_length",
    "max_ngram_count",
    "contamination_score",
    "template_contamination_score",
    "instance_contamination_score",
    "difficulty_numeric",
]

DIFFICULTY_MAP = {
    "easy": 1,
    "medium": 2,
    "hard": 3,
    "": 0,
}


def _existing_problem_ids(output_path: Path) -> set[str]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    with output_path.open("r", newline="", encoding="utf-8") as f:
        return {
            str(row.get("problem_id", "")).strip()
            for row in csv.DictReader(f)
            if row.get("problem_id")
        }


def _score_query(query: str, *, score_problem) -> float:
    if not query.strip():
        return 0.0
    try:
        return float(
            score_problem(query, family="gsm")["contamination_score"]
        )
    except Exception as e:  # noqa: BLE001
        print(f"ERROR scoring query={query!r}: {e}", file=sys.stderr)
        return -1.0


def run_triage(
    *,
    input_path: Path,
    output_path: Path,
    limit: int | None,
    resume: bool,
    max_ngram: int | None,
) -> None:
    from probes.contamination import infinigram_client as _ig
    from probes.contamination.score import score_problem

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

    rows = [
        row
        for row in rows
        if str(row.get("variant_type", "")).strip().lower() == "canonical"
        and str(row.get("problem_family", "")).strip().lower() == "algorithmic"
    ]

    if limit is not None:
        rows = rows[:limit]

    with output_path.open("a", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
            outfile.flush()

        print(f"Processing {len(rows)} problems...")
        for row in tqdm(rows, desc="ALGO Contamination Triage"):
            problem_id = str(row.get("problem_id", "")).strip()
            if problem_id and problem_id in processed_ids:
                continue

            problem_text = str(row.get("problem_text", "")).strip().strip('"')
            correct_answer = str(row.get("correct_answer", "")).strip()

            score = score_problem(
                problem_text,
                family=str(row.get("problem_family", "")),
                max_ngram=max_ngram,
            )
            template_score = _score_query(problem_text, score_problem=score_problem)
            instance_score = _score_query(
                correct_answer, score_problem=score_problem
            )

            diff_key = str(row.get("difficulty", "")).strip().lower()
            difficulty_numeric = DIFFICULTY_MAP.get(diff_key, 0)

            writer.writerow(
                {
                    "problem_id": problem_id,
                    "problem_family": row.get("problem_family", ""),
                    "problem_subtype": row.get("problem_subtype", ""),
                    "problem_text": problem_text,
                    "correct_answer": correct_answer,
                    "max_ngram_length": score["max_ngram_length"],
                    "max_ngram_count": score["max_ngram_count"],
                    "contamination_score": score["contamination_score"],
                    "template_contamination_score": template_score,
                    "instance_contamination_score": instance_score,
                    "difficulty_numeric": difficulty_numeric,
                }
            )
            outfile.flush()
            if problem_id:
                processed_ids.add(problem_id)
            time.sleep(0)

    print(f"\nDone. Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ALGO contamination triage")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--bank-path",
        type=str,
        default=str(DEFAULT_BANK),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
    )
    parser.add_argument("--max-ngram", type=int, default=None)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_false", dest="resume")
    args = parser.parse_args()

    run_triage(
        input_path=Path(args.bank_path),
        output_path=Path(args.output),
        limit=args.limit,
        resume=args.resume,
        max_ngram=args.max_ngram,
    )
