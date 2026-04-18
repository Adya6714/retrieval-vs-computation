"""Run a behavioral sweep on Probe 1 instances via OpenRouter."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from probes.contamination.verify import verify_answer  # noqa: F401

INPUT_PATH = Path("data/problems/probe1_instances.csv")
OUTPUT_PATH = Path("results/behavioral_sweep.csv")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "openai/gpt-4o-mini"
REQUEST_TIMEOUT_SECONDS = 60

OUTPUT_COLUMNS = [
    "problem_id",
    "problem_family",
    "model",
    "raw_response",
    "behavioral_correct",
]


def _existing_problem_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    with output_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {str(row.get("problem_id", "")).strip() for row in reader if row.get("problem_id")}


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _query_openrouter(problem_text: str, api_key: str) -> str:
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": problem_text}],
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    choices = payload.get("choices", [])
    if not choices:
        raise ValueError("OpenRouter response missing choices.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str):
        raise ValueError("OpenRouter response content is not a string.")
    return content


def run_model_sweep(limit: int | None = None, family: str | None = None, input_path: Path = INPUT_PATH, output_path: Path = OUTPUT_PATH) -> None:
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is not set.")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = _existing_problem_ids(output_path)
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
            if not problem_id:
                print(f"WARNING: skipping row with missing problem_id: {row}")
                continue
            if problem_id and problem_id in processed_ids:
                continue

            problem_text = str(row.get("problem_text", ""))
            problem_family = str(row.get("problem_family", ""))
            correct_answer = str(row.get("correct_answer", ""))

            try:
                raw_response = _query_openrouter(problem_text, api_key)
            except Exception as e:
                print(f"ERROR: failed on problem_id={problem_id} after retries: {e}")
                continue
            
            behavioral_correct = bool(verify_answer(problem_id, raw_response, correct_answer, problem_family))

            writer.writerow(
                {
                    "problem_id": problem_id,
                    "problem_family": problem_family,
                    "model": MODEL_NAME,
                    "raw_response": raw_response,
                    "behavioral_correct": behavioral_correct,
                }
            )
            outfile.flush()

            if problem_id:
                processed_ids.add(problem_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run behavioral sweep")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N problems")
    parser.add_argument("--family", type=str, default=None, help="Filter by problem_family")
    parser.add_argument("--model", type=str, default="openai/gpt-4o-mini", help="Override MODEL_NAME")
    args = parser.parse_args()

    MODEL_NAME = args.model

    run_model_sweep(limit=args.limit, family=args.family)
