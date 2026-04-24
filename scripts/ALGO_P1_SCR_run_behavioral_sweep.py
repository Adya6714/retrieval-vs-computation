#!/usr/bin/env python3
"""Run algorithmic behavioral sweep with strict verifier propagation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contamination.verify_algo import verify_algo


load_dotenv()

OUTPUT_COLUMNS = [
    "problem_id",
    "variant_type",
    "model",
    "model_answer",
    "ground_truth",
    "verified",
    "parse_status",
    "correct_alternative",
    "human_review_flag",
    "correct_canonical",
    "greedy_answer",
    "gave_greedy_answer",
    "difficulty_params_instance_type",
]

HUMAN_REVIEW_COLUMNS = [
    "problem_id",
    "variant_type",
    "model",
    "model_answer",
    "ground_truth",
    "reason",
]


def normalize_for_compare(text: str) -> str:
    s = str(text).lower()
    s = s.replace("→", "->")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^\w\-\>\[\]\{\},:]", "", s)
    return s


def existing_done_keys(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        return set()
    required = {"problem_id", "variant_type", "model"}
    if not required.issubset(df.columns):
        return set()
    done: set[tuple[str, str, str]] = set()
    for _, row in df.iterrows():
        done.add(
            (
                str(row["problem_id"]).strip(),
                str(row["variant_type"]).strip(),
                str(row["model"]).strip(),
            )
        )
    return done


def parse_difficulty_params(raw: str) -> dict:
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError as exc:
        raise ValueError(f"difficulty_params must be valid JSON. {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("difficulty_params JSON must decode to an object.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Algorithmic behavioral sweep.")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--family", default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    bank_path = Path(args.bank)
    if not bank_path.exists():
        raise FileNotFoundError(f"Bank not found: {bank_path}")

    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    output_path = (
        Path(args.output)
        if args.output
        else Path("results") / f"ALGO_P1_RES_behavioral_sweep_{safe_model}.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_path = Path("results/ALGO_P1_RES_human_review_queue.csv")
    review_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(bank_path, dtype=str).fillna("")
    required_cols = {
        "problem_id",
        "variant_type",
        "problem_subtype",
        "problem_text",
        "correct_answer",
        "difficulty_params",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Restrict to algorithmic suite rows.
    df = df[df["problem_id"].str.match(r"^(CC|SP|WIS)_")].copy()
    if args.family:
        family_filter = args.family.strip().lower()
        if family_filter == "algorithmic":
            pass
        else:
            df = df[df["problem_family"].str.lower() == family_filter]
    if args.limit is not None:
        df = df.head(args.limit)

    if not args.dry_run and args.model == "mock":
        raise ValueError("model='mock' is only allowed with --dry-run.")

    if args.dry_run or args.model == "mock":
        client = None
        model_name = "mock"
    else:
        import os

        if not os.environ.get("OPENROUTER_API_KEY"):
            raise EnvironmentError("OPENROUTER_API_KEY is not set.")
        from probes.behavioral.openai_client import OpenRouterClient

        client = OpenRouterClient(model=args.model)
        model_name = args.model

    done = existing_done_keys(output_path) if args.resume else set()
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    write_review_header = not review_path.exists() or review_path.stat().st_size == 0

    n_total = 0
    n_parse_failed = 0

    with output_path.open("a", newline="", encoding="utf-8") as out_f, review_path.open(
        "a", newline="", encoding="utf-8"
    ) as review_f:
        out_writer = csv.DictWriter(out_f, fieldnames=OUTPUT_COLUMNS)
        review_writer = csv.DictWriter(review_f, fieldnames=HUMAN_REVIEW_COLUMNS)
        if write_header:
            out_writer.writeheader()
        if write_review_header:
            review_writer.writeheader()

        for _, row in df.iterrows():
            pid = str(row["problem_id"]).strip()
            variant_type = str(row["variant_type"]).strip()
            key = (pid, variant_type, model_name)
            if args.resume and key in done:
                continue

            difficulty_params = parse_difficulty_params(row["difficulty_params"])
            instance_type = difficulty_params.get("instance_type")
            greedy_answer = difficulty_params.get("greedy_answer")
            if instance_type not in {"standard", "adversarial"}:
                raise ValueError(
                    f"{pid}/{variant_type}: difficulty_params.instance_type must be "
                    "'standard' or 'adversarial'."
                )
            if "greedy_answer" not in difficulty_params:
                raise ValueError(f"{pid}/{variant_type}: missing difficulty_params.greedy_answer")

            problem_text = str(row["problem_text"])
            ground_truth = str(row["correct_answer"])
            subtype = str(row["problem_subtype"]).strip()

            if args.dry_run or model_name == "mock":
                model_answer = ground_truth
            else:
                result = client.complete(pid, problem_text)
                model_answer = str(result.get("response", ""))

            verified, reason, metadata = verify_algo(
                pid,
                model_answer,
                ground_truth,
                subtype,
                variant_type,
                difficulty_params,
            )
            if not isinstance(metadata, dict):
                raise ValueError(f"{pid}/{variant_type}: verifier metadata must be dict.")
            parse_status = metadata.get("parse_status")
            if parse_status not in {
                "parsed_clean",
                "parsed_with_normalization",
                "parse_failed",
            }:
                raise ValueError(
                    f"{pid}/{variant_type}: missing/invalid parse_status from verifier metadata."
                )

            correct_alternative = bool(
                metadata.get("alternative_path", False)
                or metadata.get("alternative_set", False)
            )
            human_review_flag = parse_status == "parse_failed"
            if human_review_flag:
                n_parse_failed += 1
                review_writer.writerow(
                    {
                        "problem_id": pid,
                        "variant_type": variant_type,
                        "model": model_name,
                        "model_answer": model_answer,
                        "ground_truth": ground_truth,
                        "reason": reason,
                    }
                )

            gave_greedy_answer = ""
            if instance_type == "adversarial":
                gave_greedy_answer = str(
                    normalize_for_compare(model_answer)
                    == normalize_for_compare(str(greedy_answer))
                )

            out_writer.writerow(
                {
                    "problem_id": pid,
                    "variant_type": variant_type,
                    "model": model_name,
                    "model_answer": model_answer,
                    "ground_truth": ground_truth,
                    "verified": str(bool(verified)),
                    "parse_status": parse_status,
                    "correct_alternative": str(correct_alternative),
                    "human_review_flag": str(human_review_flag),
                    "correct_canonical": str(variant_type == "canonical" and bool(verified)),
                    "greedy_answer": str(greedy_answer),
                    "gave_greedy_answer": gave_greedy_answer,
                    "difficulty_params_instance_type": instance_type,
                }
            )
            out_f.flush()
            review_f.flush()
            done.add(key)
            n_total += 1

    print(f"Output: {output_path}")
    print(f"Human review queue: {review_path}")
    print(f"Processed rows: {n_total}")
    if n_total > 0:
        parse_fail_rate = n_parse_failed / n_total
        print(
            f"Parse failure rate: {n_parse_failed}/{n_total} = {parse_fail_rate:.1%}"
        )
        if parse_fail_rate > 0.15:
            print("Parse failure rate > 15%. Parser likely broken. STOP.")
    else:
        print("Parse failure rate: 0/0 = 0.0%")


if __name__ == "__main__":
    main()
