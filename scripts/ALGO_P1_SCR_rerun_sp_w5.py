#!/usr/bin/env python3
"""Re-run Probe 1 shortest_path W5 only (50 × 5 models).

Writes a new raw file. Does not modify existing ALGO_P1_behavioral_*.csv.
A sidecar lists those old W5 rows as superseded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.behavioral.openai_client import OpenRouterClient  # noqa: E402
from probes.behavioral.sampling import DEFAULT_TEMPERATURE  # noqa: E402
from probes.common.results_paths import RAW, DERIVED, ensure_dirs, model_slug  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402
from probes.contamination.verify_algo import verify_algo  # noqa: E402

BANK = REPO_ROOT / "data/problems/question_bank_algo.csv"
OUT_RAW = RAW / "ALGO_P1_sp_w5_rerun.csv"
SUPERSEDED = DERIVED / "ALGO_P1_sp_w5_superseded.csv"
SEED = 0

MODELS = (
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "meta-llama/llama-3.1-8b-instruct",
    "google/gemini-2.5-flash",
    "openai/o4-mini",
)

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
    "difficulty_params_instance_type",
    "temperature",
    "seed",
    "rerun",
]


def write_superseded() -> int:
    """Index old W5 rows in existing P1 raw files; do not edit those files."""
    ensure_dirs()
    rows_out: list[dict[str, str]] = []
    for path in sorted(RAW.glob("ALGO_P1_behavioral_*.csv")):
        if "sp_w5" in path.name or "review" in path.name:
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if normalize_variant(row.get("variant_type")) != "W5":
                    continue
                pid = str(row.get("problem_id", "")).strip()
                if not pid.startswith("SP_"):
                    continue
                rows_out.append(
                    {
                        "source_file": path.name,
                        "problem_id": pid,
                        "variant_type": "W5",
                        "model": str(row.get("model", "")).strip(),
                        "superseded": "True",
                        "replacement_file": OUT_RAW.name,
                        "reason": "unanswerable_prompt_original_edge_directions",
                    }
                )
    SUPERSEDED.parent.mkdir(parents=True, exist_ok=True)
    with SUPERSEDED.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_file",
                "problem_id",
                "variant_type",
                "model",
                "superseded",
                "replacement_file",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)
    return len(rows_out)


def existing_done(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    done: set[tuple[str, str, str]] = set()
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ans = str(row.get("model_answer", "")).strip()
            if ans.startswith("ERROR:"):
                continue
            done.add(
                (
                    str(row.get("problem_id", "")).strip(),
                    normalize_variant(row.get("variant_type")),
                    str(row.get("model", "")).strip(),
                )
            )
    return done


def load_w5() -> list[dict[str, str]]:
    with BANK.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        r
        for r in rows
        if str(r.get("problem_subtype", "")).strip() == "shortest_path"
        and normalize_variant(r.get("variant_type")) == "W5"
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--supersede-only", action="store_true")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--model", action="append", dest="models")
    args = parser.parse_args()
    load_dotenv(REPO_ROOT / ".env")

    n_super = write_superseded()
    print(f"Wrote {SUPERSEDED} ({n_super} superseded old W5 rows)")
    if args.supersede_only:
        return

    models = tuple(args.models) if args.models else MODELS
    problems = load_w5()
    print(f"W5 problems: {len(problems)}; models: {len(models)}; calls: {len(problems)*len(models)}")
    if args.dry_run:
        return
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise EnvironmentError("OPENROUTER_API_KEY is not set.")

    ensure_dirs()
    done = existing_done(OUT_RAW) if args.resume else set()
    write_header = not OUT_RAW.exists() or OUT_RAW.stat().st_size == 0
    clients = {
        m: OpenRouterClient(model=m, temperature=DEFAULT_TEMPERATURE, seed=SEED)
        for m in models
    }

    n = 0
    with OUT_RAW.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        for model in models:
            client = clients[model]
            for row in problems:
                pid = str(row["problem_id"]).strip()
                key = (pid, "W5", model)
                if key in done:
                    continue
                gold = str(row["correct_answer"])
                params = json.loads(row["difficulty_params"])
                result = client.complete(pid, str(row["problem_text"]))
                answer = str(result.get("response", ""))
                verified, reason, meta = verify_algo(
                    pid, answer, gold, "shortest_path", "W5", params
                )
                parse_status = str((meta or {}).get("parse_status", ""))
                writer.writerow(
                    {
                        "problem_id": pid,
                        "variant_type": "W5",
                        "model": model,
                        "model_answer": answer,
                        "ground_truth": gold,
                        "verified": str(bool(verified)),
                        "parse_status": parse_status,
                        "correct_alternative": str(bool(meta.get("alternative_path", False))),
                        "human_review_flag": str(parse_status == "parse_failed"),
                        "difficulty_params_instance_type": params.get("instance_type", ""),
                        "temperature": DEFAULT_TEMPERATURE,
                        "seed": SEED,
                        "rerun": "True",
                    }
                )
                f.flush()
                done.add(key)
                n += 1
                print(f"  {model} {pid} verified={verified} parse={parse_status}")
    print(f"Wrote {OUT_RAW} ({n} new rows)")


if __name__ == "__main__":
    main()
