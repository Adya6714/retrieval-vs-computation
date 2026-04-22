"""Phase 3 behavioral sweep entrypoint.
Use --dry-run to test the full pipeline without spending API credits.
Resume-safe: already-scored (problem_id, variant_type, model) triples are skipped
unless the latest row for that triple has raw_response starting with ERROR:.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

from probes.common.io import QUESTION_BANK_PATH, QUESTION_BANK_COLUMNS
from probes.contamination.verify import verify_answer

load_dotenv()

OUTPUT_COLUMNS = [
    "problem_id",
    "problem_family",
    "variant_type",
    "model",
    "raw_response",
    "behavioral_correct",
]


def _load_paths() -> dict:
    with open("configs/paths.yaml", "r") as f:
        return yaml.safe_load(f)


def _existing_pairs(output_path: Path) -> set[tuple[str, str, str]]:
    """Return (problem_id, variant_type, model) keys already successfully scored.

    The last row per key wins. Keys whose latest row is an API/transport failure
    (raw_response starts with 'ERROR:') are omitted so the next run retries.
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(output_path, dtype=str)
        if "problem_id" not in df.columns or "variant_type" not in df.columns:
            return set()
        df["variant_type"] = df["variant_type"].fillna("")
        if "model" not in df.columns:
            return set()
        raw_col = (
            df["raw_response"].fillna("").astype(str)
            if "raw_response" in df.columns
            else pd.Series([""] * len(df), index=df.index)
        )
        df = df.assign(_raw=raw_col)
        last = df.groupby(
            ["problem_id", "variant_type", "model"], sort=False
        ).last()
        done: set[tuple[str, str, str]] = set()
        for (pid, vtype, model), row in last.iterrows():
            if str(row["_raw"]).strip().startswith("ERROR:"):
                continue
            done.add(
                (str(pid).strip(), str(vtype).strip(), str(model).strip())
            )
        return done
    except pd.errors.EmptyDataError:
        return set()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 behavioral sweep — canonical + variants"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N rows after family/other filters")
    parser.add_argument("--family", type=str, default=None,
                        help="Filter to a single problem_family")
    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-3.7-sonnet",
        help="OpenRouter model id (default: anthropic/claude-3.7-sonnet)",
    )
    parser.add_argument("--probe", type=str, choices=["probe1", "probe2"],
                        default="probe1")
    parser.add_argument(
        "--question-bank",
        type=str,
        default=QUESTION_BANK_PATH,
        help="Path to unified question bank CSV (used for probe1)",
    )
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Skip (problem_id, variant_type) pairs already in output (default)")
    parser.add_argument("--no-resume", action="store_false", dest="resume",
                        help="Re-score everything, ignoring existing output")
    parser.add_argument("--dry-run", action="store_true",
                        help="Use MockClient — no API credits spent")
    args = parser.parse_args()

    paths = _load_paths()
    problems_dir = Path(paths.get("problems_dir", "./data/problems"))
    results_dir = Path(paths.get("results_dir", "./results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.probe == "probe1":
        instances_path = Path(args.question_bank)
    else:
        instances_path = problems_dir / f"{args.probe}_instances.csv"
    output_path = results_dir / "BW_RES_P1_behavioral_sweep.csv"
    # Backward-compat: if legacy file exists and new file does not, continue using legacy.
    legacy_output_path = results_dir / "behavioral_sweep.csv"
    if legacy_output_path.exists() and not output_path.exists():
        output_path = legacy_output_path

    # 2. Load question bank rows (canonical + W2–W6 variants, etc.)
    if not instances_path.exists():
        raise FileNotFoundError(f"Instances file not found: {instances_path}")

    df_instances = pd.read_csv(instances_path, dtype=str)
    missing_cols = set(QUESTION_BANK_COLUMNS) - set(df_instances.columns)
    if missing_cols:
        raise ValueError(
            f"Question bank missing required columns: {sorted(missing_cols)}"
        )
    df_instances["problem_id"] = df_instances["problem_id"].astype(str).str.strip()

    if args.family is not None:
        # Match against problem_subtype first (e.g. "blocksworld"),
        # falling back to problem_family for rows that don't have a subtype.
        subtype_col = df_instances["problem_subtype"].astype(str).str.strip().str.lower() \
            if "problem_subtype" in df_instances.columns \
            else pd.Series([""] * len(df_instances), index=df_instances.index)
        family_col = df_instances["problem_family"].astype(str).str.strip().str.lower()
        family_filter = args.family.lower()
        df_instances = df_instances[
            (subtype_col == family_filter) | (family_col == family_filter)
        ]

    if args.limit is not None:
        df_instances = df_instances.head(args.limit)

    # 3. Build unified sweep list
    all_rows = []
    n_skipped_empty_answer = 0
    for _, inst in df_instances.iterrows():
        correct_answer = str(inst.get("correct_answer", ""))
        if correct_answer.strip() == "" or correct_answer.strip().lower() in {"nan", "none"}:
            n_skipped_empty_answer += 1
            continue
        all_rows.append(
            {
                "problem_id": str(inst["problem_id"]).strip(),
                "problem_family": str(
                    inst.get("problem_subtype", inst.get("problem_family", ""))
                ).strip().lower(),
                "problem_text": str(inst.get("problem_text", "")),
                "correct_answer": correct_answer,
                "variant_type": str(inst.get("variant_type", "")).strip(),
            }
        )
    if n_skipped_empty_answer:
        print(
            f"Skipping {n_skipped_empty_answer} rows with empty correct_answer "
            "(e.g., unsupported variants like BW_080/W6)."
        )

    # 4. Select client
    if args.dry_run:
        from probes.behavioral.mock_client import MockClient
        client = MockClient(default_response="The answer is 42.")
        model_name = "mock"
        print("DRY RUN: using MockClient — no API credits spent.")
    else:
        import os
        if not os.environ.get("OPENROUTER_API_KEY"):
            raise EnvironmentError(
                "OPENROUTER_API_KEY is not set. "
                "Add it to .env and re-run, or use --dry-run to test locally."
            )
        from probes.behavioral.openai_client import OpenRouterClient
        client = OpenRouterClient(model=args.model)
        model_name = args.model

    # 5. Load existing results for resume
    done_pairs: set[tuple[str, str, str]] = set()
    if args.resume:
        done_pairs = _existing_pairs(output_path)

    write_header = not output_path.exists() or output_path.stat().st_size == 0

    n_processed = 0
    n_skipped = 0
    n_errors = 0

    with output_path.open("a", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
            outfile.flush()

        for row in all_rows:
            pid = row["problem_id"]
            vtype = row["variant_type"]
            family = row["problem_family"]
            problem_text = row["problem_text"]
            correct_answer = row["correct_answer"]

            # 6. Skip already-scored pairs
            if args.resume and (pid, vtype, model_name) in done_pairs:
                n_skipped += 1
                continue

            try:
                # 7a. Query the model
                result = client.complete(pid, problem_text)
                raw_response = result.get("response", "")

                # 7b. Verify answer
                try:
                    is_correct = verify_answer(
                        pid, raw_response, correct_answer, family, problem_text=problem_text
                    )
                except ValueError:
                    # Unrecognized family — log and treat as unscored
                    print(f"WARNING: unrecognized family '{family}' for {pid}. "
                          "behavioral_correct set to empty.")
                    is_correct = ""

                # 7c. Write row
                writer.writerow(
                    {
                        "problem_id": pid,
                        "problem_family": family,
                        "variant_type": vtype,
                        "model": model_name,
                        "raw_response": raw_response,
                        "behavioral_correct": is_correct,
                    }
                )
                outfile.flush()  # 7d

                done_pairs.add((pid, vtype, model_name))
                n_processed += 1

            except Exception as exc:  # 7e
                print(f"ERROR: problem_id={pid} variant_type={vtype!r}: {exc}")
                n_errors += 1

    # 8. Summary
    print(
        f"\nDone. processed={n_processed}, "
        f"skipped(resume)={n_skipped}, "
        f"errors={n_errors}"
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
