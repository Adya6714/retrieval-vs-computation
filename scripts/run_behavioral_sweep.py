"""Phase 3 behavioral sweep entrypoint.
Use --dry-run to test the full pipeline without spending API credits.
Resume-safe: already-scored (problem_id, variant_type) pairs are skipped.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv

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


def _existing_pairs(output_path: Path) -> set[tuple[str, str]]:
    """Return set of (problem_id, variant_type) pairs already in output CSV."""
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(output_path, dtype=str)
        if "problem_id" not in df.columns or "variant_type" not in df.columns:
            return set()
        # fillna so canonical rows (empty variant_type written as "") are
        # read back as "" rather than NaN, ensuring resume matching works.
        df["variant_type"] = df["variant_type"].fillna("")
        return {
            (str(r["problem_id"]).strip(), str(r["variant_type"]).strip())
            for _, r in df.iterrows()
        }
    except pd.errors.EmptyDataError:
        return set()


def _build_rows(
    df_instances: pd.DataFrame,
    df_variants: pd.DataFrame,
) -> list[dict]:
    """Combine canonical and variant rows into a unified sweep list."""
    rows: list[dict] = []

    for _, inst in df_instances.iterrows():
        rows.append(
            {
                "problem_id": str(inst["problem_id"]).strip(),
                "problem_family": str(inst.get("problem_family", "")).strip(),
                "problem_text": str(inst.get("problem_text", "")),
                "correct_answer": str(inst.get("correct_answer", "")),
                "variant_type": "",
            }
        )

    if not df_variants.empty:
        valid_pids = set(df_instances["problem_id"].astype(str).str.strip())
        for _, var in df_variants.iterrows():
            pid = str(var.get("problem_id", "")).strip()
            if pid not in valid_pids:
                continue
            # Resolve family from the canonical row
            canon_rows = df_instances[
                df_instances["problem_id"].astype(str).str.strip() == pid
            ]
            family = (
                str(canon_rows.iloc[0].get("problem_family", "")).strip()
                if not canon_rows.empty
                else ""
            )
            rows.append(
                {
                    "problem_id": pid,
                    "problem_family": family,
                    "problem_text": str(var.get("variant_text", "")),
                    "correct_answer": str(var.get("correct_answer", "")),
                    "variant_type": str(var.get("variant_type", "")).strip(),
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3 behavioral sweep — canonical + variants"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N canonical problems")
    parser.add_argument("--family", type=str, default=None,
                        help="Filter to a single problem_family")
    parser.add_argument("--model", type=str,
                        default="anthropic/claude-sonnet-4-5-20251001",
                        help="Model identifier passed to the API client")
    parser.add_argument("--probe", type=str, choices=["probe1", "probe2"],
                        default="probe1")
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

    instances_path = problems_dir / f"{args.probe}_instances.csv"
    variants_path = problems_dir / f"{args.probe}_variants.csv"
    output_path = results_dir / "behavioral_sweep.csv"

    # 2. Load canonical problems
    if not instances_path.exists():
        raise FileNotFoundError(f"Instances file not found: {instances_path}")

    df_instances = pd.read_csv(instances_path, dtype=str)
    df_instances["problem_id"] = df_instances["problem_id"].astype(str).str.strip()

    if args.family is not None:
        df_instances = df_instances[
            df_instances["problem_family"].astype(str).str.strip() == args.family
        ]

    if args.limit is not None:
        df_instances = df_instances.head(args.limit)

    # 3. Load variants
    df_variants = pd.DataFrame()
    if variants_path.exists():
        df_variants = pd.read_csv(variants_path, dtype=str)
        df_variants["problem_id"] = df_variants["problem_id"].astype(str).str.strip()
    else:
        print(f"WARNING: Variants file not found at {variants_path}. "
              "Sweeping canonical problems only.")

    # 4. Build unified sweep list
    all_rows = _build_rows(df_instances, df_variants)

    # 5. Select client
    if args.dry_run:
        from probes.behavioral.mock_client import MockClient
        client = MockClient(default_response="The answer is 42.")
        model_name = "mock"
        print("DRY RUN: using MockClient — no API credits spent.")
    else:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. "
                "Add it to .env and re-run, or use --dry-run to test locally."
            )
        from probes.behavioral.anthropic_client import AnthropicClient
        client = AnthropicClient(model=args.model)
        model_name = args.model

    # 6. Load existing results for resume
    done_pairs: set[tuple[str, str]] = set()
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
            if args.resume and (pid, vtype) in done_pairs:
                n_skipped += 1
                continue

            try:
                # 7a. Query the model
                result = client.complete(pid, problem_text)
                raw_response = result.get("response", "")

                # 7b. Verify answer
                try:
                    is_correct = verify_answer(pid, raw_response, correct_answer, family)
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

                done_pairs.add((pid, vtype))
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
