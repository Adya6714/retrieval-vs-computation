#!/usr/bin/env python3
"""Orchestrate Probe 1 behavioral sweeps: all models × all families.

Runs one model at a time, one family at a time, with --resume on every sweep.
Probe 2 and 3 are out of scope — run separately after P1 completes.

Usage:
    python scripts/run_full_sweep.py
    python scripts/run_full_sweep.py --model anthropic/claude-sonnet-4
    python scripts/run_full_sweep.py --family BW
    python scripts/run_full_sweep.py --status
    python scripts/run_full_sweep.py --model meta-llama/llama-3.1-8b-instruct --dry-run --limit 2
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGRESS_PATH = REPO_ROOT / "results" / "sweep_progress.json"

MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "deepseek/deepseek-r1-distill-llama-70b",
    "qwen/qwen-2.5-72b-instruct",
]

FAMILIES = ("BW", "GSM", "ALGO")

BW_BANK = "data/problems/question_bank_bw.csv"
GSM_BANK = "data/problems/question_bank_gsm.csv"
ALGO_BANK = "data/problems/question_bank_algo.csv"

BW_OUTPUT = REPO_ROOT / "results/raw/BW_P1_behavioral.csv"
BW_SCRIPT = REPO_ROOT / "scripts/BW_P1_SCR_run_behavioral_sweep.py"
ALGO_SCRIPT = REPO_ROOT / "scripts/ALGO_P1_SCR_run_behavioral_sweep.py"

DONE_RE = re.compile(
    r"processed=(\d+).*?skipped\(resume\)=(\d+).*?errors=(\d+)",
    re.DOTALL,
)
ALGO_PROCESSED_RE = re.compile(r"Processed rows:\s*(\d+)")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _empty_progress() -> dict:
    return {model: {fam: {"status": "pending"} for fam in FAMILIES} for model in MODELS}


def load_progress() -> dict:
    if not PROGRESS_PATH.exists():
        return _empty_progress()
    with PROGRESS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    # Ensure all models/families present
    base = _empty_progress()
    for model in MODELS:
        if model not in data:
            data[model] = base[model]
        for fam in FAMILIES:
            if fam not in data[model]:
                data[model][fam] = {"status": "pending"}
    return data


def save_progress(progress: dict) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
        f.write("\n")


def _algo_output_path(model: str) -> Path:
    sys.path.insert(0, str(REPO_ROOT))
    from probes.common.results_paths import algo_p1_behavioral

    return REPO_ROOT / algo_p1_behavioral(model)


def count_rows(model: str, family: str) -> int:
    try:
        if family == "ALGO":
            path = _algo_output_path(model)
            if not path.exists():
                return 0
            df = pd.read_csv(path, dtype=str)
            if "model" in df.columns:
                return int((df["model"] == model).sum())
            return len(df)
        if not BW_OUTPUT.exists():
            return 0
        df = pd.read_csv(BW_OUTPUT, dtype=str)
        m = df["model"] == model if "model" in df.columns else pd.Series([True] * len(df))
        if family == "GSM":
            gsm = df["problem_id"].astype(str).str.startswith("GSM")
            return int((m & gsm).sum())
        # BW: non-GSM rows for this model
        gsm = df["problem_id"].astype(str).str.startswith("GSM")
        return int((m & ~gsm).sum())
    except Exception:
        return 0


def print_status_table(progress: dict | None = None) -> None:
    progress = progress or load_progress()
    col_w = 10
    header = f"{'Model':<36} | {'BW':<{col_w}} | {'GSM':<{col_w}} | {'ALGO':<{col_w}}"
    print(header)
    print("-" * len(header))
    for model in MODELS:
        cells = []
        for fam in FAMILIES:
            entry = progress.get(model, {}).get(fam, {})
            status = entry.get("status", "pending")
            if status == "complete" and "rows" in entry:
                cells.append(f"complete ({entry['rows']})")
            else:
                cells.append(status)
        print(f"{model:<36} | {cells[0]:<{col_w}} | {cells[1]:<{col_w}} | {cells[2]:<{col_w}}")


def _parse_bw_stats(output: str) -> tuple[int, int, int]:
    m = DONE_RE.search(output)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return 0, 0, 0


def _parse_algo_stats(output: str) -> tuple[int, int, int]:
    m = ALGO_PROCESSED_RE.search(output)
    if m:
        n = int(m.group(1))
        return n, 0, 0
    return 0, 0, 0


def build_sweep_command(
    *,
    model: str,
    family: str,
    dry_run: bool,
    limit: int | None,
) -> list[str]:
    cmd = [sys.executable]
    if family == "BW":
        cmd += [
            str(BW_SCRIPT),
            "--model",
            model,
            "--family",
            "planning_suite",
            "--question-bank-path",
            BW_BANK,
            "--resume",
        ]
    elif family == "GSM":
        cmd += [
            str(BW_SCRIPT),
            "--model",
            model,
            "--family",
            "arithmetic_reasoning",
            "--question-bank-path",
            GSM_BANK,
            "--resume",
        ]
    else:  # ALGO
        cmd += [
            str(ALGO_SCRIPT),
            "--bank",
            ALGO_BANK,
            "--model",
            model,
            "--resume",
        ]
    if dry_run:
        cmd.append("--dry-run")
    if limit is not None:
        cmd += ["--limit", str(limit)]
    return cmd


def run_family_sweep(
    *,
    model: str,
    family: str,
    dry_run: bool,
    limit: int | None,
) -> tuple[int, str, int, int, int]:
    cmd = build_sweep_command(model=model, family=family, dry_run=dry_run, limit=limit)
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    out = (result.stdout or "") + (result.stderr or "")
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")

    if family == "ALGO":
        processed, skipped, errors = _parse_algo_stats(out)
    else:
        processed, skipped, errors = _parse_bw_stats(out)

    return result.returncode, out, processed, skipped, errors


def print_model_row_counts(model: str) -> None:
    bw_n = count_rows(model, "BW")
    gsm_n = count_rows(model, "GSM")
    algo_n = count_rows(model, "ALGO")
    print(f"  BW:   {bw_n} rows")
    print(f"  GSM:  {gsm_n} rows")
    print(f"  ALGO: {algo_n} rows")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Orchestrate Probe 1 behavioral sweeps (all models × families)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Run only this OpenRouter model id (all families unless --family set).",
    )
    parser.add_argument(
        "--family",
        type=str,
        choices=FAMILIES,
        default=None,
        help="Run only this family (BW, GSM, or ALGO) across selected models.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print progress table from sweep_progress.json and exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass --dry-run to sweep scripts (no API calls).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Pass --limit N to each sweep script.",
    )
    args = parser.parse_args()

    if args.status:
        progress = load_progress()
        print_status_table(progress)
        return

    models = [args.model] if args.model else list(MODELS)
    if args.model and args.model not in MODELS:
        print(f"WARNING: {args.model!r} not in default MODELS list; running anyway.")

    families = [args.family] if args.family else list(FAMILIES)
    progress = load_progress()

    for model in models:
        print(f"\n{'=' * 60}")
        print(f"MODEL: {model}")
        print(f"{'=' * 60}")

        for family in families:
            entry = progress.setdefault(model, {}).setdefault(family, {"status": "pending"})
            if entry.get("status") == "complete" and not args.dry_run:
                print(f"{model} × {family}: skip (already complete, {entry.get('rows', '?')} rows)")
                continue

            entry["status"] = "in_progress"
            entry["started_at"] = _utc_now()
            if not args.dry_run:
                save_progress(progress)

            exit_code, _out, processed, skipped, errors = run_family_sweep(
                model=model,
                family=family,
                dry_run=args.dry_run,
                limit=args.limit,
            )

            print(f"{model} × {family}: done. processed={processed} skipped={skipped} errors={errors}")
            if errors > 0:
                print(f"WARNING: {errors} error(s) for {model} × {family} — resume will retry later")
            if exit_code != 0:
                print(f"ERROR: exit code {exit_code} for {model} × {family}")
                entry["status"] = "error"
                entry["exit_code"] = exit_code
                entry["timestamp"] = _utc_now()
            elif args.dry_run:
                entry["status"] = "dry_run"
                entry["timestamp"] = _utc_now()
            else:
                rows = count_rows(model, family)
                entry["status"] = "complete"
                entry["rows"] = rows
                entry["processed"] = processed
                entry["skipped"] = skipped
                entry["errors"] = errors
                entry["timestamp"] = _utc_now()

            if not args.dry_run:
                save_progress(progress)

        print(f"\n=== {model} P1 COMPLETE ===")
        print_model_row_counts(model)

    if not args.dry_run:
        save_progress(progress)
    print("\nSweep orchestration finished.")
    print_status_table(progress)


if __name__ == "__main__":
    main()
