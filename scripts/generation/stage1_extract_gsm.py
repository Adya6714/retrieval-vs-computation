#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contamination.verify import verify_gsm_answer  # noqa: E402
from scripts.generation.utils.bank_writer import (  # noqa: E402
    BANK_COLUMNS,
    max_id_number,
    next_problem_id,
    read_existing_bank,
    write_rows,
)
from scripts.generation.utils.duplicate_detector import DuplicateDetector  # noqa: E402

GSM_SOURCE_ROOT = REPO_ROOT / "data/sources/gsm_symbolic"
GSM_SYMBOLIC_FILENAME = "GSM_symbolic.jsonl"
GSM_P1_PATH = GSM_SOURCE_ROOT / "GSM_p1.jsonl"
GSM_P2_PATH = GSM_SOURCE_ROOT / "GSM_p2.jsonl"
GSM_P1_FILENAME = "GSM_p1.jsonl"
GSM_P2_FILENAME = "GSM_p2.jsonl"
QUESTION_BANK = REPO_ROOT / "data/problems/question_bank_gsm.csv"
OUT_CSV = REPO_ROOT / "data/staging/gsm_canonical.csv"

TARGET_HIGH = {"easy": 3, "medium": 3, "hard": 3}
TARGET_MEDIUM = {"easy": 5, "medium": 5, "hard": 5}

ANSWER_RE = re.compile(r"####\s*([^\n\r]+)")
NUMERIC_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
def _find_gsm_jsonl(filename: str) -> Path:
    if not GSM_SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"GSM JSONL search root not found: {GSM_SOURCE_ROOT}"
        )
    match = next(GSM_SOURCE_ROOT.rglob(filename), None)
    if match is None:
        raise FileNotFoundError(
            f"Could not find {filename} under search root {GSM_SOURCE_ROOT}"
        )
    return match


def _normalize_number(text: str) -> str:
    m = NUMERIC_RE.search((text or "").strip())
    if not m:
        return (text or "").strip()
    raw = m.group(0).replace(",", "")
    try:
        val = float(raw)
        if abs(val - int(val)) < 1e-9:
            return str(int(val))
        return str(val)
    except ValueError:
        return raw


def _extract_answer_from_solution(solution_text: str) -> str | None:
    matches = ANSWER_RE.findall(solution_text or "")
    if not matches:
        return None
    return _normalize_number(matches[-1].strip())


def _combined_max_id(*dfs) -> int:
    return max((max_id_number(df, "GSM") for df in dfs), default=0)


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{i}: {exc}") from exc
            rows.append(payload)
    return rows


def _line_has_numeric_equality(line: str) -> bool:
    """True when a line has '=' with at least one number on each side."""
    if "=" not in line:
        return False
    lhs, _, rhs = line.partition("=")
    return bool(NUMERIC_RE.search(lhs)) and bool(NUMERIC_RE.search(rhs))


def _count_arithmetic_ops(solution_text: str) -> int:
    return sum(
        1
        for ln in str(solution_text or "").splitlines()
        if _line_has_numeric_equality(ln)
    )


def _difficulty_from_solution(solution_text: str) -> str:
    op_count = _count_arithmetic_ops(solution_text)
    if op_count <= 2:
        return "easy"
    if op_count <= 4:
        return "medium"
    return "hard"


def _safe_text(payload: dict[str, Any], keys: list[str]) -> str:
    for k in keys:
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _make_row(
    *,
    problem_id: str,
    problem_text: str,
    answer: str,
    difficulty: str,
    contamination_pole: str,
    subtype: str,
    template_id: str,
    jsonl_name: str,
) -> dict[str, str]:
    return {
        "problem_id": problem_id,
        "variant_type": "canonical",
        "problem_text": problem_text,
        "correct_answer": answer,
        "problem_family": "arithmetic_reasoning",
        "problem_subtype": subtype,
        "difficulty": difficulty,
        "contamination_pole": contamination_pole,
        "source": f"gsm_symbolic:template_id={template_id}",
        "verifier_function": "verify_gsm_answer",
        "difficulty_params": "",
        "notes": f"stage1 extraction from {jsonl_name}",
        "status": "ok",
        "selection_reason": "stage1_canonical_extraction",
    }


def _extract_from_sources(
    sources: list[tuple[Path, str, str]],
    targets: dict[str, int],
    start_max: int,
    detector: DuplicateDetector,
) -> tuple[list[dict[str, str]], int]:
    out: list[dict[str, str]] = []
    next_max = start_max
    counts = {k: 0 for k in targets}

    for jsonl_path, contamination_pole, subtype in sources:
        payloads = _iter_jsonl(jsonl_path)
        for payload in payloads:
            if all(counts[d] >= targets[d] for d in targets):
                break

            template_raw = payload.get("template_id", payload.get("id"))
            template_id = str(template_raw).strip() if template_raw is not None else ""
            if not template_id or detector.is_used(template_id):
                continue

            problem_text = _safe_text(payload, ["question", "problem", "prompt"])
            solution_text = _safe_text(payload, ["answer", "solution", "rationale"])
            if not problem_text or not solution_text:
                continue

            answer = _extract_answer_from_solution(solution_text)
            if answer is None:
                continue
            if not verify_gsm_answer(f"#### {answer}", answer):
                continue

            difficulty = _difficulty_from_solution(solution_text)
            if counts[difficulty] >= targets[difficulty]:
                continue

            next_max += 1
            pid = next_problem_id("GSM", next_max - 1)
            out.append(
                _make_row(
                    problem_id=pid,
                    problem_text=problem_text,
                    answer=answer,
                    difficulty=difficulty,
                    contamination_pole=contamination_pole,
                    subtype=subtype,
                    template_id=template_id,
                    jsonl_name=jsonl_path.name,
                )
            )
            counts[difficulty] += 1
            detector.mark_used(template_id)

        if all(counts[d] >= targets[d] for d in targets):
            break

    return out, next_max


def _resolve_medium_jsonl(path: Path, filename: str) -> Path:
    if path.exists():
        return path
    return _find_gsm_jsonl(filename)


def _extract_high(start_max: int, detector: DuplicateDetector) -> tuple[list[dict[str, str]], int]:
    source_symbolic = _find_gsm_jsonl(GSM_SYMBOLIC_FILENAME)
    return _extract_from_sources(
        [(source_symbolic, "high", "gsm_symbolic")],
        TARGET_HIGH,
        start_max,
        detector,
    )


def _extract_medium(start_max: int, detector: DuplicateDetector) -> tuple[list[dict[str, str]], int]:
    source_p1 = _resolve_medium_jsonl(GSM_P1_PATH, GSM_P1_FILENAME)
    source_p2 = _resolve_medium_jsonl(GSM_P2_PATH, GSM_P2_FILENAME)
    return _extract_from_sources(
        [
            (source_p1, "medium", "gsm_p1p2"),
            (source_p2, "medium", "gsm_p1p2"),
        ],
        TARGET_MEDIUM,
        start_max,
        detector,
    )


def _load_staging_rows() -> list[dict[str, str]]:
    if not OUT_CSV.exists():
        return []
    with OUT_CSV.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{k: str(row.get(k, "")) for k in BANK_COLUMNS} for row in reader]


def _reclassify_gsm_042(rows: list[dict[str, str]]) -> bool:
    changed = False
    for row in rows:
        if str(row.get("problem_id", "")).strip() == "GSM_042":
            if row.get("difficulty") != "easy":
                row["difficulty"] = "easy"
                changed = True
    return changed


def _print_summary(selected: list[dict[str, str]], tier: str) -> None:
    targets: dict[str, int] = {}
    if tier in {"high", "both"}:
        for k, v in TARGET_HIGH.items():
            targets[k] = targets.get(k, 0) + v
    if tier in {"medium", "both"}:
        for k, v in TARGET_MEDIUM.items():
            targets[k] = targets.get(k, 0) + v

    difficulty_counts = {k: 0 for k in targets}
    for row in selected:
        d = str(row.get("difficulty", "")).strip().lower()
        if d in difficulty_counts:
            difficulty_counts[d] += 1

    high_count = sum(1 for r in selected if r.get("contamination_pole") == "high")
    medium_count = sum(1 for r in selected if r.get("contamination_pole") == "medium")

    print("Selection summary:")
    print(f"  Tier mode: {tier}")
    print(f"  Total selected: {len(selected)}/{sum(targets.values()) if targets else 0}")
    for diff in ("easy", "medium", "hard"):
        if diff in targets:
            print(f"  Difficulty {diff}: {difficulty_counts.get(diff, 0)}/{targets[diff]}")
    print(f"  Contamination high (GSM_symbolic): {high_count}")
    print(f"  Contamination medium (GSM_p1/p2): {medium_count}")
    print()
    for row in selected:
        print(
            f"{row['problem_id']} | {row['difficulty']} | "
            f"{row['contamination_pole']} | {row['source']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 GSM canonical extraction")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Print selection summary only")
    mode.add_argument("--run", action="store_true", help="Append to data/staging/gsm_canonical.csv")
    parser.add_argument(
        "--tier",
        choices=["high", "medium", "both"],
        default="both",
        help="high=GSM_symbolic only, medium=GSM_p1/p2 only, both=run both paths",
    )
    args = parser.parse_args()

    bank_df = read_existing_bank(QUESTION_BANK)
    staging_df = read_existing_bank(OUT_CSV)

    detector = DuplicateDetector(QUESTION_BANK, OUT_CSV)
    current_max = _combined_max_id(bank_df, staging_df)

    selected: list[dict[str, str]] = []
    if args.tier in {"high", "both"}:
        high_rows, current_max = _extract_high(current_max, detector)
        selected.extend(high_rows)

    if args.tier in {"medium", "both"}:
        medium_rows, current_max = _extract_medium(current_max, detector)
        selected.extend(medium_rows)

    _print_summary(selected, args.tier)

    if args.run:
        existing_rows = _load_staging_rows()
        reclassified = _reclassify_gsm_042(existing_rows)
        merged = existing_rows + selected
        write_rows(merged, OUT_CSV)
        print(f"\nWrote {len(merged)} rows to {OUT_CSV} ({len(selected)} newly added)")
        if reclassified:
            print("Post-process: GSM_042 difficulty updated to easy")
    else:
        print("\nDry-run mode: no files written.")


if __name__ == "__main__":
    main()
