#!/usr/bin/env python3
"""Re-score existing Probe 1 raw responses with the current verifiers.

Reads results/raw/*P1_behavioral*.csv, never writes back to results/raw/,
and writes sibling files under results/derived/ with a ``_rescored`` suffix.

Does not call any model API.

Inclusion rule: a row is excluded only for a documented instrument reason.
Accuracy denominators use included=True rows only.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(sys.maxsize)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.variants import normalize_variant  # noqa: E402
from probes.contamination.verify import (  # noqa: E402
    LAST_VERIFY_META,
    mystery_action_mapping,
    parse_action_mapping_from_notes,
    verify_answer,
)
from probes.contamination.verify_algo import LAST_VERIFY_META as ALGO_META  # noqa: E402
from probes.contamination.verify_algo import verify_algo  # noqa: E402

RAW_DIR = REPO_ROOT / "results/raw"
DERIVED_DIR = REPO_ROOT / "results/derived"

BANKS = {
    "BW": REPO_ROOT / "data/problems/question_bank_bw.csv",
    "ALGO": REPO_ROOT / "data/problems/question_bank_algo.csv",
    "GSM": REPO_ROOT / "data/problems/question_bank_gsm.csv",
}

_ALGO_SUBTYPES = {"coin_change", "shortest_path", "wis", "wis_independent_set"}
_VERIFIER_FAMILIES = {
    "blocksworld",
    "mystery_blocksworld",
    "logistics",
    "arithmetic_reasoning",
    "shortest_path",
    "weighted_interval_scheduling",
    "coin_change",
    "knapsack",
    "gsm",
}

P1_PATTERNS = (
    "BW_P1_behavioral*.csv",
    "GSM_P1_behavioral_*.csv",
    "ALGO_P1_behavioral_*.csv",
)

# Fixed exclusion vocabulary. in_bank_ok marks included rows.
REASON_IN_BANK_OK = "in_bank_ok"
REASON_MISSING_BANK = "missing_bank_row"
REASON_API_ERROR = "api_error"
REASON_PARSE_FAILED = "parse_failed"
REASON_UNANSWERABLE = "unanswerable_prompt"
REASON_MOCK = "mock_row"
REASON_INVALID_GOLD = "invalid_gold"

MOCK_MODELS = {"mock", "the answer is 42."}


def _resolve_verifier_family(*, pid: str, problem_family: str, problem_subtype: str) -> str:
    fam = str(problem_family or "").strip().lower()
    sub = str(problem_subtype or "").strip().lower()
    if sub in _VERIFIER_FAMILIES:
        return sub
    if fam in _VERIFIER_FAMILIES:
        return fam
    if fam == "planning_suite":
        pid_up = str(pid or "").strip().upper()
        if pid_up.startswith("MBW_"):
            return "mystery_blocksworld"
        if pid_up.startswith("BW_"):
            return "blocksworld"
        if pid_up.startswith("LOG_"):
            return "logistics"
        return "blocksworld"
    return fam


def _load_bank_index(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    index: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            str(row.get("problem_id", "")).strip(),
            normalize_variant(row.get("variant_type", "")),
        )
        index[key] = row
    return index


def _family_from_filename(name: str) -> str:
    if name.startswith("ALGO_"):
        return "ALGO"
    if name.startswith("GSM_"):
        return "GSM"
    return "BW"


def _parse_bool(value: object) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _model_answer(row: dict[str, str]) -> str:
    if "raw_response" in row:
        return str(row.get("raw_response") or "")
    return str(row.get("model_answer") or "")


def _old_correct(row: dict[str, str]) -> bool | None:
    if "behavioral_correct" in row:
        return _parse_bool(row.get("behavioral_correct"))
    if "verified" in row:
        return _parse_bool(row.get("verified"))
    return None


def _is_mock_model(model: str) -> bool:
    return str(model or "").strip().lower() in MOCK_MODELS


def _is_api_error(answer: str) -> bool:
    return str(answer or "").strip().startswith("ERROR:")


def _family_verify_method(row_family: str, vf: str, last: str) -> str:
    if row_family == "GSM" or vf in {"gsm", "arithmetic_reasoning"}:
        return "numeric"
    if row_family == "ALGO" or vf in _ALGO_SUBTYPES or vf == "weighted_interval_scheduling":
        return last or "algo_strict"
    return last or ""


def _rescore_included(
    family: str,
    raw: dict[str, str],
    bank_row: dict[str, str],
) -> tuple[bool | None, str, str, str]:
    """Return (correct, verify_method, rescore_reason, exclusion_or_ok)."""
    answer = _model_answer(raw)
    pid = str(raw.get("problem_id") or bank_row.get("problem_id") or "").strip()
    variant = normalize_variant(raw.get("variant_type") or bank_row.get("variant_type"))
    subtype = str(bank_row.get("problem_subtype", "")).strip()
    try:
        if family == "ALGO" or subtype.lower() in _ALGO_SUBTYPES:
            ok, reason, meta = verify_algo(
                pid,
                answer,
                bank_row.get("correct_answer", ""),
                subtype,
                variant,
                bank_row.get("difficulty_params", ""),
                notes=bank_row.get("notes"),
                problem_text=bank_row.get("problem_text"),
            )
            method = _family_verify_method(family, subtype.lower(), "algo_strict")
            parse_status = str((meta or {}).get("parse_status") or ALGO_META.get("parse_status") or "")
            if parse_status == "parse_failed":
                return None, method, reason, REASON_PARSE_FAILED
            return bool(ok), method, reason, REASON_IN_BANK_OK
        mapping = parse_action_mapping_from_notes(bank_row.get("notes"))
        vf = _resolve_verifier_family(
            pid=pid,
            problem_family=str(bank_row.get("problem_family", "")),
            problem_subtype=subtype,
        )
        if vf == "mystery_blocksworld":
            mapping = mystery_action_mapping(
                bank_row.get("notes"), bank_row.get("problem_text"), explicit=mapping
            )
        ok = verify_answer(
            pid,
            answer,
            bank_row.get("correct_answer", ""),
            vf,
            problem_text=bank_row.get("problem_text", ""),
            action_mapping=mapping,
        )
        method = _family_verify_method(
            family, vf, str(LAST_VERIFY_META.get("verify_method") or "")
        )
        if ok is None:
            return None, method, "unparsed_state", REASON_PARSE_FAILED
        if ok is True:
            return True, method, "correct", REASON_IN_BANK_OK
        return False, method, "incorrect", REASON_IN_BANK_OK
    except Exception as exc:
        return None, "", f"{type(exc).__name__}: {exc}", REASON_PARSE_FAILED


def _load_invalid_gold() -> set[tuple[str, str]]:
    path = REPO_ROOT / "data/problems/mystery_invalid_gold.csv"
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {
            (str(r.get("problem_id", "")).strip(), normalize_variant(r.get("variant_type")))
            for r in csv.DictReader(f)
        }


def _p1_files(raw_dir: Path) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in P1_PATTERNS:
        for path in sorted(raw_dir.glob(pattern)):
            if path in seen or "review" in path.name.lower():
                continue
            seen.add(path)
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DERIVED_DIR)
    args = parser.parse_args()

    raw_dir = args.raw_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    banks = {name: _load_bank_index(path) for name, path in BANKS.items()}
    invalid_gold = _load_invalid_gold()
    summary: dict[tuple[str, str, str, str], dict[str, int]] = defaultdict(
        lambda: {"n": 0, "old_correct": 0, "new_correct": 0, "changed": 0}
    )
    coverage: dict[str, int] = defaultdict(int)
    written: list[Path] = []

    for path in _p1_files(raw_dir):
        family = _family_from_filename(path.name)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            src_fields = list(reader.fieldnames or [])
            rows = list(reader)

        extra = [
            col
            for col in (
                "variant_type_normalized",
                "included",
                "exclusion_reason",
                "old_verified",
                "rescored_correct",
                "verify_method",
                "rescore_reason",
                "verdict_changed",
            )
            if col not in src_fields
        ]
        out_fields = src_fields + extra
        out_path = out_dir / f"{path.stem}_rescored.csv"

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
            writer.writeheader()
            for raw in rows:
                pid = str(raw.get("problem_id", "")).strip()
                variant_raw = str(raw.get("variant_type", "")).strip()
                variant = normalize_variant(variant_raw)
                model = str(raw.get("model") or "").strip() or path.stem
                answer = _model_answer(raw)
                old = _old_correct(raw)

                bank_row = None
                row_family = family
                key = (pid, variant)
                for name, index in banks.items():
                    if key in index:
                        bank_row = index[key]
                        row_family = name
                        break

                if _is_mock_model(model) or variant == "MOCK":
                    included = False
                    reason = REASON_MOCK
                    new = None
                    method = ""
                    detail = "mock_row"
                elif bank_row is None:
                    included = False
                    reason = REASON_MISSING_BANK
                    new = None
                    method = ""
                    detail = "missing_bank_row"
                elif (pid, variant) in invalid_gold:
                    included = False
                    reason = REASON_INVALID_GOLD
                    new = None
                    method = ""
                    detail = "invalid_gold"
                elif _is_api_error(answer):
                    included = False
                    reason = REASON_API_ERROR
                    new = None
                    method = ""
                    detail = "api_error"
                else:
                    new, method, detail, reason = _rescore_included(
                        row_family, raw, bank_row
                    )
                    included = reason == REASON_IN_BANK_OK

                coverage[reason] += 1
                new_bool = new is True if included else None
                changed = (
                    included
                    and old is not None
                    and new is not None
                    and old != new
                )
                subtype = str(
                    (bank_row or {}).get("problem_subtype")
                    or raw.get("problem_family")
                    or family
                )
                if included:
                    stats = summary[(path.name, model, subtype, variant)]
                    stats["n"] += 1
                    if old is True:
                        stats["old_correct"] += 1
                    if new is True:
                        stats["new_correct"] += 1
                    if changed:
                        stats["changed"] += 1

                out_row = dict(raw)
                out_row["variant_type_normalized"] = variant
                out_row["included"] = included
                out_row["exclusion_reason"] = reason
                if included:
                    if "behavioral_correct" in out_row:
                        out_row["behavioral_correct"] = bool(new)
                    if "verified" in out_row:
                        out_row["verified"] = bool(new)
                    out_row["rescored_correct"] = bool(new)
                else:
                    out_row["rescored_correct"] = ""
                out_row["old_verified"] = "" if old is None else old
                out_row["verify_method"] = method
                out_row["rescore_reason"] = detail
                out_row["verdict_changed"] = changed if included else ""
                writer.writerow(out_row)

        written.append(out_path)
        print(f"Wrote {out_path} ({len(rows)} rows)")

    print()
    header = (
        f"{'source':<42} {'model':<36} {'family/subtype':<24} {'variant':<12} "
        f"{'old_acc':>8} {'new_acc':>8} {'n':>5} {'changed':>8}"
    )
    print(header)
    print("-" * len(header))
    summary_path = out_dir / "P1_rescore_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_file",
                "model",
                "family",
                "variant",
                "old_accuracy",
                "new_accuracy",
                "n",
                "n_changed",
            ],
        )
        writer.writeheader()
        for (source, model, fam, variant), stats in sorted(summary.items()):
            n = stats["n"]
            old_acc = stats["old_correct"] / n if n else 0.0
            new_acc = stats["new_correct"] / n if n else 0.0
            print(
                f"{source:<42} {model:<36} {fam:<24} {variant:<12} "
                f"{old_acc:8.3f} {new_acc:8.3f} {n:5d} {stats['changed']:8d}"
            )
            writer.writerow(
                {
                    "source_file": source,
                    "model": model,
                    "family": fam,
                    "variant": variant,
                    "old_accuracy": f"{old_acc:.4f}",
                    "new_accuracy": f"{new_acc:.4f}",
                    "n": n,
                    "n_changed": stats["changed"],
                }
            )
    print(f"\nSummary: {summary_path}")

    coverage_path = out_dir / "P1_rescore_coverage.csv"
    with coverage_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["exclusion_reason", "n", "included"]
        )
        writer.writeheader()
        print("\nCoverage:")
        for reason, n in sorted(coverage.items()):
            included = reason == REASON_IN_BANK_OK
            print(f"  {reason}: {n} (included={included})")
            writer.writerow(
                {
                    "exclusion_reason": reason,
                    "n": n,
                    "included": included,
                }
            )
    print(f"Coverage: {coverage_path}")
    print(f"Rescored files: {len(written)}")


if __name__ == "__main__":
    main()
