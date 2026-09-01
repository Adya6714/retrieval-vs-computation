"""Gold-in-gold-out regression: each bank row's own correct_answer must verify True.

Feed every row in the three question banks back into the production verifier as
if it were a model response. A correct verifier returns True. Exceptions are
counted as failures (they change the denominator if left uncaught).

Parametrized by (family, problem_subtype, variant_type). Run with ``-s`` to
print the pass-count table.

Recorded baseline (commit this file before any verifier fix):
  BW blocksworld  W3: 25/50      BW mystery W3: 10/15   BW mystery W6: 10/15
  ALGO shortest_path W5: 0/50    ALGO wis W3: 25/30 (5 raise ValueError)
  GSM: all pass
  Everything not listed above currently passes.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest

from probes.common.variants import normalize_variant
from probes.contamination.verify import (
    mystery_action_mapping,
    parse_action_mapping_from_notes,
    verify_answer,
)
from probes.contamination.verify_algo import verify_algo

REPO_ROOT = Path(__file__).resolve().parents[1]

BANKS: dict[str, Path] = {
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


def _resolve_verifier_family(
    *,
    pid: str,
    problem_family: str,
    problem_subtype: str,
) -> str:
    """Match scripts/BW_P1_SCR_run_behavioral_sweep._resolve_verifier_family."""
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


def _load_bank(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def verify_gold_row(family: str, row: dict[str, str]) -> tuple[bool, str | None]:
    """Return (ok, error_type). error_type is set when the verifier raises."""
    pid = str(row.get("problem_id", "")).strip()
    subtype = str(row.get("problem_subtype", "")).strip()
    variant = str(row.get("variant_type", "")).strip()
    gold = row.get("correct_answer", "")
    text = row.get("problem_text", "")
    params = row.get("difficulty_params", "")
    try:
        if family == "ALGO" or subtype.lower() in _ALGO_SUBTYPES:
            ok, _reason, _meta = verify_algo(
                pid, gold, gold, subtype, variant, params
            )
            return bool(ok), None
        vf = _resolve_verifier_family(
            pid=pid,
            problem_family=str(row.get("problem_family", "")),
            problem_subtype=subtype,
        )
        action_mapping = parse_action_mapping_from_notes(row.get("notes"))
        if vf == "mystery_blocksworld":
            action_mapping = mystery_action_mapping(
                row.get("notes"), text, explicit=action_mapping
            )
        ok = verify_answer(
            pid,
            gold,
            gold,
            vf,
            problem_text=text,
            action_mapping=action_mapping,
        )
        return bool(ok), None
    except Exception as exc:
        return False, type(exc).__name__


def _invalid_gold_keys() -> set[tuple[str, str]]:
    path = REPO_ROOT / "data/problems/mystery_invalid_gold.csv"
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {
            (str(r.get("problem_id", "")).strip(), normalize_variant(r.get("variant_type")))
            for r in csv.DictReader(f)
        }


def collect_gold_roundtrip_counts() -> list[tuple[str, str, str, int, int, int, int]]:
    """Return (family, subtype, variant, n_pass, n_error, n_excluded, n_total)."""
    excluded_keys = _invalid_gold_keys()
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for family, path in BANKS.items():
        for row in _load_bank(path):
            key = (
                family,
                str(row.get("problem_subtype", "")).strip(),
                normalize_variant(row.get("variant_type", "")),
            )
            groups[key].append(row)

    table: list[tuple[str, str, str, int, int, int, int]] = []
    for key in sorted(groups):
        family, subtype, variant = key
        n_pass = n_error = n_excluded = 0
        rows = groups[key]
        for row in rows:
            pid = str(row.get("problem_id", "")).strip()
            vt = normalize_variant(row.get("variant_type", ""))
            if (pid, vt) in excluded_keys:
                n_excluded += 1
                continue
            ok, err = verify_gold_row(family, row)
            if err:
                n_error += 1
            elif ok:
                n_pass += 1
        table.append((family, subtype, variant, n_pass, n_error, n_excluded, len(rows)))
    return table


def format_gold_roundtrip_table(
    table: list[tuple[str, str, str, int, int, int, int]],
) -> str:
    header = (
        f"{'family':<6} {'subtype':<24} {'variant':<12} "
        f"{'pass/incl':>12} {'excl':>6} {'error':>8}"
    )
    lines = [header, "-" * len(header)]
    for family, subtype, variant, n_pass, n_error, n_excl, n_total in table:
        n_incl = n_total - n_excl
        lines.append(
            f"{family:<6} {subtype:<24} {variant:<12} "
            f"{n_pass:>4}/{n_incl:<4} {n_excl:>6} {n_error:>8}"
        )
    return "\n".join(lines)


_TABLE_CACHE: list[tuple[str, str, str, int, int, int, int]] | None = None


def _table() -> list[tuple[str, str, str, int, int, int, int]]:
    global _TABLE_CACHE
    if _TABLE_CACHE is None:
        _TABLE_CACHE = collect_gold_roundtrip_counts()
    return _TABLE_CACHE


def test_gold_roundtrip_prints_pass_table(capsys: pytest.CaptureFixture[str]) -> None:
    """Print the full (family, subtype, variant) pass-count table."""
    rendered = format_gold_roundtrip_table(_table())
    print("\n" + rendered)
    assert _table(), "question banks produced no gold-in-gold-out groups"


@pytest.mark.parametrize(
    "family, problem_subtype, variant_type",
    [
        (family, subtype, variant)
        for family, subtype, variant, _p, _e, _x, _n in _table()
    ],
)
def test_gold_in_gold_out(family: str, problem_subtype: str, variant_type: str) -> None:
    match = [
        row
        for row in _table()
        if row[0] == family and row[1] == problem_subtype and row[2] == variant_type
    ]
    assert match, f"missing group {(family, problem_subtype, variant_type)}"
    _fam, _sub, _var, n_pass, n_error, n_excl, n_total = match[0]
    n_incl = n_total - n_excl
    assert n_error == 0, (
        f"{family} {problem_subtype} {variant_type}: {n_error} raised"
    )
    assert n_pass == n_incl, (
        f"{family} {problem_subtype} {variant_type}: {n_pass}/{n_incl} included "
        f"({n_excl} excluded invalid_gold, n_total={n_total})"
    )


if __name__ == "__main__":
    print(format_gold_roundtrip_table(collect_gold_roundtrip_counts()))
