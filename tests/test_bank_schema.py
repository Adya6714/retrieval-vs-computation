"""Schema guards for question-bank CSVs (C1 hygiene)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ALGO_BANK = REPO / "data/problems/question_bank_algo.csv"

ALLOWED_VERIFIER_BY_SUBTYPE = {
    "coin_change": {"verify_coinchange"},
    "shortest_path": {"verify_sp"},
    "wis": {"verify_wis"},
}


def test_algo_verifier_function_one_per_subtype():
    assert ALGO_BANK.exists(), f"missing {ALGO_BANK}"
    with ALGO_BANK.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "empty ALGO bank"
    bad: list[str] = []
    for i, row in enumerate(rows, start=2):
        sub = str(row.get("problem_subtype") or "").strip()
        vf = str(row.get("verifier_function") or "").strip()
        allowed = ALLOWED_VERIFIER_BY_SUBTYPE.get(sub)
        if allowed is None:
            bad.append(f"line {i}: unknown subtype {sub!r}")
            continue
        if vf not in allowed:
            bad.append(
                f"line {i} {row.get('problem_id')}/{row.get('variant_type')}: "
                f"verifier_function={vf!r} not in {sorted(allowed)} for {sub}"
            )
    assert not bad, "verifier_function schema violations:\n" + "\n".join(bad[:20])
