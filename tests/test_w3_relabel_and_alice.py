"""C2 guards: W3b relabel columns; no Alice-article W3 corruption."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.generation.utils.variant_utils import apply_mapping

REPO = Path(__file__).resolve().parents[1]
BANKS = [
    REPO / "data/problems/question_bank_gsm.csv",
    REPO / "data/problems/question_bank_bw.csv",
    REPO / "data/problems/question_bank_algo.csv",
]


@pytest.mark.parametrize("path", BANKS, ids=["gsm", "bw", "algo"])
def test_w3_rows_carry_w3b_isomorph_labels(path: Path):
    assert path.exists()
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert "variant_subtype" in (rows[0] if rows else {})
    assert "w3_kind" in (rows[0] if rows else {})
    w3 = [r for r in rows if str(r.get("variant_type") or "").strip() == "W3"]
    assert w3, f"no W3 rows in {path.name}"
    for r in w3:
        assert r.get("variant_subtype") == "W3b", (path.name, r.get("problem_id"))
        assert r.get("w3_kind") == "isomorph", (path.name, r.get("problem_id"))


def test_bw_w3_has_no_alice_robot_arm_article_bug():
    path = REPO / "data/problems/question_bank_bw.csv"
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    bad = [
        r["problem_id"]
        for r in rows
        if str(r.get("variant_type")) == "W3"
        and "Alice robot arm" in (r.get("problem_text") or "")
    ]
    assert not bad, f"article-collision W3 rows remain: {bad}"


def test_apply_mapping_preserves_article_but_renames_block_a():
    mapping = {"a": "Alice", "pick-up": "recruit", "stack": "promote"}
    text = (
        "You are a robot arm. You can hold one block at a time. "
        "Current state: Blocks a and b are clear. "
        "pick-up a\nstack a b\nRespond with a numbered list."
    )
    out = apply_mapping(text, mapping)
    assert "You are a robot arm" in out
    assert "at a time" in out
    assert "with a numbered" in out
    assert "Blocks Alice" in out
    assert "recruit Alice" in out
    assert "promote Alice" in out
    assert "Alice robot arm" not in out
