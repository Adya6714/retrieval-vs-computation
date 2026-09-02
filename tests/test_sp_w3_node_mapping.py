"""SP W3 node_mapping recovery and Path-line preference."""

from __future__ import annotations

import csv
from pathlib import Path

from probes.contamination.verify import parse_entity_mapping_from_notes
from probes.contamination.verify_algo import (
    recover_sp_node_mapping,
    resolve_sp_node_mapping,
    verify_algo,
    verify_sp,
)

REPO = Path(__file__).resolve().parents[1]
BANK = REPO / "data/problems/question_bank_algo.csv"


def _sp_w3_rows() -> list[dict[str, str]]:
    with BANK.open(newline="", encoding="utf-8") as f:
        return [
            r
            for r in csv.DictReader(f)
            if str(r.get("problem_subtype", "")).strip() == "shortest_path"
            and str(r.get("variant_type", "")).strip().lower() == "w3"
        ]


def test_entity_mapping_persisted_in_notes_for_city_rows():
    rows = _sp_w3_rows()
    from_notes = sum(1 for r in rows if parse_entity_mapping_from_notes(r.get("notes")))
    assert from_notes == 50
    sp019 = next(r for r in rows if r["problem_id"] == "SP_019")
    mapping = parse_entity_mapping_from_notes(sp019["notes"])
    assert mapping is not None
    assert mapping["0"] == "Berlin"
    assert mapping["6"] == "Cologne"


def test_recover_sp_node_mapping_aligns_by_position_without_notes():
    rows = _sp_w3_rows()
    sp019 = next(r for r in rows if r["problem_id"] == "SP_019")
    recovered = recover_sp_node_mapping(sp019["problem_text"], sp019["difficulty_params"])
    assert recovered is not None
    notes_map = parse_entity_mapping_from_notes(sp019["notes"])
    assert recovered == notes_map


def test_resolve_prefers_notes_over_empty_params():
    rows = _sp_w3_rows()
    sp019 = next(r for r in rows if r["problem_id"] == "SP_019")
    resolved = resolve_sp_node_mapping(
        sp019["difficulty_params"],
        notes=sp019["notes"],
        problem_text=sp019["problem_text"],
    )
    assert resolved["0"] == "Berlin"


def test_verify_sp_prefers_final_path_line_over_cot_arrows():
    params = {
        "directed": True,
        "source": 0,
        "target": 3,
        "graph": [
            {"u": 0, "v": 1, "w": 1},
            {"u": 0, "v": 2, "w": 4},
            {"u": 1, "v": 2, "w": 1},
            {"u": 2, "v": 3, "w": 4},
        ],
        "node_mapping": {"0": "Berlin", "1": "Hamburg", "2": "Frankfurt", "3": "Cologne"},
    }
    cot = (
        "Berlin → Hamburg (1), Frankfurt (4). "
        "Berlin → Hamburg is not the destination. "
        "Path: Berlin -> Hamburg -> Frankfurt -> Cologne, Cost: 6"
    )
    ok, reason = verify_sp(cot, "Path: 0 -> 1 -> 2 -> 3, Cost: 6", params)
    assert ok, reason


def test_city_gold_verifies_as_path_not_cost_only():
    rows = _sp_w3_rows()
    sp019 = next(r for r in rows if r["problem_id"] == "SP_019")
    ok, reason, meta = verify_algo(
        "SP_019",
        sp019["correct_answer"],
        sp019["correct_answer"],
        "shortest_path",
        "W3",
        sp019["difficulty_params"],
        notes=sp019["notes"],
        problem_text=sp019["problem_text"],
    )
    assert ok, reason
    assert reason != "correct_cost_only"
    assert meta.get("path_provided") is True
