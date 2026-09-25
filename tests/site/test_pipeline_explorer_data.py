"""Guards the website's Pipeline Explorer data against drifting from the banks."""
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "site/data/pipeline_explorer.json"
BANKS = {f: ROOT / f"data/problems/question_bank_{f.lower()}.csv" for f in ["GSM", "BW", "ALGO"]}


@pytest.fixture(scope="module")
def payload():
    if not DATA.exists():
        pytest.skip("run scripts/site/build_pipeline_explorer_data.py first")
    return json.loads(DATA.read_text())


def test_items_present(payload):
    fams = {i["family"] for i in payload["items"]}
    assert {"GSM", "BW", "ALGO"} <= fams


def test_text_and_gold_match_bank(payload):
    for item in payload["items"]:
        bank = pd.read_csv(BANKS[item["family"]])
        for v in item["variants"]:
            row = bank[(bank.problem_id == item["problem_id"]) & (bank.variant_type == v["variant"])]
            assert len(row) == 1, (item["problem_id"], v["variant"])
            assert row.iloc[0]["problem_text"] == v["text"]
            assert str(row.iloc[0]["correct_answer"]) == v["gold"]


def test_no_mock_models(payload):
    for item in payload["items"]:
        for v in item["variants"]:
            for r in v["probe1"]:
                assert "mock" not in r["model"].lower()


def test_gold_rule(payload):
    for item in payload["items"]:
        for v in item["variants"]:
            expected = "re-derived" if v["variant"] in ("W5", "W6") else "fixed"
            assert v["gold_rule"] == expected
