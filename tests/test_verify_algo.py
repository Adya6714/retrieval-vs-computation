from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probes.contamination.verify_algo import LAST_VERIFY_META, verify_algo


def _cc_params():
    return {"denominations": [1, 5, 6, 9], "target": 11, "greedy_succeeds": False}


def test_cc_correct_exact_match():
    ok, _, _meta = verify_algo(
        "CC_X",
        "Count: 2, Coins: [5, 6]",
        "Count: 2\nCoins: [5, 6]",
        "coin_change",
        "canonical",
        _cc_params(),
    )
    assert ok


def test_cc_correct_different_order():
    ok, _, _meta = verify_algo(
        "CC_X",
        "Count: 2, Coins: [6, 5]",
        "Count: 2\nCoins: [5, 6]",
        "coin_change",
        "canonical",
        _cc_params(),
    )
    assert ok


def test_cc_correct_different_decomposition_same_count():
    params = {"denominations": [1, 4, 5, 6], "target": 10, "greedy_succeeds": False}
    ok, _, _meta = verify_algo(
        "CC_Y",
        "Count: 2, Coins: [6, 4]",
        "Count: 2\nCoins: [5, 5]",
        "coin_change",
        "canonical",
        params,
    )
    assert ok


def test_cc_wrong_count():
    ok, reason, _meta = verify_algo(
        "CC_X",
        "Count: 3, Coins: [5, 5, 1]",
        "Count: 2\nCoins: [5, 6]",
        "coin_change",
        "canonical",
        _cc_params(),
    )
    assert not ok
    assert "wrong_count" in reason


def test_cc_greedy_wrong_adversarial():
    params = {"denominations": [1, 3, 4], "target": 6, "greedy_succeeds": False}
    ok, _, _meta = verify_algo(
        "CC_ADV",
        "Count: 3, Coins: [4, 1, 1]",
        "Count: 2\nCoins: [3, 3]",
        "coin_change",
        "canonical",
        params,
    )
    assert not ok


def test_cc_parse_failure():
    ok, reason, _meta = verify_algo(
        "CC_X",
        "the answer is seven",
        "Count: 2\nCoins: [5, 6]",
        "coin_change",
        "canonical",
        _cc_params(),
    )
    assert not ok
    assert reason.startswith("parse_failed:")
    assert LAST_VERIFY_META["parse_status"] == "parse_failed"


def test_cc_w3_scoops_correct():
    params = {
        "denominations": [10, 17],
        "target": 27,
        "greedy_succeeds": False,
        "scoop_names": {"scoop_A": 17, "scoop_B": 10},
    }
    ok, _, _meta = verify_algo(
        "CC_W3",
        "Total: 2, Scoops: [scoop_A, scoop_B]",
        "Count: 2\nCoins: [17, 10]",
        "coin_change",
        "W3",
        params,
    )
    assert ok


def _sp_params():
    return {
        "graph": [
            {"u": 0, "v": 1, "w": 1},
            {"u": 1, "v": 3, "w": 2},
            {"u": 0, "v": 2, "w": 1},
            {"u": 2, "v": 3, "w": 2},
        ],
        "directed": True,
        "source": 0,
        "target": 3,
    }


def test_sp_correct_same_path():
    ok, _, _meta = verify_algo(
        "SP_X",
        "Path: 0 → 1 → 3, Cost: 3",
        "Path: 0 → 1 → 3, Cost: 3",
        "shortest_path",
        "canonical",
        _sp_params(),
    )
    assert ok


def test_sp_correct_alternative_optimal_path():
    ok, _, _meta = verify_algo(
        "SP_X",
        "Path: 0 -> 2 -> 3 (cost: 3)",
        "Path: 0 → 1 → 3, Cost: 3",
        "shortest_path",
        "canonical",
        _sp_params(),
    )
    assert ok
    assert LAST_VERIFY_META["correct_alternative_path"] is True


def test_sp_wrong_cost():
    ok, reason, _meta = verify_algo(
        "SP_X",
        "Cost: 4",
        "Path: 0 → 1 → 3, Cost: 3",
        "shortest_path",
        "canonical",
        _sp_params(),
    )
    assert not ok
    assert "wrong_cost" in reason


def test_sp_valid_path_wrong_cost():
    ok, reason, _meta = verify_algo(
        "SP_X",
        "Path: 0 -> 1 -> 3, Cost: 4",
        "Path: 0 → 1 → 3, Cost: 3",
        "shortest_path",
        "canonical",
        _sp_params(),
    )
    assert not ok
    assert "path_cost_mismatch" in reason


def test_sp_parse_failure():
    ok, reason, _meta = verify_algo(
        "SP_X",
        "foobar route yes",
        "Path: 0 → 1 → 3, Cost: 3",
        "shortest_path",
        "canonical",
        _sp_params(),
    )
    assert not ok
    assert reason.startswith("parse_failed:")


def test_sp_w3_letter_labels_correct():
    params = dict(_sp_params())
    params["node_mapping"] = {"0": "Hub A", "1": "Hub B", "2": "Hub C", "3": "Hub D"}
    ok, _, _meta = verify_algo(
        "SP_X",
        "Path: Hub A → Hub C → Hub D, Cost: 3",
        "Path: 0 → 1 → 3, Cost: 3",
        "shortest_path",
        "W3",
        params,
    )
    assert ok


def _wis_params():
    return {
        "intervals": [
            [0, 3, 10],
            [3, 5, 6],
            [0, 5, 15],
            [5, 8, 8],
        ],
        "zero_indexed": True,
        "greedy_succeeds": False,
    }


def test_wis_correct_same_set():
    ok, _, _meta = verify_algo(
        "WIS_X",
        "Selected: {0, 1, 3}, Total: 24",
        "Selected: {0, 1, 3}, Total: 24",
        "wis",
        "canonical",
        _wis_params(),
    )
    assert ok


def test_wis_correct_alternative_optimal_set():
    params = {"intervals": [[0, 2, 10], [2, 4, 10], [0, 4, 20]], "zero_indexed": True, "greedy_succeeds": False}
    ok, _, _meta = verify_algo(
        "WIS_Y",
        "Selected: {2}, Total: 20",
        "Selected: {0, 1}, Total: 20",
        "wis",
        "canonical",
        params,
    )
    assert ok
    assert LAST_VERIFY_META["correct_alternative_set"] is True


def test_wis_non_independent_set():
    ok, reason, _meta = verify_algo(
        "WIS_X",
        "Selected: {0, 2}, Total: 24",
        "Selected: {0, 1, 3}, Total: 24",
        "wis",
        "canonical",
        _wis_params(),
    )
    assert not ok
    assert "overlap" in reason


def test_wis_wrong_total():
    ok, reason, _meta = verify_algo(
        "WIS_X",
        "Selected: {0, 1, 3}, Total: 23",
        "Selected: {0, 1, 3}, Total: 24",
        "wis",
        "canonical",
        _wis_params(),
    )
    assert not ok
    assert "wrong_total" in reason


def test_wis_w3_renamed_labels_correct():
    params = dict(_wis_params())
    params["item_mapping"] = {"0": "Space A", "1": "Space B", "2": "Space C", "3": "Space D"}
    ok, _, _meta = verify_algo(
        "WIS_X",
        "Spaces: {Space A, Space B, Space D}, Total: 24",
        "Selected: {0, 1, 3}, Total: 24",
        "wis",
        "W3",
        params,
    )
    assert ok
