"""Tests for the v1.1 pipeline. Run: PYTHONPATH=. pytest tests_v11 -q"""

import random

import pytest

from rvc11 import algo, arith, bank, experiments, planning, prompts, qa, verify


@pytest.fixture(scope="module")
def items():
    bank.build()
    return bank.load()


def test_all_qa_gates_pass(items):
    assert qa.run_all()


def test_strip_instructions_removes_format_and_verbosity():
    t, tail = bank.strip_instructions('"Compute 2+3. Give just the final answer without units and nothing else."')
    assert t == "Compute 2+3." and "final answer" in tail
    t, tail = bank.strip_instructions("Goal: a on b. Respond with a numbered list of actions only. No explanation.")
    assert t == "Goal: a on b." and tail.startswith("Respond")
    assert bank.strip_instructions("Plain problem with no instruction?")[1] == ""


def test_planner_and_simulator_agree_on_random_instances():
    rnd = random.Random(7)
    for _ in range(200):
        blocks = [chr(97 + i) for i in range(rnd.randint(2, 8))]

        def towers():
            order, on, prev = blocks[:], set(), None
            rnd.shuffle(order)
            for b in order:
                if prev and rnd.random() < 0.6:
                    on.add((b, prev))
                prev = b
            return on

        init = planning.close_state(towers(), set(blocks))
        goal = {f for f in planning.close_state(towers(), set(blocks)) if f[0] == "on"}
        assert planning.state_is_physical(init)[0]
        assert planning.simulate(init, goal, planning.reference_plan(init, goal))[0]


def test_plan_parser_is_strict_and_handles_renamed_actions():
    v = planning.Vocab({"pick-up": "select X", "put-down": "release X", "stack": "place X under Y", "unstack": "remove X from Y"},
                       {"a": "Alice", "b": "Bob"})
    assert v.parse_line("2) Place Alice under Bob.") == ("stack", ("a", "b"))
    assert v.parse_line("stack Alice Bob") is None
    assert planning.parse_plan("1. select Alice\nthen I am done", v)[1] == "malformed_line"


def test_illegal_plan_step_is_rejected(items):
    it = next(i for i in items if i["item_id"] == "BW_001::canonical")
    lines = it["gold_block"].splitlines()
    assert verify.score("FINAL ANSWER:\n" + "\n".join(lines), it).correct
    assert not verify.score("FINAL ANSWER:\n" + "\n".join(lines[1:]), it).correct       # skips a pick-up
    assert not verify.score("FINAL ANSWER:\n" + "\n".join(lines[:-2]), it).correct      # goal not reached


def test_alternative_optimal_answers_are_accepted():
    sp = {"kind": "sp", "directed": True, "source": 0, "target": 3, "edges": [[0, 1, 2], [1, 3, 3], [0, 2, 3], [2, 3, 2]],
          "labels": {"0": ["0"], "1": ["1"], "2": ["2"], "3": ["3"]}}
    assert algo.verify_sp("Path: 0 -> 1 -> 3\nCost: 5", sp)[0] and algo.verify_sp("Path: 0 -> 2 -> 3\nCost: 5", sp)[0]
    assert not algo.verify_sp("Cost: 5", sp)[0]                       # cost alone is not an answer
    assert not algo.verify_sp("Path: 0 -> 3\nCost: 5", sp)[0]          # edge does not exist
    cc = {"kind": "cc", "denominations": [1, 3, 4], "target": 6}
    assert algo.verify_cc("Count: 2\nValues: 3, 3", cc)[0]
    assert not algo.verify_cc("Count: 2", cc)[0] and not algo.verify_cc("Count: 3\nValues: 4, 1, 1", cc)[0]


def test_numeric_parsing():
    assert arith.verify_numeric("$1,234.50", {"answer": 1234.5})[0]
    assert arith.verify_numeric("3 trips of 120", {"answer": 3})[1] == "format_fail"
    assert arith.numbers_in("denominations: [1,3,4,7], target 6") == [1.0, 3.0, 4.0, 7.0, 6.0]


def test_marker_extraction_takes_last_marker():
    block, marked = verify.extract_block("My FINAL ANSWER: 4 was wrong.\n**Final Answer:** 5")
    assert marked and block == "5"


def test_prompts_of_one_problem_differ_only_in_task_text(items):
    by = {i["item_id"]: i for i in items}
    a, b = by["GSM_001::canonical"], by["GSM_001::W3"]
    for ins in prompts.INSTRUCTIONS:
        assert prompts.build_prompt(a, ins).replace(a["task_text"], "") == prompts.build_prompt(b, ins).replace(b["task_text"], "")


def test_call_ids_are_stable_and_unique(items):
    from rvc11 import config

    plan = experiments.build_plan("p1_surface", config.load_study(), config.load_models(), items, ["mock_oracle"], limit=2)
    ids = [c["call_id"] for c in plan]
    assert len(ids) == len(set(ids))
    assert ids == [c["call_id"] for c in experiments.build_plan("p1_surface", config.load_study(), config.load_models(), items, ["mock_oracle"], limit=2)]
