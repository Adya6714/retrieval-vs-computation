from probes.behavioral.phase2_bw import (
    compute_cci,
    compute_tep,
    run_phase2_session,
    validate_plan_single_arm,
)


class DummyClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def execute_step(self, planned_step, instance):
        _ = planned_step, instance
        response = self._responses[self.calls]
        self.calls += 1
        return response


def test_run_phase2_session_injects_step_skip_after_two_same_errors():
    client = DummyClient(
        [
            "not an action",
            "still not an action",
            "pick-up a",
        ]
    )
    plan = ["pick-up a", "stack a b", "pick-up a"]

    result = run_phase2_session(plan, instance={}, model_client=client)

    assert result["status"] == "complete"
    assert client.calls == 3
    assert result["log"][1]["executed"] == "STEP_SKIP"
    assert result["log"][1]["status"] == "illegal_both"
    assert "auto-skipped after 2x consecutive format_error" in result["log"][1]["note"]
    assert result["log"][2]["executed"] == "pick-up a"


def test_run_phase2_session_aborts_after_more_than_five_skips():
    responses = ["bad", "bad"] * 6
    plan = [f"step {i}" for i in range(12)]
    client = DummyClient(responses)

    result = run_phase2_session(plan, instance={}, model_client=client)

    assert result["status"] == "aborted: excessive illegal steps"
    assert sum(1 for row in result["log"] if row["executed"] == "STEP_SKIP") == 6


def test_compute_cci_excludes_step_skip_and_aborted_sessions():
    complete_session = {
        "status": "complete",
        "log": [
            {"planned": "pick-up a", "executed": "pick-up a"},
            {"planned": "stack a b", "executed": "STEP_SKIP"},
            {"planned": "put-down a", "executed": "put-down c"},
        ],
    }
    aborted_session = {"status": "aborted: excessive illegal steps", "log": []}

    assert compute_cci(complete_session) == 0.5
    assert compute_cci(aborted_session) is None


def test_double_pickup_illegal_detection():
    plan = ["pick-up A", "pick-up B", "stack B A", "put-down A"]
    client = DummyClient(["pick-up A", "DOUBLE_PICKUP_ILLEGAL", "stack b a"])

    result = run_phase2_session(plan, instance={}, model_client=client)

    assert result["status"] == "complete"
    assert result["log"][1]["status"] == "illegal_both"
    assert result["log"][1]["executed"] == "DOUBLE_PICKUP_ILLEGAL"
    assert result["log"][2]["executed"] == "STEP_SKIP"
    assert result["log"][3]["executed"] == "stack b a"
    # CCI denominator excludes only the illegal action here: steps 1, 3, 4 -> N=3.
    assert compute_cci(result, exclude_step_skip=False) == (1 / 3)


def test_plan_validity_checker():
    plan_text = "1. pick-up A\n2. pick-up B\n3. stack B A"

    valid, reason = validate_plan_single_arm(plan_text)

    assert valid is False
    assert reason == "double_pickup: picked up b while a already held"


def test_injection_at_step_n():
    non_injected_steps = [
        "pick-up a",
        "stack a b",
        "put-down a",
        "pick-up c",
        "stack c a",
    ]
    injected_steps = [
        "pick-up a",
        "stack a b",
        "INJECTED_FALSE_STATE",
        "put-down b",
        "pick-up d",
    ]
    injection_step_idx = 2

    tep = compute_tep(non_injected_steps, injected_steps, injection_step_idx)
    cci = compute_cci(
        {
            "status": "complete",
            "log": [{"planned": s, "executed": s} for s in non_injected_steps],
        }
    )

    assert tep == 1.0
    assert cci == 1.0


def test_empty_plan_handling():
    client = DummyClient([])
    result = run_phase2_session([], instance={}, model_client=client)

    assert result["status"] == "skipped: empty_plan"
    assert compute_cci(result) is None
    assert compute_tep([], [], 0) is None


def test_real_bw_plan_sample():
    plan = [
        "unstack C from A",
        "put-down C",
        "unstack A from B",
        "put-down A",
        "pick-up B",
        "stack B on C",
        "pick-up A",
        "stack A on B",
    ]
    non_injected = [s.lower() for s in plan]
    injected = [
        "unstack c from a",
        "put-down c",
        "unstack a from b",
        "put-down a",
        "pick-up d",
        "stack d on e",
        "pick-up f",
        "stack f on d",
    ]

    tep = compute_tep(non_injected, injected, injection_step_idx=3)
    cci = compute_cci(
        {
            "status": "complete",
            "log": [{"planned": s, "executed": s} for s in non_injected],
        }
    )

    assert tep == 1.0
    assert cci == 1.0


