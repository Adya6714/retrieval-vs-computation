"""BW Phase 2 session execution helpers.

This module protects Phase 2 runs from repeating format-error loops by
auto-skipping after two consecutive identical parse errors.
"""

from __future__ import annotations

import re
from typing import Any


def parse_plan(plan: Any) -> list[str]:
    """Parse a free-form plan into normalized step strings."""
    if isinstance(plan, list):
        return [str(step).strip().lower() for step in plan if str(step).strip()]

    if not plan:
        return []

    steps: list[str] = []
    for line in str(plan).splitlines():
        line = line.strip()
        if not line:
            continue
        line = re.sub(r"^\d+[\.\)\:]\s*", "", line)
        line = re.sub(r"^step\s+\d+[\.\:\)]?\s*", "", line, flags=re.IGNORECASE)
        line = line.strip().lower()
        if line:
            steps.append(line)
    return steps


def parse_bw_action(response: Any) -> dict[str, str]:
    """Parse a BW action from model output into structured status."""
    if isinstance(response, dict):
        response = response.get("response", "")

    text = str(response or "").strip().lower()
    if not text:
        return {"status": "error", "error_type": "empty_output"}
    if "double_pickup_illegal" in text:
        return {"status": "error", "error_type": "double_pickup_illegal"}

    verbs = ("pick-up", "put-down", "stack", "unstack")
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        candidate = re.sub(r"^\d+[\.\)\:]\s*", "", candidate)
        candidate = re.sub(
            r"^step\s+\d+[\.\:\)]?\s*", "", candidate, flags=re.IGNORECASE
        ).strip()
        if any(candidate.startswith(verb) for verb in verbs):
            return {"status": "ok", "action": candidate}
    return {"status": "error", "error_type": "format_error"}


def validate_plan_single_arm(plan: Any) -> tuple[bool, str | None]:
    """Validate simple BW single-arm consistency for pick-up/put-down actions."""
    steps = parse_plan(plan)
    held_block: str | None = None

    for step in steps:
        parts = step.split()
        if len(parts) < 2:
            continue
        verb = parts[0]
        obj = parts[1]

        if verb == "pick-up":
            if held_block is not None:
                return (
                    False,
                    f"double_pickup: picked up {obj} while {held_block} already held",
                )
            held_block = obj
        elif verb == "put-down":
            held_block = None
        elif verb == "stack":
            held_block = None
        elif verb == "unstack":
            # Unstack implies picking up an item from another block.
            if held_block is not None:
                return (
                    False,
                    f"double_pickup: picked up {obj} while {held_block} already held",
                )
            held_block = obj

    return True, None


def compute_tep(
    non_injected_steps: list[str], injected_steps: list[str], injection_step_idx: int
) -> float | None:
    """Compute TEP over post-injection steps only."""
    post_non = non_injected_steps[injection_step_idx + 1 :]
    post_inj = injected_steps[injection_step_idx + 1 :]
    compared = min(len(post_non), len(post_inj))
    if compared == 0:
        return None
    changed = sum(1 for i in range(compared) if post_non[i] != post_inj[i])
    return changed / compared


def run_phase2_session(plan: Any, instance: Any, model_client: Any) -> dict[str, Any]:
    """Execute one Phase 2 session with loop protection for repeated errors."""
    steps = parse_plan(plan)
    if not steps:
        return {"status": "skipped: empty_plan", "log": []}

    execution_log: list[dict[str, Any]] = []
    error_count = 0
    last_error = None
    skip_count = 0
    force_skip_next = False

    for step_idx, planned_step in enumerate(steps):
        if force_skip_next:
            execution_log.append(
                {
                    "step_idx": step_idx,
                    "planned": planned_step,
                    "executed": "STEP_SKIP",
                    "status": "illegal_both",
                    "note": "auto-skipped after double_pickup_illegal",
                }
            )
            skip_count += 1
            force_skip_next = False
            if skip_count > 5:
                return {
                    "status": "aborted: excessive illegal steps",
                    "log": execution_log,
                }
            continue

        response = model_client.execute_step(planned_step, instance)
        parse_result = parse_bw_action(response)

        if parse_result["status"] == "error":
            current_error = parse_result["error_type"]

            if current_error == "double_pickup_illegal":
                execution_log.append(
                    {
                        "step_idx": step_idx,
                        "planned": planned_step,
                        "executed": "DOUBLE_PICKUP_ILLEGAL",
                        "status": "illegal_both",
                        "note": "single-arm violation detected",
                    }
                )
                force_skip_next = True
                continue

            if current_error == last_error:
                error_count += 1
            else:
                error_count = 1
                last_error = current_error

            if error_count >= 2:
                execution_log.append(
                    {
                        "step_idx": step_idx,
                        "planned": planned_step,
                        "executed": "STEP_SKIP",
                        "status": "illegal_both",
                        "note": f"auto-skipped after 2x consecutive {current_error}",
                    }
                )
                skip_count += 1
                error_count = 0
                last_error = None

                if skip_count > 5:
                    return {
                        "status": "aborted: excessive illegal steps",
                        "log": execution_log,
                    }
                continue

            execution_log.append(
                {
                    "step_idx": step_idx,
                    "planned": planned_step,
                    "executed": "PARSE_ERROR",
                    "status": "error",
                    "note": current_error,
                }
            )
            continue

        error_count = 0
        last_error = None
        execution_log.append(
            {
                "step_idx": step_idx,
                "planned": planned_step,
                "executed": parse_result["action"],
                "status": "ok",
            }
        )

    return {"status": "complete", "log": execution_log}


def compute_cci(session_log: dict[str, Any], exclude_step_skip: bool = True) -> float | None:
    """Compute CCI from a Phase 2 session log.

    Aborted sessions and STEP_SKIP rows are excluded by design.
    """
    if session_log["status"] in (
        "aborted: excessive illegal steps",
        "skipped: empty_plan",
    ):
        return None

    excluded = {"DOUBLE_PICKUP_ILLEGAL"}
    if exclude_step_skip:
        excluded.add("STEP_SKIP")
    valid_steps = [s for s in session_log["log"] if s["executed"] not in excluded]
    matching = sum(1 for s in valid_steps if s["planned"] == s["executed"])
    return matching / len(valid_steps) if valid_steps else None
