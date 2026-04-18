"""
CCI measures whether the model's execution follows its
own plan. Low CCI indicates plan confabulation. Pilot result: CCI=0.26 on 7
Blocksworld instances (GPT-4o). Only applies to Probe 2 Blocksworld instances.
"""

from __future__ import annotations


def compute_cci(
    problem_id: str,
    generated_plan: list[str],
    executed_steps: list[str],
) -> dict:
    gen_len = len(generated_plan) if generated_plan else 0
    exec_len = len(executed_steps) if executed_steps else 0
    length_mismatch = abs(gen_len - exec_len) > 1

    if not generated_plan or not executed_steps:
        return {
            "problem_id": problem_id,
            "cci": None,
            "matched_steps": 0,
            "total_steps_compared": 0,
            "generated_plan_length": gen_len,
            "executed_steps_length": exec_len,
            "length_mismatch": length_mismatch,
        }

    # Normalise both lists: strip whitespace, lowercase each move
    gen_norm = [s.strip().lower() for s in generated_plan]
    exec_norm = [s.strip().lower() for s in executed_steps]

    total_steps_compared = min(len(gen_norm), len(exec_norm))
    matched_steps = 0

    # Align by position and count matches
    for i in range(total_steps_compared):
        if gen_norm[i] == exec_norm[i]:
            matched_steps += 1

    cci = round(matched_steps / total_steps_compared, 4)

    return {
        "problem_id": problem_id,
        "cci": cci,
        "matched_steps": matched_steps,
        "total_steps_compared": total_steps_compared,
        "generated_plan_length": gen_len,
        "executed_steps_length": exec_len,
        "length_mismatch": length_mismatch,
    }
