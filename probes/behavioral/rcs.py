"""
RCS is reported separately from CSS because W6 changes
the correct answer. Only applicable to Blocksworld, Shortest Path, Coin Change.
"""

from __future__ import annotations

from probes.contamination.verify import verify_answer


def compute_rcs(
    problem_id: str,
    w6_model_answer: str,
    w6_correct_answer: str,
    family: str,
) -> dict:
    valid_families = {"blocksworld", "shortest_path", "coin_change"}
    
    if family not in valid_families:
        raise ValueError(f"W6 reversal is not defined for family: {family}")

    w6_correct = bool(
        verify_answer(problem_id, w6_model_answer, w6_correct_answer, family)
    )

    return {
        "problem_id": problem_id,
        "family": family,
        "w6_correct": w6_correct,
        "model_answer": w6_model_answer,
        "correct_answer": w6_correct_answer,
    }
