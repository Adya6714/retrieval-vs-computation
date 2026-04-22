"""
RCS is reported separately from CSS because W5 changes
the correct answer. Only applicable to Blocksworld, Shortest Path, Coin Change.
"""

from __future__ import annotations

from probes.contamination.verify import verify_answer


def compute_rcs(
    problem_id: str,
    w5_model_answer: str,
    w5_correct_answer: str,
    family: str,
) -> dict:
    valid_families = {"blocksworld", "shortest_path", "coin_change"}
    
    if family not in valid_families:
        raise ValueError(f"W5 reversal is not defined for family: {family}")

    w5_correct = bool(
        verify_answer(problem_id, w5_model_answer, w5_correct_answer, family)
    )

    return {
        "problem_id": problem_id,
        "family": family,
        "w5_correct": w5_correct,
        "model_answer": w5_model_answer,
        "correct_answer": w5_correct_answer,
    }
