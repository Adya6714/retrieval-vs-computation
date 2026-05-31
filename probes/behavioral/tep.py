"""
TEP measures state-reactivity after mid-execution
corruption. High TEP alone is not diagnostic — it is meaningful only when
paired with CCI. Pilot result: TEP=0.70 on 7 Blocksworld instances (GPT-4o).
"""

from __future__ import annotations


def compute_tep(
    problem_id: str,
    pre_corruption_steps: list[str],
    post_corruption_steps: list[str],
    corrupted_state: str,
    expected_adapted_steps: list[str],
    original_plan: list[str] | None = None,
) -> dict:
    # 1. Normalise all move lists
    post_norm = [s.strip().lower() for s in post_corruption_steps]
    expected_norm = [s.strip().lower() for s in expected_adapted_steps]

    # 2. Compute adaptation_score
    adapt_compared = min(len(post_norm), len(expected_norm))
    if adapt_compared == 0:
        adaptation_score = None
        tep = None
    else:
        adapt_matches = 0
        for i in range(adapt_compared):
            if post_norm[i] == expected_norm[i]:
                adapt_matches += 1
        adaptation_score = round(adapt_matches / adapt_compared, 4)
        tep = adaptation_score

    # 3. Compute continuation_score (if original_plan is provided)
    continuation_score = None
    if original_plan is not None:
        orig_norm = [s.strip().lower() for s in original_plan]
        offset = len(pre_corruption_steps)
        orig_remaining = orig_norm[offset:]

        cont_compared = min(len(post_norm), len(orig_remaining))
        if cont_compared > 0:
            cont_matches = 0
            for i in range(cont_compared):
                if post_norm[i] == orig_remaining[i]:
                    cont_matches += 1
            continuation_score = round(cont_matches / cont_compared, 4)

    # 4 & 5. TEP classification
    if tep is not None:
        if tep >= 0.6:
            interpretation = "state_reactive"
        elif tep < 0.4:
            interpretation = "plan_locked"
        else:
            interpretation = "ambiguous"
    else:
        # Fallback interpretation if no post steps
        interpretation = "insufficient_data"

    return {
        "problem_id": problem_id,
        "tep": tep,
        "adaptation_score": adaptation_score,
        "continuation_score": continuation_score,
        "post_corruption_steps_count": len(post_norm),
        "corrupted_state": corrupted_state,
        "interpretation": interpretation,
    }
