"""
CAS (Consistent Answer Signature) applies to hard-tier problems where the model
fails on all or most variants. It distinguishes structured failure (model fails
the same way across variants — consistent wrong answer) from noisy failure
(different wrong answer each time).

CAS is only meaningful on hard-tier instances and should not be computed on 
instances where the model is correct on most variants.
"""

from __future__ import annotations

from collections import Counter
from probes.common.parsers import extract_numeric, extract_path, extract_plan


def compute_cas(
    problem_id: str,
    variant_responses: list[dict],
    family: str,
) -> dict:
    # 1. Filter to only incorrect responses
    incorrect_responses = [resp for resp in variant_responses if not resp.get("is_correct", False)]
    n_incorrect = len(incorrect_responses)

    # 5. If fewer than 2 incorrect responses, return cas=None
    if n_incorrect < 2:
        return {
            "problem_id": problem_id,
            "family": family,
            "cas": None,
            "n_incorrect": n_incorrect,
            "most_common_wrong_answer": None,
            "failure_mode": "insufficient_data",
        }

    numeric_families = {"gsm", "coin_change", "knapsack", "weighted_interval_scheduling"}
    plan_families = {"blocksworld", "logistics", "mystery_blocksworld"}

    extracted_answers = []
    
    # 3. For each incorrect response, extract the model's answer
    for variant in incorrect_responses:
        ans = str(variant.get("model_answer", ""))
        
        extracted = None
        if family == "shortest_path":
            ca = str(variant.get("correct_answer", ""))
            # Differentiate between path sequence and path length numeric
            if "," in ca or "->" in ca:
                extracted = extract_path(ans)
            else:
                extracted = extract_numeric(ans)
        elif family in numeric_families:
            extracted = extract_numeric(ans)
        elif family in plan_families:
            extracted = extract_plan(ans)

        # Convert list to tuple to ensure hashability for collections.Counter
        if isinstance(extracted, list):
            extracted = tuple(extracted)
            
        extracted_answers.append(extracted)

    # 4. CAS = fraction of incorrect responses matching most common extracted answer
    counts = Counter(extracted_answers)
    most_common_ans, most_common_count = counts.most_common(1)[0]
    
    cas = round(most_common_count / n_incorrect, 4)
    
    if cas >= 0.8:
        failure_mode = "consistent"
    else:
        failure_mode = "mixed"

    # Format output for most_common_wrong_answer strictly as str | None
    mcwa_formatted = None
    if most_common_ans is not None:
        if isinstance(most_common_ans, tuple):
            mcwa_formatted = str(list(most_common_ans))
        else:
            mcwa_formatted = str(most_common_ans)

    return {
        "problem_id": problem_id,
        "family": family,
        "cas": cas,
        "n_incorrect": n_incorrect,
        "most_common_wrong_answer": mcwa_formatted,
        "failure_mode": failure_mode,
    }
