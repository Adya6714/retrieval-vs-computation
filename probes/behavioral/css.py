"""
CSS (Consistency Surface Score) measures the fraction of applicable variants
on which a model's answer matches its answer on the canonical problem.

CSS is undefined for W6 variants (reversal changes the correct answer).
Use rcs.py for W6. W4 and W6 are partial variants reported separately.
"""

from __future__ import annotations

from probes.contamination.verify import verify_answer


def compute_css(
    problem_id: str,
    canonical_answer: str,
    variant_responses: list[dict],
    family: str,
) -> dict:
    if not variant_responses:
        # Note: Empty variant_responses list, so we cannot compute CSS.
        return {
            "problem_id": problem_id,
            "family": family,
            "css": None,
            "variants_evaluated": 0,
            "variants_correct": 0,
            "per_variant": {},
        }

    variants_evaluated = 0
    variants_correct = 0
    per_variant = {}

    for variant in variant_responses:
        variant_type = variant["variant_type"]
        
        if variant_type == "W6":
            raise ValueError(f"W6 variant encountered for {problem_id}. CSS is undefined for W6.")
            
        is_correct = bool(verify_answer(
            problem_id, 
            variant["model_answer"], 
            variant["correct_answer"], 
            family
        ))
        
        per_variant[variant_type] = is_correct
        variants_evaluated += 1
        if is_correct:
            variants_correct += 1

    css = round(variants_correct / variants_evaluated, 4) if variants_evaluated > 0 else None

    return {
        "problem_id": problem_id,
        "family": family,
        "css": css,
        "variants_evaluated": variants_evaluated,
        "variants_correct": variants_correct,
        "per_variant": per_variant,
    }
