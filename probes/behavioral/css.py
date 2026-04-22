"""
CSS (Consistency Surface Score) measures the fraction of applicable variants
on which a model's answer matches its answer on the canonical problem.

CSS is undefined for W5 variants (reversal changes the correct answer).
Use rcs.py for W5. W4 and W5 are partial variants reported separately.
"""

from __future__ import annotations

from collections import Counter

import pandas as pd

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
        
        if variant_type == "W5":
            raise ValueError(f"W5 variant encountered for {problem_id}. CSS is undefined for W5.")
            
        is_correct = bool(verify_answer(
            problem_id,
            variant["model_answer"],
            variant["correct_answer"],
            family,
            problem_text=variant.get("problem_text"),
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


def compute_var(df: pd.DataFrame, variant_type: str, model: str) -> float | None:
    rows = df[
        (df["variant_type"].astype(str).str.upper() == variant_type.upper())
        & (df["model"].astype(str) == model)
    ]
    if len(rows) == 0:
        return None
    vals = rows["behavioral_correct"].astype(str).str.strip().str.lower().map(
        {"true": True, "false": False}
    )
    vals = vals.dropna()
    if len(vals) == 0:
        return None
    return round(float(vals.mean()), 4)


def compute_pdas(df: pd.DataFrame, model: str) -> float | None:
    var_teardown = compute_var(df, "W5", model)
    var_forward = compute_var(df, "canonical", model)
    if var_teardown is None or var_forward is None:
        return None
    return round(var_teardown - var_forward, 4)


def compute_pdas_reversal(df: pd.DataFrame, model: str) -> float | None:
    var_proc = compute_var(df, "W6", model)
    var_forward = compute_var(df, "canonical", model)
    if var_proc is None or var_forward is None:
        return None
    return round(var_proc - var_forward, 4)


def compute_dts(df: pd.DataFrame, variant_type: str, model: str) -> float | None:
    bw = df[
        (df["variant_type"].astype(str).str.upper() == variant_type.upper())
        & (df["model"].astype(str) == model)
        & (df["problem_family"].astype(str) == "blocksworld")
    ]
    mbw = df[
        (df["variant_type"].astype(str).str.upper() == variant_type.upper())
        & (df["model"].astype(str) == model)
        & (df["problem_family"].astype(str) == "mystery_blocksworld")
    ]
    var_bw = compute_var(bw, variant_type, model) if len(bw) else None
    var_mbw = compute_var(mbw, variant_type, model) if len(mbw) else None
    if var_bw is None or var_mbw is None:
        return None
    return round(var_bw - var_mbw, 4)


def compute_vri(df: pd.DataFrame, model: str) -> dict[str, float | None]:
    structural_vars = ["W2", "W4"]
    struct_vals = [compute_var(df, v, model) for v in structural_vars]
    struct_vals = [v for v in struct_vals if v is not None]
    vri_structural = round(sum(struct_vals) / len(struct_vals), 4) if struct_vals else None
    vri_vocabulary = compute_var(df, "W3", model)
    gap = None
    if vri_structural is not None and vri_vocabulary is not None:
        gap = round(vri_structural - vri_vocabulary, 4)
    return {
        "structural": vri_structural,
        "vocabulary": vri_vocabulary,
        "gap": gap,
    }


def compute_cfs(df: pd.DataFrame, problem_id: str, model: str) -> float:
    variants = df[
        (df["problem_id"].astype(str) == problem_id)
        & (df["model"].astype(str) == model)
        & (df["variant_type"].astype(str).str.upper().isin(["CANONICAL", "W2", "W3", "W4"]))
    ]
    if len(variants) == 0:
        return 0.0
    first_actions = []
    for raw in variants["raw_response"].tolist():
        if pd.isna(raw):
            first_actions.append("")
            continue
        txt = str(raw).strip()
        first = txt.split("\n")[0].lower() if txt else ""
        first_actions.append(first)
    if not first_actions:
        return 0.0
    mode_count = Counter(first_actions).most_common(1)[0][1]
    return round(mode_count / len(first_actions), 4)
