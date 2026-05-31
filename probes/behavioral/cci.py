"""
CCI measures whether the model's execution follows its
own plan. Low CCI indicates plan confabulation. Pilot result: CCI=0.26 on 7
Blocksworld instances (GPT-4o). Only applies to Probe 2 Blocksworld instances.
"""

from __future__ import annotations

import pandas as pd


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


def aggregate_mean_cci_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean CCI per model from a frame containing a cci_score column."""
    required = {"model", "cci_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")
    rows: list[dict] = []
    for model, group in df.groupby("model", dropna=False):
        vals = pd.to_numeric(group["cci_score"], errors="coerce").dropna()
        n = int(len(vals))
        if n == 0:
            continue
        rows.append({"model": str(model), "mean_cci": float(vals.mean()), "n": n})
    out = pd.DataFrame(rows)
    return out.sort_values(["model"]).reset_index(drop=True) if not out.empty else out


def aggregate_mean_tep_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean TEP per model from a frame containing a tep_score column."""
    required = {"model", "tep_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")
    rows: list[dict] = []
    for model, group in df.groupby("model", dropna=False):
        vals = pd.to_numeric(group["tep_score"], errors="coerce").dropna()
        n = int(len(vals))
        if n == 0:
            continue
        rows.append({"model": str(model), "mean_tep": float(vals.mean()), "n": n})
    out = pd.DataFrame(rows)
    return out.sort_values(["model"]).reset_index(drop=True) if not out.empty else out


def aggregate_valid_divergence_rate_per_model(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid_divergence rate per model."""
    required = {"model", "valid_divergence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")
    rows: list[dict] = []
    for model, group in df.groupby("model", dropna=False):
        vals = (
            group["valid_divergence"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1.0, "false": 0.0})
            .dropna()
        )
        n = int(len(vals))
        if n == 0:
            continue
        rows.append(
            {
                "model": str(model),
                "valid_divergence_rate": float(vals.mean()),
                "n": n,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["model"]).reset_index(drop=True) if not out.empty else out
