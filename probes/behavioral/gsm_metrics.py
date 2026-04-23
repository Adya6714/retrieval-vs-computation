"""GSM-specific behavioral metrics."""

from __future__ import annotations

import pandas as pd


def _to_bool_series(series: pd.Series) -> pd.Series:
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )
    return mapped.dropna().astype(bool)


def _score_mean(df: pd.DataFrame) -> float | None:
    if df.empty:
        return None
    vals = _to_bool_series(df["behavioral_correct"])
    if vals.empty:
        return None
    return float(vals.mean())


def w4_gap(df: pd.DataFrame) -> pd.DataFrame:
    """Per problem/model prose-vs-formal gap using W1/W2/W3 vs W4."""
    required = {
        "problem_id",
        "model",
        "variant_type",
        "behavioral_correct",
        "contamination_pole",
        "difficulty",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for (problem_id, model), group in df.groupby(["problem_id", "model"], dropna=False):
        vtype = group["variant_type"].astype(str).str.strip().str.lower()
        prose_df = group[vtype.isin(["w1", "w2", "w3"])]
        w4_df = group[vtype == "w4"]

        prose_var = _score_mean(prose_df)
        w4_var = _score_mean(w4_df)
        if prose_var is None or w4_var is None:
            continue

        rows.append(
            {
                "problem_id": str(problem_id),
                "model": str(model),
                "prose_var": prose_var,
                "w4_var": w4_var,
                "gap": prose_var - w4_var,
                "contamination_pole": str(group["contamination_pole"].iloc[0]),
                "difficulty": str(group["difficulty"].iloc[0]),
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(["model", "problem_id"]).reset_index(drop=True) if not out.empty else out


def vri(df: pd.DataFrame) -> pd.DataFrame:
    """Compute model-level VRI structural/vocabulary split."""
    required = {"model", "variant_type", "behavioral_correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    for model, group in df.groupby("model", dropna=False):
        vtype = group["variant_type"].astype(str).str.strip().str.lower()
        w2 = _score_mean(group[vtype == "w2"])
        w4 = _score_mean(group[vtype == "w4"])
        w3 = _score_mean(group[vtype == "w3"])

        structural_vals = [x for x in [w2, w4] if x is not None]
        vri_structural = (
            float(sum(structural_vals) / len(structural_vals)) if structural_vals else None
        )
        vri_vocabulary = w3
        vri_gap = (
            vri_structural - vri_vocabulary
            if vri_structural is not None and vri_vocabulary is not None
            else None
        )
        rows.append(
            {
                "model": str(model),
                "vri_structural": vri_structural,
                "vri_vocabulary": vri_vocabulary,
                "vri_gap": vri_gap,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["model"]).reset_index(drop=True) if not out.empty else out


def rcs_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """RCS from W6 rows, grouped by model and difficulty."""
    required = {"model", "variant_type", "difficulty", "behavioral_correct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing required columns: {sorted(missing)}")

    w6 = df[df["variant_type"].astype(str).str.strip().str.lower() == "w6"].copy()
    rows: list[dict] = []
    for (model, difficulty), group in w6.groupby(["model", "difficulty"], dropna=False):
        vals = _to_bool_series(group["behavioral_correct"])
        n = int(len(vals))
        if n == 0:
            continue
        rows.append(
            {
                "model": str(model),
                "difficulty": str(difficulty),
                "rcs": float(vals.mean()),
                "n": n,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["model", "difficulty"]).reset_index(drop=True) if not out.empty else out


def step_count_sensitivity(df: pd.DataFrame, question_bank: pd.DataFrame) -> pd.DataFrame:
    """Mean per-problem CSS grouped by subtype (gsm_symbolic vs gsm_p1p2)."""
    required_df = {"problem_id", "model", "variant_type", "behavioral_correct"}
    missing_df = required_df - set(df.columns)
    if missing_df:
        raise ValueError(f"df missing required columns: {sorted(missing_df)}")

    required_qb = {"problem_id", "problem_subtype"}
    missing_qb = required_qb - set(question_bank.columns)
    if missing_qb:
        raise ValueError(f"question_bank missing required columns: {sorted(missing_qb)}")

    wb = question_bank[question_bank["variant_type"].astype(str).str.lower() == "canonical"].copy()
    wb = wb[["problem_id", "problem_subtype"]].drop_duplicates("problem_id")

    scoped = df[df["variant_type"].astype(str).str.lower().isin(["canonical", "w2", "w3", "w4"])]
    merged = scoped.merge(wb, on="problem_id", how="left")
    merged = merged[merged["problem_subtype"].astype(str).isin(["gsm_symbolic", "gsm_p1p2"])]

    per_problem_rows: list[dict] = []
    for (model, problem_id), group in merged.groupby(["model", "problem_id"], dropna=False):
        vals = _to_bool_series(group["behavioral_correct"])
        if vals.empty:
            continue
        per_problem_rows.append(
            {
                "model": str(model),
                "problem_id": str(problem_id),
                "problem_subtype": str(group["problem_subtype"].iloc[0]),
                "css": float(vals.mean()),
            }
        )
    per_problem = pd.DataFrame(per_problem_rows)
    if per_problem.empty:
        return pd.DataFrame(columns=["model", "problem_subtype", "mean_css", "n"])

    out_rows: list[dict] = []
    for (model, subtype), group in per_problem.groupby(["model", "problem_subtype"], dropna=False):
        out_rows.append(
            {
                "model": str(model),
                "problem_subtype": str(subtype),
                "mean_css": float(group["css"].mean()),
                "n": int(len(group)),
            }
        )
    out = pd.DataFrame(out_rows)
    return out.sort_values(["model", "problem_subtype"]).reset_index(drop=True)


def cci_by_contamination(probe2_results: pd.DataFrame) -> pd.DataFrame:
    """Summarize CCI by contamination pole and model."""
    required = {"model", "contamination_pole", "cci"}
    missing = required - set(probe2_results.columns)
    if missing:
        raise ValueError(
            f"probe2_results missing required columns: {sorted(missing)}"
        )

    rows: list[dict] = []
    for (model, pole), group in probe2_results.groupby(
        ["model", "contamination_pole"], dropna=False
    ):
        vals = pd.to_numeric(group["cci"], errors="coerce").dropna()
        n = int(len(vals))
        if n == 0:
            continue
        rows.append(
            {
                "model": str(model),
                "contamination_pole": str(pole),
                "mean_cci": float(vals.mean()),
                "median_cci": float(vals.median()),
                "n": n,
            }
        )
    out = pd.DataFrame(rows)
    return (
        out.sort_values(["model", "contamination_pole"]).reset_index(drop=True)
        if not out.empty
        else out
    )
