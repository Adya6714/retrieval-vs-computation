"""Variant Accuracy Rate (VAR) aggregation helpers."""

from __future__ import annotations

import pandas as pd

from probes.common.stats import bootstrap_ci


def _to_bool_series(series: pd.Series) -> pd.Series:
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )
    return mapped.dropna().astype(bool)


def compute_var(behavioral_results: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model, per-variant VAR with bootstrap 95% CI."""
    if behavioral_results.empty:
        return pd.DataFrame(
            columns=["model", "variant_type", "var_score", "n", "ci_lower", "ci_upper"]
        )

    required = {"model", "variant_type", "behavioral_correct"}
    missing = required - set(behavioral_results.columns)
    if missing:
        raise ValueError(f"behavioral_results missing required columns: {sorted(missing)}")

    rows: list[dict] = []
    grouped = behavioral_results.groupby(["model", "variant_type"], dropna=False)
    for (model, variant_type), group in grouped:
        vals = _to_bool_series(group["behavioral_correct"])
        n = int(len(vals))
        if n == 0:
            continue
        floats = vals.astype(float).tolist()
        var_score = float(sum(floats) / n)
        ci_lower, ci_upper = bootstrap_ci(floats, ci=0.95)
        rows.append(
            {
                "model": str(model),
                "variant_type": str(variant_type),
                "var_score": round(var_score, 6),
                "n": n,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=["model", "variant_type", "var_score", "n", "ci_lower", "ci_upper"]
        )
    return out.sort_values(["model", "variant_type"]).reset_index(drop=True)
