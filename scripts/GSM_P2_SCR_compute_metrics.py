#!/usr/bin/env python3
"""Compute Probe 2 GSM summary metrics."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.behavioral.cci import (
    aggregate_mean_cci_per_model,
    aggregate_mean_tep_per_model,
    aggregate_valid_divergence_rate_per_model,
)
from probes.behavioral.gsm_metrics import cci_by_contamination
from probes.common.stats import bootstrap_ci


INPUT_PATH = Path("results/GSM_P2_RES_cci.csv")
OUTPUT_PATH = Path("results/GSM_P2_RES_metrics_summary.csv")


def _bootstrap_bounds(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    return bootstrap_ci(values, ci=0.95)


def _to_summary_rows(
    *,
    metric: str,
    rows_df: pd.DataFrame,
    value_col: str,
    group_cols: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for _, r in rows_df.iterrows():
        row = {
            "metric": metric,
            "model": str(r.get("model", "")),
            "contamination_pole": "",
            "value": float(r[value_col]),
            "n": int(r.get("n", 0)),
        }
        if "contamination_pole" in group_cols:
            row["contamination_pole"] = str(r.get("contamination_pole", ""))
        rows.append(row)
    return rows


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, dtype=str)
    if df.empty:
        raise ValueError(f"No rows found in {INPUT_PATH}")

    for col in ["cci_score", "tep_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    valid_div_bool = (
        df["valid_divergence"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
    )

    # Exclude valid divergence rows from CCI confabulation aggregates.
    cci_df = df[~valid_div_bool].copy()
    cci_df = cci_df.rename(columns={"cci_score": "cci"})

    mean_cci = aggregate_mean_cci_per_model(cci_df.rename(columns={"cci": "cci_score"}))
    mean_tep = aggregate_mean_tep_per_model(df)
    vd_rate = aggregate_valid_divergence_rate_per_model(df)
    cci_contam = cci_by_contamination(cci_df)

    summary_rows: list[dict] = []
    summary_rows.extend(
        _to_summary_rows(
            metric="mean_cci_excluding_valid_divergence",
            rows_df=mean_cci,
            value_col="mean_cci",
            group_cols=["model"],
        )
    )
    summary_rows.extend(
        _to_summary_rows(
            metric="mean_tep",
            rows_df=mean_tep,
            value_col="mean_tep",
            group_cols=["model"],
        )
    )
    summary_rows.extend(
        _to_summary_rows(
            metric="valid_divergence_rate",
            rows_df=vd_rate,
            value_col="valid_divergence_rate",
            group_cols=["model"],
        )
    )
    summary_rows.extend(
        _to_summary_rows(
            metric="mean_cci_by_contamination_excluding_valid_divergence",
            rows_df=cci_contam,
            value_col="mean_cci",
            group_cols=["model", "contamination_pole"],
        )
    )

    summary = pd.DataFrame(
        summary_rows,
        columns=["metric", "model", "contamination_pole", "value", "n"],
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT_PATH, index=False)

    print("Probe 2 metrics summary (95% bootstrap CI):")

    print("\nMean CCI per model (valid_divergence excluded):")
    for model, group in cci_df.groupby("model", dropna=False):
        vals = pd.to_numeric(group["cci"], errors="coerce").dropna().tolist()
        if not vals:
            continue
        lo, hi = _bootstrap_bounds(vals)
        print(
            f"- {model}: mean={sum(vals)/len(vals):.4f}, n={len(vals)}, "
            f"ci95=[{lo:.4f}, {hi:.4f}]"
        )

    print("\nMean TEP per model:")
    for model, group in df.groupby("model", dropna=False):
        vals = pd.to_numeric(group["tep_score"], errors="coerce").dropna().tolist()
        if not vals:
            continue
        lo, hi = _bootstrap_bounds(vals)
        print(
            f"- {model}: mean={sum(vals)/len(vals):.4f}, n={len(vals)}, "
            f"ci95=[{lo:.4f}, {hi:.4f}]"
        )

    print("\nValid divergence rate per model:")
    for model, group in df.groupby("model", dropna=False):
        vals = (
            group["valid_divergence"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"true": 1.0, "false": 0.0})
            .dropna()
            .tolist()
        )
        if not vals:
            continue
        lo, hi = _bootstrap_bounds(vals)
        print(
            f"- {model}: rate={sum(vals)/len(vals):.4f}, n={len(vals)}, "
            f"ci95=[{lo:.4f}, {hi:.4f}]"
        )

    print("\nCCI by contamination tier (valid_divergence excluded):")
    for (model, pole), group in cci_df.groupby(["model", "contamination_pole"], dropna=False):
        vals = pd.to_numeric(group["cci"], errors="coerce").dropna().tolist()
        if not vals:
            continue
        lo, hi = _bootstrap_bounds(vals)
        print(
            f"- {model} | {pole}: mean={sum(vals)/len(vals):.4f}, n={len(vals)}, "
            f"ci95=[{lo:.4f}, {hi:.4f}]"
        )

    print(f"\nWrote: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
