#!/usr/bin/env python3
"""Compute BW Probe 1 summary metrics.

Removed metrics (paper refactor — not named paper outputs):
  CSS, CFS, RCS, GSS, CPP, SC, FDI, RDI, ADC (derived columns), VWC
  (moved to scripts/exploratory/compute_vwc_exploratory.py), and W6_Gap
  (GSM/ALGO only).

Primary CSV outputs: VAR, PDAS (VAR(W5) − VAR(canonical) on standard BW instances
matching ^BW_\\d+$ (standard blocksworld only; excludes BW_E and MBW).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.stats import bootstrap_ci


def _bw_standard_blocksworld_mask(problem_id: pd.Series) -> pd.Series:
    """Standard BW instances only (exclude BW_E* extended and MBW* mystery)."""
    return problem_id.astype(str).str.match(r"^BW_\d+$", case=False)


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0})


def _instance_set(problem_id: str) -> str:
    pid = str(problem_id)
    if pid.startswith("MBW"):
        return "mbw"
    if pid.startswith("BW_E"):
        return "extended_bw"
    return "standard_bw"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BW Probe 1 metrics")
    parser.add_argument(
        "--sweep",
        default="results/raw/BW_P1_behavioral.csv",
        help="Behavioral sweep CSV",
    )
    parser.add_argument(
        "--output",
        default="results/deprecated/BW_P1_metrics.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()
    np.random.seed(42)

    sweep_path = Path(args.sweep)
    if not sweep_path.exists():
        raise FileNotFoundError(f"Missing BW sweep: {sweep_path}")
    df = pd.read_csv(sweep_path, dtype=str).fillna("")
    df = df[df["model"].astype(str).str.lower() != "mock"].copy()
    if "behavioral_correct" not in df.columns:
        raise ValueError("BW sweep missing behavioral_correct column.")

    df["correct"] = _to_bool(df["behavioral_correct"])
    df["variant_type"] = df["variant_type"].astype(str).str.strip().str.lower()
    df["instance_set"] = df["problem_id"].map(_instance_set)

    rows: list[dict] = []
    for (model, variant, instance_set), g in df.groupby(
        ["model", "variant_type", "instance_set"], dropna=False
    ):
        vals = g["correct"].dropna().astype(float).to_numpy()
        if len(vals) == 0:
            continue
        lo, hi = bootstrap_ci(vals.tolist(), n_resamples=10000)
        rows.append(
            {
                "family": "BW",
                "model": model,
                "variant_type": variant,
                "metric_name": "VAR",
                "metric_value": float(np.mean(vals)),
                "n": int(len(vals)),
                "denominator_n": int(len(vals)),
                "instance_set": instance_set,
                "ci_lower": float(lo),
                "ci_upper": float(hi),
            }
        )

    std_mask = _bw_standard_blocksworld_mask(df["problem_id"])
    rng = np.random.default_rng(42)
    for model, g_all in df.groupby("model", dropna=False):
        g = g_all[std_mask.loc[g_all.index]].copy()
        canon = g[g["variant_type"] == "canonical"][["problem_id", "correct"]]
        w5 = g[g["variant_type"] == "w5"][["problem_id", "correct"]]
        paired = canon.merge(w5, on="problem_id", suffixes=("_can", "_w5"))
        if paired.empty:
            continue
        d = (paired["correct_w5"] - paired["correct_can"]).astype(float).to_numpy()
        pt = float(np.mean(d))
        boots: list[float] = []
        for _ in range(10000):
            idx = rng.integers(0, len(d), size=len(d))
            boots.append(float(np.mean(d[idx])))
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        rows.append(
            {
                "family": "BW",
                "model": model,
                "variant_type": "w5_vs_canonical",
                "metric_name": "PDAS",
                "metric_value": pt,
                "n": int(len(d)),
                "denominator_n": int(len(d)),
                "instance_set": "standard_bw",
                "ci_lower": lo,
                "ci_upper": hi,
            }
        )

    out = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote metrics: {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
