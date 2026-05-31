#!/usr/bin/env python3
"""Compute GSM Probe 1 metrics from behavioral sweep outputs.

Removed metrics (paper refactor — not named paper outputs):
  CSS, CFS, RCS, GSS, CPP, SC, FDI, RDI, ADC (derived columns), VWC
  (moved to scripts/exploratory/compute_vwc_exploratory.py), and legacy
  VRI_structural / VRI_vocabulary / VRI_gap / RAR_W5 named outputs.

Primary CSV outputs: VAR, W6_Gap (GSM family only for W6_Gap).
Intermediate (not written to CSV): _vri_direction_internal per contamination pole
(mean across models of per-model avg(VAR(W1), VAR(W2), VAR(W4)) − VAR(W3)).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.stats import bootstrap_ci


def _norm_gsm_variant(v: object) -> str:
    s = str(v).strip()
    if not s:
        return s
    low = s.lower()
    if low == "canonical":
        return "canonical"
    if re.fullmatch(r"w[1-6]", low):
        return f"W{low[1]}"
    return s


def _to_bool_series(series: pd.Series) -> pd.Series:
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )
    return mapped.dropna().astype(bool)


def _bootstrap_diff(a: np.ndarray, b: np.ndarray, n_resamples: int) -> tuple[float, float]:
    if len(a) == 0 or len(b) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    diffs: list[float] = []
    for _ in range(n_resamples):
        ra = a[rng.integers(0, len(a), len(a))]
        rb = b[rng.integers(0, len(b), len(b))]
        diffs.append(float(np.mean(ra) - np.mean(rb)))
    return (
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def _model_variant_var(
    sweep: pd.DataFrame, model: str, variant: str, contamination_pole: str
) -> float | None:
    g = sweep[
        (sweep["model"] == model)
        & (sweep["variant_type"] == variant)
        & (sweep["contamination_pole"] == contamination_pole)
    ]
    if g.empty:
        return None
    return float(g["correct_bool"].mean())


def _compute_vri_direction_internal(sweep: pd.DataFrame) -> dict[str, float]:
    """avg(VAR(W1), VAR(W2), VAR(W4)) − VAR(W3) per model, then mean across models per pole."""
    poles = sorted(sweep["contamination_pole"].astype(str).str.strip().unique().tolist())
    poles = [p for p in poles if p]
    out: dict[str, float] = {}
    for pole in poles:
        print(f"\n[VRI] contamination_pole={pole!r}")
        per_model_vri: list[float] = []
        for model in sorted(sweep["model"].unique()):
            w1 = _model_variant_var(sweep, str(model), "W1", pole)
            w2 = _model_variant_var(sweep, str(model), "W2", pole)
            w3 = _model_variant_var(sweep, str(model), "W3", pole)
            w4 = _model_variant_var(sweep, str(model), "W4", pole)
            structural = [x for x in (w1, w2, w4) if x is not None]
            if not structural or w3 is None:
                print(f"  {model}: skip (missing variant VAR)")
                continue
            structural_mean = float(sum(structural) / len(structural))
            vri = structural_mean - w3
            per_model_vri.append(vri)
            print(
                f"  {model}: VAR(W1)={w1:.4f} VAR(W2)={w2:.4f} VAR(W3)={w3:.4f} "
                f"VAR(W4)={w4:.4f} avg(W1,W2,W4)={structural_mean:.4f} VRI={vri:.4f}"
            )
        if not per_model_vri:
            raise ValueError(f"No per-model VRI values for contamination_pole={pole!r}")
        pole_mean = float(np.mean(per_model_vri))
        out[pole] = pole_mean
        print(f"[VRI] VRI_{pole} (mean across {len(per_model_vri)} models) = {pole_mean:.4f}")
    return out


def _add_metric(
    rows: list[dict],
    *,
    model: str,
    metric_name: str,
    metric_value: float | str,
    n: int | str = "",
    variant_type: str = "",
    contamination_pole: str = "",
    ci_lower: float | str = "",
    ci_upper: float | str = "",
) -> None:
    rows.append(
        {
            "model": model,
            "metric_name": metric_name,
            "variant_type": variant_type,
            "contamination_pole": contamination_pole,
            "n": n,
            "metric_value": metric_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute GSM Probe 1 metrics")
    parser.add_argument(
        "--sweep-results",
        nargs="+",
        required=False,
        default=[
            "results/raw/GSM_P1_behavioral_claude.csv",
            "results/raw/GSM_P1_behavioral_gpt4o.csv",
            "results/raw/GSM_P1_behavioral_llama.csv",
        ],
    )
    parser.add_argument("--bank", required=False, default="data/problems/question_bank_gsm.csv")
    parser.add_argument("--output", required=False, default="results/derived/GSM_P1_metrics.csv")
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    args = parser.parse_args()

    sweep_frames = []
    for p in args.sweep_results:
        path = Path(p)
        if not path.exists() or path.stat().st_size <= 200:
            continue
        sweep_frames.append(pd.read_csv(path, dtype=str).fillna(""))
    if not sweep_frames:
        raise FileNotFoundError(
            "No non-empty GSM behavioral sweep files found. "
            f"Checked: {args.sweep_results}"
        )
    sweep = pd.concat(sweep_frames, ignore_index=True)
    sweep = sweep[sweep["model"].astype(str).str.lower() != "mock"].copy()
    sweep["variant_type"] = sweep["variant_type"].map(_norm_gsm_variant)

    bank = pd.read_csv(Path(args.bank), dtype=str).fillna("")
    bank["variant_type"] = bank["variant_type"].map(_norm_gsm_variant)

    required_sweep = {
        "problem_id",
        "variant_type",
        "model",
        "behavioral_correct",
        "contamination_pole",
    }
    miss = required_sweep - set(sweep.columns)
    if miss:
        raise ValueError(f"Sweep results missing required columns: {sorted(miss)}")

    required_bank = {"problem_id", "variant_type", "contamination_pole"}
    miss_bank = required_bank - set(bank.columns)
    if miss_bank:
        raise ValueError(f"Bank missing required columns: {sorted(miss_bank)}")

    canon_pole = (
        bank[bank["variant_type"].astype(str).str.lower() == "canonical"][
            ["problem_id", "contamination_pole"]
        ]
        .drop_duplicates("problem_id")
        .set_index("problem_id")["contamination_pole"]
        .to_dict()
    )
    empty_pole = sweep["contamination_pole"].astype(str).str.strip() == ""
    sweep.loc[empty_pole, "contamination_pole"] = sweep.loc[empty_pole, "problem_id"].map(canon_pole).fillna("")

    mapped = (
        sweep["behavioral_correct"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
    )
    if mapped.isna().any():
        bad = int(mapped.isna().sum())
        raise ValueError(f"Found {bad} rows with non-boolean behavioral_correct values.")
    sweep["correct_bool"] = mapped.astype(bool)

    metric_rows: list[dict] = []

    for (model, variant, pole), g in sweep.groupby(["model", "variant_type", "contamination_pole"], dropna=False):
        vals = g["correct_bool"].astype(float).to_numpy()
        if len(vals) == 0:
            continue
        lo, hi = bootstrap_ci(vals.tolist(), n_resamples=args.bootstrap_n, ci=0.95)
        _add_metric(
            metric_rows,
            model=str(model),
            metric_name="VAR",
            variant_type=str(variant),
            contamination_pole=str(pole),
            n=int(len(vals)),
            metric_value=float(np.mean(vals)),
            ci_lower=lo,
            ci_upper=hi,
        )

    # kept for prose reporting, not a named paper metric
    _vri_direction_internal = _compute_vri_direction_internal(sweep)
    assert _vri_direction_internal["medium"] > 0, (
        "VRI_medium should be positive when W1/W2/W4 exceed W3 per model; "
        f"got {_vri_direction_internal['medium']:.4f}"
    )

    for model, g_model in sweep.groupby("model", dropna=False):
        c = g_model[g_model["variant_type"] == "canonical"]["correct_bool"].astype(float).to_numpy()
        w6 = g_model[g_model["variant_type"] == "W6"]["correct_bool"].astype(float).to_numpy()
        if len(c) == 0 or len(w6) == 0:
            continue
        lo_w6, hi_w6 = _bootstrap_diff(c, w6, args.bootstrap_n)
        _add_metric(
            metric_rows,
            model=str(model),
            metric_name="W6_Gap",
            variant_type="W6",
            n=min(len(c), len(w6)),
            metric_value=float(np.mean(c) - np.mean(w6)),
            ci_lower=lo_w6,
            ci_upper=hi_w6,
        )

    out_df = pd.DataFrame(
        metric_rows,
        columns=[
            "model",
            "metric_name",
            "variant_type",
            "contamination_pole",
            "n",
            "metric_value",
            "ci_lower",
            "ci_upper",
        ],
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote metrics: {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
