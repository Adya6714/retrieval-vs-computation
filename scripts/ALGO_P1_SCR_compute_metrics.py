#!/usr/bin/env python3
"""Compute Probe 1 algorithmic metrics from behavioral sweep outputs.

Removed metrics (paper refactor — not named paper outputs):
  CSS, CFS, RCS, GSS (and GSS regressions), VRI_*, Formalism_Gap, HDR, SC, FDI,
  RDI, CPP, ADC (derived metric columns; raw stated_algorithm and
  greedy_assessment_correct remain in phase1 CSVs), VWC (moved to
  scripts/exploratory/compute_vwc_exploratory.py).

Primary CSV outputs: VAR (includes W6 after K1 audit fix).
ALGO accuracies use 110 canonical IDs; clone audit effective n is 51
(results/derived/bank_clone_audit.csv) — treat n=110 as non-independent.
Intermediate (not written to CSV): _std_adv_gap_internal per model/subtype.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.stats import cluster_bootstrap_ci
from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.clones import cluster_ids_for  # noqa: E402


def _to_bool(val: object) -> bool:
    return str(val).strip().lower() == "true"


def _parse_difficulty_params(raw: object) -> dict:
    """Parse bank difficulty_params; tolerate NaN, floats, and empty values."""
    if raw is None:
        return {}
    if isinstance(raw, float):
        if np.isnan(raw):
            return {}
        return {}
    if isinstance(raw, dict):
        return raw
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none"}:
        return {}
    try:
        params = json.loads(s)
    except json.JSONDecodeError:
        return {}
    return params if isinstance(params, dict) else {}


def _norm_algo_variant(v: object) -> str:
    s = str(v).strip()
    if not s:
        return s
    sl = s.lower()
    if sl == "canonical":
        return "canonical"
    if len(sl) == 2 and sl[0] == "w" and sl[1].isdigit():
        return f"W{sl[1]}"
    return s


def _bootstrap_diff(
    a: np.ndarray, b: np.ndarray, n_resamples: int
) -> tuple[float, float]:
    if len(a) == 0 or len(b) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_resamples):
        ra = a[rng.integers(0, len(a), len(a))]
        rb = b[rng.integers(0, len(b), len(b))]
        diffs.append(float(np.mean(ra) - np.mean(rb)))
    return (
        float(np.percentile(diffs, 2.5)),
        float(np.percentile(diffs, 97.5)),
    )


def _add_metric(
    rows: list[dict],
    *,
    model: str,
    subtype: str,
    variant_type: str,
    metric_name: str,
    metric_value: float | str,
    ci_lower: float | str = "",
    ci_upper: float | str = "",
) -> None:
    rows.append(
        {
            "model": model,
            "subtype": subtype,
            "variant_type": variant_type,
            "metric_name": metric_name,
            "metric_value": metric_value,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sweep-results",
        nargs="+",
        required=False,
        default=[
            "results/raw/ALGO_P1_behavioral_claude.csv",
            "results/raw/ALGO_P1_behavioral_gpt4o.csv",
            "results/raw/ALGO_P1_behavioral_llama.csv",
        ],
    )
    parser.add_argument("--bank", required=False, default="data/problems/question_bank_algo.csv")
    parser.add_argument("--output", required=False, default="results/deprecated/ALGO_P1_metrics.csv")
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    args = parser.parse_args()

    sweep_frames = [pd.read_csv(Path(p), dtype=str).fillna("") for p in args.sweep_results]
    sweep = pd.concat(sweep_frames, ignore_index=True)
    sweep = sweep[sweep["model"].astype(str).str.lower() != "mock"].copy()
    sweep["variant_type"] = sweep["variant_type"].map(_norm_algo_variant)
    bank = pd.read_csv(Path(args.bank), dtype=str).fillna("")
    bank["variant_type"] = bank["variant_type"].map(_norm_algo_variant)

    required_sweep = {
        "problem_id",
        "variant_type",
        "model",
        "model_answer",
        "ground_truth",
        "verified",
        "difficulty_params_instance_type",
    }
    miss = required_sweep - set(sweep.columns)
    if miss:
        raise ValueError(f"Sweep results missing required columns: {sorted(miss)}")

    required_bank = {"problem_id", "variant_type", "problem_subtype", "difficulty_params"}
    miss_bank = required_bank - set(bank.columns)
    if miss_bank:
        raise ValueError(f"Bank missing required columns: {sorted(miss_bank)}")

    merged = sweep.merge(
        bank[
            [
                "problem_id",
                "variant_type",
                "problem_subtype",
                "difficulty_params",
                "correct_answer",
                "notes",
                "problem_text",
            ]
        ].rename(columns={"correct_answer": "bank_ground_truth"}),
        on=["problem_id", "variant_type"],
        how="left",
        validate="many_to_one",
    )
    if merged["problem_subtype"].eq("").any():
        raise ValueError("Join failed for some rows: missing problem_subtype after merge.")

    parsed_params: list[dict] = []
    for _, r in merged.iterrows():
        params = _parse_difficulty_params(r["difficulty_params"])
        inst = str(r.get("difficulty_params_instance_type") or "").strip().lower()
        if inst and "instance_type" not in params:
            params = {**params, "instance_type": inst}
        parsed_params.append(params)
    merged["difficulty_params_obj"] = parsed_params

    reverified: list[bool] = []
    for _, r in merged.iterrows():
        gt = str(r.get("bank_ground_truth") or "").strip() or str(r.get("ground_truth") or "")
        try:
            ok, _reason, _meta = verify_algo(
                str(r["problem_id"]),
                str(r["model_answer"]),
                gt,
                str(r["problem_subtype"]),
                str(r["variant_type"]),
                r["difficulty_params_obj"],
                notes=str(r.get("notes") or ""),
                problem_text=str(r.get("problem_text") or ""),
            )
            reverified.append(bool(ok))
        except Exception:
            reverified.append(_to_bool(r.get("verified", False)))
    merged["verified_bool"] = reverified
    merged["subtype"] = merged["problem_subtype"]
    merged["instance_type"] = merged["difficulty_params_instance_type"]
    merged = filter_excluded(merged, family="ALGO")

    metric_rows: list[dict] = []

    for (model, subtype, variant), g in merged.groupby(["model", "subtype", "variant_type"]):
        vals = g["verified_bool"].astype(float).tolist()
        mean = float(np.mean(vals))
        lo, hi = cluster_bootstrap_ci(
            vals,
            cluster_ids_for(g["problem_id"].astype(str).tolist()),
            n_resamples=args.bootstrap_n,
            seed=42,
        )
        _add_metric(
            metric_rows,
            model=model,
            subtype=subtype,
            variant_type=variant,
            metric_name="VAR",
            metric_value=mean,
            ci_lower=lo,
            ci_upper=hi,
        )

    for (model, subtype), g in merged.groupby(["model", "subtype"]):
        std_canon = g[(g["variant_type"] == "canonical") & (g["instance_type"] == "standard")][
            "verified_bool"
        ].astype(float)
        adv_canon = g[(g["variant_type"] == "canonical") & (g["instance_type"] == "adversarial")][
            "verified_bool"
        ].astype(float)
        if not std_canon.empty and not adv_canon.empty:
            # kept for prose reporting, not a named paper metric
            _std_adv_gap_internal = float(std_canon.mean() - adv_canon.mean())  # noqa: F841

    out_df = pd.DataFrame(
        metric_rows, columns=["model", "subtype", "variant_type", "metric_name", "metric_value", "ci_lower", "ci_upper"]
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote metrics: {out_path} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()
