#!/usr/bin/env python3
"""ALGO-only triangulation pipeline across Probe 1/2/3 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _to_bool(x: Any) -> bool | None:
    s = str(x).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None


def load_data(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    behavioral_parts = []
    for p in args.behavioral_results:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Behavioral results file not found: {path}")
        behavioral_parts.append(pd.read_csv(path, dtype=str).fillna(""))
    behavioral = pd.concat(behavioral_parts, ignore_index=True)

    probe2_path = Path(args.probe2_metrics)
    contamination_path = Path(args.contamination)
    bank_path = Path(args.bank)
    for p in [probe2_path, contamination_path, bank_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input file not found: {p}")
    bw_metrics = pd.DataFrame()
    if args.bw_metrics:
        bw_path = Path(args.bw_metrics)
        if not bw_path.exists():
            raise FileNotFoundError(f"BW metrics baseline file not found: {bw_path}")
        bw_metrics = pd.read_csv(bw_path, dtype=str).fillna("")

    return {
        "behavioral": behavioral,
        "probe2": pd.read_csv(probe2_path, dtype=str).fillna(""),
        "contamination": pd.read_csv(contamination_path, dtype=str).fillna(""),
        "bank": pd.read_csv(bank_path, dtype=str).fillna(""),
        "bw_metrics": bw_metrics,
    }


def _parse_bank(bank: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        bank,
        {"problem_id", "variant_type", "problem_subtype", "difficulty_params"},
        "bank",
    )
    b = bank[bank["variant_type"].str.strip().str.lower() == "canonical"].copy()
    if b["problem_id"].duplicated().any():
        raise ValueError("Canonical bank has duplicate problem_id rows.")

    def parse_params(pid: str, raw: str) -> dict[str, Any]:
        if not str(raw).strip():
            raise ValueError(f"{pid}: missing difficulty_params")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"{pid}: invalid difficulty_params JSON: {e}") from e

    b["params"] = [parse_params(pid, raw) for pid, raw in zip(b["problem_id"], b["difficulty_params"])]
    b["instance_type"] = b["params"].map(lambda p: str(p.get("instance_type", "")).strip().lower())
    b["greedy_succeeds_expected"] = b["params"].map(lambda p: p.get("greedy_succeeds", None))
    b["critical_step_index"] = b["params"].map(
        lambda p: int(p.get("critical_step_index", -1))
        if str(p.get("critical_step_index", "")).strip() != ""
        else -1
    )
    return b[["problem_id", "problem_subtype", "instance_type", "greedy_succeeds_expected", "critical_step_index"]]


def _build_per_instance_behavioral(behavioral: pd.DataFrame) -> pd.DataFrame:
    _require_columns(
        behavioral,
        {
            "problem_id",
            "variant_type",
            "model",
            "verified",
            "correct_canonical",
            "gave_greedy_answer",
        },
        "behavioral",
    )
    b = behavioral.copy()
    b["variant_type"] = b["variant_type"].str.strip()
    b["verified_bool"] = b["verified"].map(_to_bool)
    b["correct_canonical_bool"] = b["correct_canonical"].map(_to_bool)
    b["gave_greedy_answer_bool"] = b["gave_greedy_answer"].map(_to_bool)
    b["parse_failed"] = b["parse_status"].astype(str).str.strip().str.lower().eq("parse_failed") if "parse_status" in b.columns else False

    keys = b[["problem_id", "model"]].drop_duplicates()
    can = b[b["variant_type"] == "canonical"][["problem_id", "model", "verified_bool", "parse_failed"]].rename(
        columns={"verified_bool": "correct_canonical_variant", "parse_failed": "parse_failed_canonical"}
    )
    w3 = b[b["variant_type"] == "W3"][["problem_id", "model", "verified_bool", "parse_failed"]].rename(
        columns={"verified_bool": "correct_W3", "parse_failed": "parse_failed_W3"}
    )

    per = keys.merge(can, on=["problem_id", "model"], how="left", validate="one_to_one")
    per = per.merge(w3, on=["problem_id", "model"], how="left", validate="one_to_one")
    agg = b.groupby(["problem_id", "model"], as_index=False).agg(
        correct_canonical=("correct_canonical_bool", "max"),
        gave_greedy_answer=("gave_greedy_answer_bool", "max"),
        any_parse_failed=("parse_failed", "max"),
    )
    per = per.merge(agg, on=["problem_id", "model"], how="left", validate="one_to_one")
    per["VAR_canonical"] = per["correct_canonical_variant"].map(lambda x: 1.0 if x is True else 0.0 if x is False else np.nan)
    per["VAR_W3"] = per["correct_W3"].map(lambda x: 1.0 if x is True else 0.0 if x is False else np.nan)
    per["VRI_gap"] = per["VAR_canonical"] - per["VAR_W3"]
    return per


def _pivot_probe2(probe2: pd.DataFrame) -> pd.DataFrame:
    _require_columns(probe2, {"model", "subtype", "metric_name", "metric_value"}, "probe2_metrics")
    p = probe2.copy()
    p["metric_value_num"] = pd.to_numeric(p["metric_value"], errors="coerce")
    pivot = p.pivot_table(index=["model", "subtype"], columns="metric_name", values="metric_value_num", aggfunc="first").reset_index()
    pivot.columns = [str(c) for c in pivot.columns]
    pivot["subtype"] = pivot["subtype"].str.strip().str.lower()
    return pivot


def merge_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    bank = _parse_bank(data["bank"])
    per = _build_per_instance_behavioral(data["behavioral"])
    _require_columns(
        data["contamination"],
        {"problem_id", "problem_subtype", "instance_contamination_score", "template_contamination_score", "difficulty_numeric"},
        "contamination",
    )
    cont = data["contamination"][["problem_id", "problem_subtype", "instance_contamination_score", "template_contamination_score", "difficulty_numeric"]].copy()
    cont["problem_subtype"] = cont["problem_subtype"].str.strip().str.lower()
    cont["instance_contamination_score"] = pd.to_numeric(cont["instance_contamination_score"], errors="coerce")
    cont["template_contamination_score"] = pd.to_numeric(cont["template_contamination_score"], errors="coerce")
    cont["difficulty_numeric"] = pd.to_numeric(cont["difficulty_numeric"], errors="coerce")
    # Resume/appended contamination runs can duplicate problem_id rows.
    # Keep one row per key, preferring the last non-null values.
    if cont.duplicated(subset=["problem_id", "problem_subtype"]).any():
        cont = (
            cont.sort_index()
            .groupby(["problem_id", "problem_subtype"], as_index=False)
            .agg(
                {
                    "instance_contamination_score": "last",
                    "template_contamination_score": "last",
                    "difficulty_numeric": "last",
                }
            )
        )

    merged = per.merge(bank, on="problem_id", how="left", validate="many_to_one")
    merged["problem_subtype"] = merged["problem_subtype"].str.strip().str.lower()
    merged = merged.merge(cont, on=["problem_id", "problem_subtype"], how="left", validate="many_to_one")

    p2 = _pivot_probe2(data["probe2"])
    merged = merged.merge(
        p2,
        left_on=["model", "problem_subtype"],
        right_on=["model", "subtype"],
        how="left",
        validate="many_to_one",
    ).drop(columns=["subtype"])

    if merged.duplicated(subset=["problem_id", "model"]).any():
        raise ValueError("Merge produced duplicate (problem_id, model) rows.")
    print(f"merge coverage: {len(merged)} rows ({merged['problem_id'].nunique()} problems x {merged['model'].nunique()} models)")
    return merged


def compute_convergence_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["greedy_assessment_correct", "critical_point_identified"]:
        if col not in out.columns:
            out[col] = ""
    out["greedy_assessment_correct_bool"] = out["greedy_assessment_correct"].map(_to_bool)
    out["critical_point_identified_num"] = pd.to_numeric(out["critical_point_identified"], errors="coerce")
    out["gave_greedy_answer_bool"] = out["gave_greedy_answer"].map(_to_bool)
    out["greedy_succeeds_expected_bool"] = out["greedy_succeeds_expected"].map(
        lambda x: x if isinstance(x, bool) else _to_bool(x)
    )
    out["greedy_succeeds"] = np.where(
        out["greedy_succeeds_expected_bool"].isna(),
        np.nan,
        np.where(out["greedy_succeeds_expected_bool"] == True, out["gave_greedy_answer_bool"] == True, out["gave_greedy_answer_bool"] == False),  # noqa: E712
    )

    out["instance_rank_pct"] = out.groupby("problem_subtype")["instance_contamination_score"].rank(method="average", pct=True)
    out["instance_contamination_half"] = np.where(out["instance_rank_pct"] > 0.5, "top", "bottom")
    out["ACI"] = pd.to_numeric(out["CCI_algorithm"], errors="coerce") if "CCI_algorithm" in out.columns else np.nan

    required_for_label = ["VAR_canonical", "VAR_W3", "instance_contamination_score", "greedy_succeeds"]
    out["missing_core"] = out[required_for_label].isna().any(axis=1)
    out["missing_phase2"] = out["CCI_composite"].isna() if "CCI_composite" in out.columns else True
    out["parse_failure_or_missing"] = out["any_parse_failed"].fillna(False).astype(bool)

    out["convergence_label"] = "mixed"
    out.loc[out["missing_core"] | out["parse_failure_or_missing"] | out["missing_phase2"], "convergence_label"] = "ambiguous"

    retrieval_mask = (
        (out["VAR_canonical"] > 0.5)
        & (out["VAR_W3"] < 0.2)
        & (out["instance_contamination_half"] == "top")
        & (out["greedy_succeeds"] == True)  # noqa: E712
        & (out["convergence_label"] != "ambiguous")
    )
    computation_mask = (
        (out["VAR_W3"] > 0.5)
        & (out["ACI"] > 0.5)
        & (out["instance_contamination_half"] == "bottom")
        & (out["convergence_label"] != "ambiguous")
    )
    out.loc[retrieval_mask, "convergence_label"] = "retrieval_signal"
    out.loc[~retrieval_mask & computation_mask, "convergence_label"] = "computation_signal"
    print(f"% ambiguous classifications: {100.0 * (out['convergence_label'] == 'ambiguous').mean():.1f}%")
    return out


def build_table1(df: pd.DataFrame, bw_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subtype, fam in [("coin_change", "CC"), ("shortest_path", "SP"), ("wis", "WIS")]:
        sub = df[df["problem_subtype"] == subtype].copy()
        slope = np.nan
        fit = sub[["VAR_canonical", "instance_contamination_score"]].dropna()
        if len(fit) >= 2 and fit["instance_contamination_score"].nunique() > 1:
            slope = float(np.polyfit(fit["instance_contamination_score"], fit["VAR_canonical"], 1)[0])
        rows.append(
            {
                "family": fam,
                "VAR(canonical)": float(sub["VAR_canonical"].mean(skipna=True)),
                "VAR(W3)": float(sub["VAR_W3"].mean(skipna=True)),
                "VRI_gap": float(sub["VRI_gap"].mean(skipna=True)),
                "GSS/PDAS": float(pd.to_numeric(sub["greedy_succeeds"], errors="coerce").mean(skipna=True)),
                "Contamination-VAR slope": slope,
                "Convergence rate": float((sub["convergence_label"] != "ambiguous").mean()),
            }
        )

    if not bw_metrics.empty:
        bw_row = {
            "family": "BW",
            "VAR(canonical)": np.nan,
            "VAR(W3)": np.nan,
            "VRI_gap": np.nan,
            "GSS/PDAS": np.nan,
            "Contamination-VAR slope": np.nan,
            "Convergence rate": np.nan,
        }
        for key, candidates in {
            "VAR(canonical)": ["var_canonical", "VAR(canonical)", "var"],
            "VAR(W3)": ["var_w3", "VAR(W3)"],
            "VRI_gap": ["vri_gap", "vri"],
            "GSS/PDAS": ["gss", "pdas"],
            "Contamination-VAR slope": ["contamination_var_slope", "slope"],
            "Convergence rate": ["convergence_rate"],
        }.items():
            for c in candidates:
                if c in bw_metrics.columns:
                    bw_row[key] = pd.to_numeric(bw_metrics[c], errors="coerce").dropna().mean()
                    break
        rows.insert(0, bw_row)
    return pd.DataFrame(rows)


def _fit_reg(reg: pd.DataFrame):
    return smf.ols(
        "VAR_canonical ~ instance_contamination_score + C(problem_subtype) + greedy_succeeds_num",
        data=reg,
    ).fit()


def run_regression(df: pd.DataFrame, bootstrap_n: int, rng: np.random.Generator) -> tuple[pd.DataFrame, str]:
    out = []
    text_parts = []
    for model, sub in df.groupby("model"):
        reg = sub.copy()
        reg["greedy_succeeds_num"] = pd.to_numeric(reg["greedy_succeeds"], errors="coerce")
        reg = reg[["VAR_canonical", "instance_contamination_score", "problem_subtype", "greedy_succeeds_num"]].dropna()
        if len(reg) < 8:
            text_parts.append(f"{model}: insufficient rows for OLS (n={len(reg)}).")
            continue
        fit = _fit_reg(reg)
        cont = float(fit.params.get("instance_contamination_score", np.nan))
        greedy = float(fit.params.get("greedy_succeeds_num", np.nan))

        cont_bs = []
        greedy_bs = []
        n = len(reg)
        for _ in range(bootstrap_n):
            idx = rng.integers(0, n, size=n)
            b = reg.iloc[idx].reset_index(drop=True)
            try:
                bf = _fit_reg(b)
                cont_bs.append(float(bf.params.get("instance_contamination_score", np.nan)))
                greedy_bs.append(float(bf.params.get("greedy_succeeds_num", np.nan)))
            except Exception:
                continue
        if not cont_bs or not greedy_bs:
            raise ValueError(f"{model}: bootstrap failed.")
        c_lo, c_hi = float(np.percentile(cont_bs, 2.5)), float(np.percentile(cont_bs, 97.5))
        g_lo, g_hi = float(np.percentile(greedy_bs, 2.5)), float(np.percentile(greedy_bs, 97.5))
        out.append(
            {
                "model": model,
                "coef_instance_contamination_score": cont,
                "coef_instance_contamination_score_ci_lower": c_lo,
                "coef_instance_contamination_score_ci_upper": c_hi,
                "coef_greedy_succeeds": greedy,
                "coef_greedy_succeeds_ci_lower": g_lo,
                "coef_greedy_succeeds_ci_upper": g_hi,
                "n_rows": len(reg),
            }
        )
        text_parts.append(
            f"{model}\n"
            f"  instance_contamination_score: {cont:.4f} (95% CI {c_lo:.4f}, {c_hi:.4f})\n"
            f"  greedy_succeeds: {greedy:.4f} (95% CI {g_lo:.4f}, {g_hi:.4f})\n"
            f"  n={len(reg)}"
        )
    return pd.DataFrame(out), "\n\n".join(text_parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="ALGO P3 triangulation pipeline.")
    parser.add_argument("--behavioral-results", nargs="+", required=True)
    parser.add_argument("--probe2-metrics", required=True)
    parser.add_argument("--contamination", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--bw-metrics", required=False, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--regression-output", required=True)
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    args = parser.parse_args()

    np.random.seed(42)
    rng = np.random.default_rng(42)

    data = load_data(args)
    merged = merge_data(data)
    diagnosed = compute_convergence_labels(merged)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    diagnosed.to_csv(out_path, index=False)

    table1 = build_table1(diagnosed, data["bw_metrics"])
    reg_df, reg_text = run_regression(diagnosed, args.bootstrap_n, rng)

    reg_path = Path(args.regression_output)
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    with reg_path.open("w", encoding="utf-8") as f:
        f.write("ALGO TRIANGULATION REGRESSION SUMMARY\n\n")
        f.write(reg_text + "\n\n")
        f.write("TABLE 1 (CROSS-FAMILY)\n")
        f.write(table1.to_string(index=False))
        f.write("\n\nREGRESSION ROWS\n")
        if reg_df.empty:
            f.write("(none)\n")
        else:
            f.write(reg_df.to_string(index=False))
            f.write("\n")

    print(f"Wrote triangulation dataset: {out_path} ({len(diagnosed)} rows)")
    print(f"Wrote regression summary: {reg_path}")


if __name__ == "__main__":
    main()
