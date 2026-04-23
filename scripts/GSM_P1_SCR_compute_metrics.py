#!/usr/bin/env python3
"""Compute GSM Probe 1 metrics from behavioral sweep outputs."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.behavioral.css import compute_css
from probes.behavioral.gsm_metrics import (
    rcs_by_difficulty,
    step_count_sensitivity,
    vri,
    w4_gap,
)
from probes.behavioral.rcs import compute_rcs
from probes.behavioral.var import compute_var


RESULTS_DIR = Path("results")
QUESTION_BANK_PATH = Path("data/problems/gsm_question_bank.csv")


def _to_bool_series(series: pd.Series) -> pd.Series:
    mapped = (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
    )
    return mapped.dropna().astype(bool)


def _load_behavioral_results() -> pd.DataFrame:
    files = sorted(RESULTS_DIR.glob("GSM_P1_RES_behavioral_sweep_*.csv"))
    if not files:
        raise FileNotFoundError(
            "No files found matching results/GSM_P1_RES_behavioral_sweep_*.csv"
        )
    frames = [pd.read_csv(path, dtype=str) for path in files]
    df = pd.concat(frames, ignore_index=True)
    return df


def _compute_css_table(df: pd.DataFrame, qb: pd.DataFrame) -> pd.DataFrame:
    required = {"problem_id", "model", "variant_type", "raw_response", "correct_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"behavioral results missing required columns for CSS: {sorted(missing)}")

    contamination_lookup = (
        qb[qb["variant_type"].astype(str).str.lower() == "canonical"][
            ["problem_id", "contamination_pole", "difficulty"]
        ]
        .drop_duplicates("problem_id")
        .set_index("problem_id")
    )

    rows: list[dict] = []
    for (model, problem_id), group in df.groupby(["model", "problem_id"], dropna=False):
        family = str(group["problem_family"].iloc[0])
        canonical = group[group["variant_type"].astype(str).str.lower() == "canonical"]
        if canonical.empty:
            continue
        canonical_answer = str(canonical.iloc[0]["correct_answer"])

        variants = group[group["variant_type"].astype(str).str.lower().isin(["w1", "w2", "w3", "w4"])]
        variant_payload = []
        for _, r in variants.iterrows():
            variant_payload.append(
                {
                    "variant_type": str(r["variant_type"]),
                    "model_answer": str(r["raw_response"]),
                    "correct_answer": str(r["correct_answer"]),
                    "problem_text": "",
                }
            )
        css_result = compute_css(
            problem_id=str(problem_id),
            canonical_answer=canonical_answer,
            variant_responses=variant_payload,
            family=family,
        )
        meta = contamination_lookup.loc[problem_id] if problem_id in contamination_lookup.index else None
        rows.append(
            {
                "problem_id": str(problem_id),
                "model": str(model),
                "problem_family": family,
                "css": css_result["css"],
                "variants_evaluated": css_result["variants_evaluated"],
                "variants_correct": css_result["variants_correct"],
                "contamination_pole": "" if meta is None else str(meta["contamination_pole"]),
                "difficulty": "" if meta is None else str(meta["difficulty"]),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "problem_id",
                "model",
                "problem_family",
                "css",
                "variants_evaluated",
                "variants_correct",
                "contamination_pole",
                "difficulty",
            ]
        )
    return out.sort_values(["model", "problem_id"]).reset_index(drop=True)


def _compute_rcs_table(df: pd.DataFrame) -> pd.DataFrame:
    w6 = df[df["variant_type"].astype(str).str.lower() == "w6"].copy()
    rows: list[dict] = []
    for _, r in w6.iterrows():
        pid = str(r["problem_id"])
        family = str(r["problem_family"])
        model = str(r["model"])
        raw_response = str(r["raw_response"])
        correct_answer = str(r["correct_answer"])
        # Existing rcs.py defines RCS via reversal correctness. For GSM, family may be
        # outside its strict whitelist; in that case fall back to behavioral correctness.
        try:
            rcs_res = compute_rcs(
                problem_id=pid,
                w5_model_answer=raw_response,
                w5_correct_answer=correct_answer,
                family=family,
            )
            rcs_val = bool(rcs_res["w5_correct"])
        except ValueError:
            rcs_val = bool(
                str(r.get("behavioral_correct", "")).strip().lower() == "true"
            )
        rows.append(
            {
                "problem_id": pid,
                "model": model,
                "variant_type": "w6",
                "rcs": rcs_val,
                "difficulty": str(r.get("difficulty", "")),
                "contamination_pole": str(r.get("contamination_pole", "")),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["model", "problem_id"]).reset_index(drop=True) if not out.empty else out


def _write_table(name: str, df: pd.DataFrame) -> Path:
    out_path = RESULTS_DIR / f"GSM_P1_RES_{name}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    behavioral = _load_behavioral_results()
    question_bank = pd.read_csv(QUESTION_BANK_PATH, dtype=str)

    var_df = compute_var(behavioral)
    css_df = _compute_css_table(behavioral, question_bank)
    rcs_df = _compute_rcs_table(behavioral)
    w4_gap_df = w4_gap(behavioral)
    vri_df = vri(behavioral)
    rcs_diff_df = rcs_by_difficulty(behavioral)
    step_css_df = step_count_sensitivity(behavioral, question_bank)

    outputs = {
        "var": var_df,
        "css": css_df,
        "rcs": rcs_df,
        "w4_gap": w4_gap_df,
        "vri": vri_df,
        "rcs_by_difficulty": rcs_diff_df,
        "step_count_sensitivity": step_css_df,
    }

    written: list[tuple[str, Path, int]] = []
    for name, table in outputs.items():
        path = _write_table(name, table)
        written.append((name, path, len(table)))

    print("Metric summary:")
    print("metric_name,rows,output_path")
    for name, path, count in written:
        print(f"{name},{count},{path}")


if __name__ == "__main__":
    main()
