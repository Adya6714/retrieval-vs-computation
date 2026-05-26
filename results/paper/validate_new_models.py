#!/usr/bin/env python3
"""Validate new-model sweep outputs and write summary metrics."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("<REPO_ROOT>")
RAW = REPO / "results/raw"
PAPER = REPO / "results/paper"
PAPER.mkdir(parents=True, exist_ok=True)

MODELS = {
    "gemini": "google/gemini-2.5-flash",
    "o1mini": "openai/o1-mini",
}

P1_BW_GSM_COLS = [
    "problem_id", "variant_type", "model", "raw_response",
    "behavioral_correct", "correct_answer", "problem_family",
    "contamination_pole", "difficulty",
]
P1_ALGO_COLS = [
    "problem_id", "variant_type", "model", "model_answer", "ground_truth",
    "verified", "parse_status",
]
GSM_P2_COLS = [
    "problem_id", "model", "cci_score", "tep_score", "session_b_correct",
    "phase1_parseable",
]


def _bool(s) -> bool:
    return str(s).strip().lower() in {"true", "1", "yes"}


def validate_p1_bw_gsm(path: Path, expected: int, ref: Path | None) -> dict:
    out = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        out["error"] = "missing"
        return out
    df = pd.read_csv(path, dtype=str).fillna("")
    out["rows"] = len(df)
    out["cols"] = list(df.columns)
    out["cols_ok"] = list(df.columns) == P1_BW_GSM_COLS
    out["errors"] = int(df["raw_response"].astype(str).str.startswith("ERROR:").sum())
    out["empty_correct"] = int(df["behavioral_correct"].astype(str).str.strip().eq("").sum())
    out["accuracy"] = float(df["behavioral_correct"].map(_bool).mean())
    out["expected_rows"] = expected
    out["complete"] = len(df) >= expected
    if ref and ref.exists():
        out["cols_match_ref"] = list(df.columns) == list(pd.read_csv(ref, nrows=0).columns)
    by_var = df.groupby(df["variant_type"].str.upper())["behavioral_correct"].apply(
        lambda s: s.map(_bool).mean()
    )
    out["acc_by_variant"] = {k: round(float(v), 3) for k, v in by_var.items()}
    return out


def validate_p1_algo(path: Path, expected: int, ref: Path) -> dict:
    out = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        out["error"] = "missing"
        return out
    df = pd.read_csv(path, dtype=str).fillna("")
    ref_cols = list(pd.read_csv(ref, nrows=0).columns)
    out["rows"] = len(df)
    out["cols_match_ref"] = list(df.columns) == ref_cols
    out["errors"] = int(df.get("model_answer", pd.Series(dtype=str)).astype(str).str.startswith("ERROR:").sum())
    out["parse_failed"] = int((df.get("parse_status", "") == "parse_failed").sum()) if "parse_status" in df.columns else None
    if "verified" in df.columns:
        out["verified_rate"] = float(df["verified"].map(_bool).mean())
    out["expected_rows"] = expected
    out["complete"] = len(df) >= expected
    return out


def validate_gsm_p2(path: Path, expected: int = 44) -> dict:
    out = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        out["error"] = "missing"
        return out
    df = pd.read_csv(path, dtype=str).fillna("")
    out["rows"] = len(df)
    out["cols_ok"] = all(c in df.columns for c in GSM_P2_COLS)
    out["accuracy"] = float(df["session_b_correct"].map(_bool).mean())
    out["cci_mean"] = float(pd.to_numeric(df["cci_score"], errors="coerce").mean())
    out["tep_mean"] = float(pd.to_numeric(df["tep_score"], errors="coerce").mean())
    out["phase1_unparseable"] = int((df["phase1_parseable"].map(_bool) == False).sum())
    out["expected_rows"] = expected
    out["complete"] = len(df) >= expected
    return out


def var_table(path: Path, family_label: str) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str).fillna("")
    col = "verified" if "verified" in df.columns else "behavioral_correct"
    df["correct"] = df[col].map(_bool)
    df["variant_type"] = df["variant_type"].str.upper().replace({"CANONICAL": "canonical"})
  # normalize W variants
    def norm_v(v):
        s = str(v).strip()
        if s.lower() == "canonical":
            return "canonical"
        if re.fullmatch(r"[wW][1-6]", s):
            return f"W{s[-1]}"
        return s
    df["variant_type"] = df["variant_type"].map(norm_v)
    return df.groupby("variant_type")["correct"].mean().to_frame("accuracy")


def main() -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report: dict = {"generated_at": ts, "models": MODELS, "validation": {}, "var_tables": {}}

    ref_gsm = RAW / "GSM_P1_behavioral_claude.csv"
    ref_algo = RAW / "ALGO_P1_behavioral_claude.csv"

    report["validation"]["gemini_gsm_p1"] = validate_p1_bw_gsm(
        RAW / "GSM_P1_behavioral_gemini.csv", 288, ref_gsm
    )
    report["validation"]["gemini_bw_p1"] = validate_p1_bw_gsm(
        RAW / "BW_P1_behavioral_gemini.csv", 455, RAW / "BW_P1_behavioral.csv"
    )
    report["validation"]["gemini_algo_p1"] = validate_p1_algo(
        RAW / "ALGO_P1_behavioral_gemini.csv", 640, ref_algo
    )
    report["validation"]["o1mini_gsm_p1"] = validate_p1_bw_gsm(
        RAW / "GSM_P1_behavioral_o1mini.csv", 288, ref_gsm
    )
    report["validation"]["o1mini_bw_p1"] = validate_p1_bw_gsm(
        RAW / "BW_P1_behavioral_o1mini.csv", 455, None
    )
    report["validation"]["o1mini_algo_p1"] = validate_p1_algo(
        RAW / "ALGO_P1_behavioral_o1mini.csv", 640, ref_algo
    )
    report["validation"]["gemini_gsm_p2"] = validate_gsm_p2(RAW / "GSM_P2_phase1_gemini.csv")
    report["validation"]["o1mini_gsm_p2"] = validate_gsm_p2(RAW / "GSM_P2_phase1_o1mini.csv")

    for name, path in [
        ("GSM_gemini", RAW / "GSM_P1_behavioral_gemini.csv"),
        ("GSM_o1mini", RAW / "GSM_P1_behavioral_o1mini.csv"),
        ("BW_gemini", RAW / "BW_P1_behavioral_gemini.csv"),
        ("ALGO_gemini", RAW / "ALGO_P1_behavioral_gemini.csv"),
        ("ALGO_o1mini", RAW / "ALGO_P1_behavioral_o1mini.csv"),
    ]:
        vt = var_table(path, name)
        if vt is not None:
            report["var_tables"][name] = vt.round(3).to_dict()

    out_json = PAPER / "new_models_validation_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [f"# New models validation report — {ts}\n"]
    for k, v in report["validation"].items():
        lines.append(f"## {k}\n```json\n{json.dumps(v, indent=2)}\n```\n")
    (PAPER / "new_models_validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_json}")
    print(json.dumps(report["validation"], indent=2))


if __name__ == "__main__":
    main()
