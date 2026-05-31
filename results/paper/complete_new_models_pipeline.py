#!/usr/bin/env python3
"""Dedupe new-model raw CSVs and write paper metrics for gemini + o4-mini."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RAW = REPO / "results/raw"
PAPER = REPO / "results/paper"
PAPER.mkdir(parents=True, exist_ok=True)

O4_MODEL = "openai/o4-mini"
GEMINI_MODEL = "google/gemini-2.5-flash"


def _bool(s: object) -> bool:
    return str(s).strip().lower() in {"true", "1", "yes"}


def _norm_variant(v: object) -> str:
    s = str(v).strip()
    if s.lower() == "canonical":
        return "canonical"
    if re.fullmatch(r"[wW][1-6]", s):
        return f"W{s[-1]}"
    return s


def dedupe_csv(path: Path, key_cols: list[str]) -> int:
    if not path.exists():
        return 0
    df = pd.read_csv(path, dtype=str).fillna("")
    n0 = len(df)
    # Prefer last row; drop ERROR rows when a good row exists for same key.
    err_col = "raw_response" if "raw_response" in df.columns else "model_answer"
    df["_is_err"] = df[err_col].astype(str).str.startswith("ERROR:")
    df["_ord"] = np.arange(len(df))
    df = df.sort_values(["_is_err", "_ord"]).drop_duplicates(subset=key_cols, keep="first")
    df = df.drop(columns=["_is_err", "_ord"])
    df.to_csv(path, index=False)
    return n0 - len(df)


def filter_model(path: Path, model: str, out: Path | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    if "model" not in df.columns:
        return df
    sub = df[df["model"] == model].copy()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        sub.to_csv(out, index=False)
    return sub


def p1_var_table(df: pd.DataFrame, label: str) -> pd.DataFrame:
    col = "verified" if "verified" in df.columns else "behavioral_correct"
    df = df.copy()
    df["correct"] = df[col].map(_bool)
    df["variant_type"] = df["variant_type"].map(_norm_variant)
    vt = df.groupby("variant_type")["correct"].mean().to_frame(label)
    return vt


def gsm_p2_metrics(path: Path) -> dict:
    if not path.exists():
        return {"error": "missing"}
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[df["model"] == O4_MODEL] if "model" in df.columns else df
    return {
        "n": len(df),
        "accuracy": float(df["session_b_correct"].map(_bool).mean()) if len(df) else None,
        "cci_mean": float(pd.to_numeric(df["cci_score"], errors="coerce").mean()) if len(df) else None,
        "tep_mean": float(pd.to_numeric(df["tep_score"], errors="coerce").mean()) if len(df) else None,
        "phase1_parseable": float(df["phase1_parseable"].map(_bool).mean()) if len(df) else None,
    }


def coverage_report() -> dict:
    report: dict = {"generated_at": datetime.now(timezone.utc).isoformat(), "files": {}}
    specs = [
        ("GSM_P1_behavioral_gemini.csv", GEMINI_MODEL, 288, ["problem_id", "variant_type", "model"]),
        ("BW_P1_behavioral_gemini.csv", GEMINI_MODEL, 455, ["problem_id", "variant_type", "model"]),
        ("ALGO_P1_behavioral_gemini.csv", GEMINI_MODEL, 640, ["problem_id", "variant_type", "model"]),
        ("GSM_P2_phase1_gemini.csv", GEMINI_MODEL, 44, ["problem_id", "model"]),
        ("ALGO_P2_phase1_gemini.csv", GEMINI_MODEL, 110, ["problem_id", "model"]),
        ("GSM_P1_behavioral_o1mini.csv", O4_MODEL, 288, ["problem_id", "variant_type", "model"]),
        ("BW_P1_behavioral_o1mini.csv", O4_MODEL, 455, ["problem_id", "variant_type", "model"]),
        ("ALGO_P1_behavioral_o1mini.csv", O4_MODEL, 640, ["problem_id", "variant_type", "model"]),
        ("GSM_P2_phase1_o1mini.csv", O4_MODEL, 44, ["problem_id", "model"]),
    ]
    for fname, model, target, keys in specs:
        p = RAW / fname
        entry: dict = {"target": target, "model": model}
        if not p.exists():
            entry["status"] = "missing"
            report["files"][fname] = entry
            continue
        df = pd.read_csv(p, dtype=str).fillna("")
        if "model" in df.columns:
            df = df[df["model"] == model]
        if "raw_response" in df.columns:
            err_col = "raw_response"
        elif "model_answer" in df.columns:
            err_col = "model_answer"
        else:
            err_col = None
        if err_col:
            good = df[~df[err_col].astype(str).str.startswith("ERROR:")]
            n_err = int(df[err_col].astype(str).str.startswith("ERROR:").sum())
        else:
            good = df
            n_err = 0
        if "variant_type" in keys:
            n_good = good.groupby(["problem_id", "variant_type"]).ngroups
        else:
            n_good = good["problem_id"].nunique()
        entry.update(
            {
                "rows": len(df),
                "good_keys": int(n_good),
                "errors": n_err,
                "complete": n_good >= target,
            }
        )
        report["files"][fname] = entry
    return report


def main() -> None:
    # Dedupe gemini ALGO P1
    dropped = dedupe_csv(
        RAW / "ALGO_P1_behavioral_gemini.csv",
        ["problem_id", "variant_type", "model"],
    )
    print(f"Deduped ALGO_P1_behavioral_gemini.csv: removed {dropped} rows")

    report = coverage_report()
    (PAPER / "new_models_coverage.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # VAR tables
    var_frames = []
    for slug, model, path in [
        ("gemini", GEMINI_MODEL, RAW / "GSM_P1_behavioral_gemini.csv"),
        ("o4mini", O4_MODEL, RAW / "GSM_P1_behavioral_o1mini.csv"),
    ]:
        if path.exists():
            df = pd.read_csv(path, dtype=str).fillna("")
            df = df[df["model"] == model]
            var_frames.append(p1_var_table(df, slug))
    if var_frames:
        pd.concat(var_frames, axis=1).round(3).to_csv(PAPER / "GSM_VAR_gemini_o4mini.csv")

    # GSM P2 metrics
    p2 = {
        "gemini": gsm_p2_metrics(RAW / "GSM_P2_phase1_gemini.csv"),
        "o4mini": gsm_p2_metrics(RAW / "GSM_P2_phase1_o1mini.csv"),
    }
    (PAPER / "GSM_P2_metrics_new_models.json").write_text(json.dumps(p2, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nWrote {PAPER / 'new_models_coverage.json'}")


if __name__ == "__main__":
    main()
