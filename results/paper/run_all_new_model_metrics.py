#!/usr/bin/env python3
"""Run post-sweep metrics for gemini + o4-mini (partial OK)."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

BASE = REPO
RAW = BASE / "results/raw"
PAPER = BASE / "results/paper"
PAPER.mkdir(parents=True, exist_ok=True)

MODELS = {
    "claude": "anthropic/claude-sonnet-4",
    "gpt4o": "openai/gpt-4o",
    "llama": "meta-llama/llama-3.1-8b-instruct",
    "gemini": "google/gemini-2.5-flash",
    "o4mini": "openai/o4-mini",
}


def _bool(s) -> bool:
    return str(s).strip().lower() in {"true", "1", "yes"}


def load_p1(name: str) -> pd.DataFrame | None:
    if name in ("claude", "gpt4o", "llama"):
        if name == "claude":
            paths = [RAW / "GSM_P1_behavioral_claude.csv", RAW / "ALGO_P1_behavioral_claude.csv"]
        elif name == "gpt4o":
            paths = [RAW / "GSM_P1_behavioral_gpt4o.csv", RAW / "ALGO_P1_behavioral_gpt4o.csv"]
        else:
            paths = [RAW / "GSM_P1_behavioral_llama.csv", RAW / "ALGO_P1_behavioral_llama.csv"]
        frames = [pd.read_csv(p, dtype=str) for p in paths if p.exists()]
        return pd.concat(frames) if frames else None
    path = RAW / f"GSM_P1_behavioral_{name}.csv"
    if name == "gemini":
        gsm = RAW / "GSM_P1_behavioral_gemini.csv"
        bw = RAW / "BW_P1_behavioral_gemini.csv"
        algo = RAW / "ALGO_P1_behavioral_gemini.csv"
        frames = [pd.read_csv(p, dtype=str) for p in [gsm, bw, algo] if p.exists()]
        return pd.concat(frames) if frames else None
    if name == "o4mini":
        frames = []
        for p in [RAW / "GSM_P1_behavioral_o1mini.csv", RAW / "BW_P1_behavioral_o1mini.csv", RAW / "ALGO_P1_behavioral_o1mini.csv"]:
            if p.exists():
                df = pd.read_csv(p, dtype=str)
                df = df[df["model"] == MODELS["o4mini"]]
                frames.append(df)
        return pd.concat(frames) if frames else None
    return pd.read_csv(path, dtype=str) if path.exists() else None


def var_tables() -> None:
    for family, prefix, names in [
        ("GSM", "GSM", ["claude", "gpt4o", "llama", "gemini", "o4mini"]),
        ("BW", "BW", ["gemini", "o4mini"]),
        ("ALGO", "ALGO", ["claude", "gpt4o", "llama", "gemini", "o4mini"]),
    ]:
        frames = []
        for n in names:
            if family == "GSM" and n in ("claude", "gpt4o", "llama"):
                p = RAW / f"GSM_P1_behavioral_{n}.csv"
            elif family == "BW":
                if n == "gemini":
                    p = RAW / "BW_P1_behavioral_gemini.csv"
                else:
                    p = RAW / "BW_P1_behavioral_o1mini.csv"
            elif family == "ALGO":
                if n in ("claude", "gpt4o", "llama", "gemini"):
                    p = RAW / f"ALGO_P1_behavioral_{n}.csv"
                else:
                    p = RAW / "ALGO_P1_behavioral_o1mini.csv"
            else:
                p = RAW / f"GSM_P1_behavioral_{n}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p, dtype=str)
            if n == "o4mini":
                df = df[df["model"] == MODELS["o4mini"]]
            col = "verified" if "verified" in df.columns else "behavioral_correct"
            err_col = "raw_response" if "raw_response" in df.columns else "model_answer"
            df = df[~df[err_col].astype(str).str.startswith("ERROR:")]
            df["model_name"] = n
            df["correct"] = df[col].map(_bool)
            frames.append(df)
        if not frames:
            continue
        combined = pd.concat(frames)
        vt = combined.groupby(["model_name", "variant_type"])["correct"].mean().unstack()
        out = PAPER / f"{prefix}_VAR_all_models.csv"
        vt.round(3).to_csv(out)
        print(f"Wrote {out}")


def gsm_p2_metrics() -> None:
    rows = []
    for name in ["claude", "gpt4o", "llama", "gemini", "o4mini"]:
        if name == "o4mini":
            p = RAW / "GSM_P2_phase1_o1mini.csv"
        else:
            p = RAW / f"GSM_P2_phase1_{name}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, dtype=str)
        if name == "o4mini":
            df = df[df["model"] == MODELS["o4mini"]]
        if df.empty:
            continue
        df["correct"] = df["session_b_correct"].map(_bool)
        cci = pd.to_numeric(df["cci_score"], errors="coerce")
        tep = pd.to_numeric(df["tep_score"], errors="coerce")
        rows.append(
            {
                "model": name,
                "n": len(df),
                "accuracy": df["correct"].mean(),
                "cci_mean": cci.mean(),
                "tep_mean": tep.mean(),
            }
        )
    if rows:
        res = pd.DataFrame(rows).sort_values("accuracy", ascending=False)
        res.to_csv(PAPER / "TEP_ordering_all_models.csv", index=False)
        print(res.to_string())
        print(f"Wrote {PAPER / 'TEP_ordering_all_models.csv'}")


def main() -> None:
    var_tables()
    gsm_p2_metrics()
    summary = {"models": MODELS, "note": "o4mini rows exclude ERROR responses"}
    (PAPER / "evaluation_summary.json").write_text(json.dumps(summary, indent=2))
    print("Metrics pass complete.")


if __name__ == "__main__":
    main()
