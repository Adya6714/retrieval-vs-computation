from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


P1_PRIMARY_BLUE = "#1D4ED8"
P1_ACCENT_BLUE = "#3B82F6"
P1_LIGHT_BLUE = "#DBEAFE"
P3_GOLD = "#F59E0B"
P3_CORAL = "#EF4444"
GREEN = "#10B981"
GRAY = "#9CA3AF"

MODEL_ORDER = [
    "anthropic/claude-3.7-sonnet",
    "openai/gpt-4o",
    "meta-llama/llama-3.1-8b-instruct",
]
MODEL_LABELS = {
    "anthropic/claude-3.7-sonnet": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
}

VARIANT_ORDER = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]


def load_csv_candidate(paths: Iterable[str]) -> pd.DataFrame:
    for p in paths:
        path = Path(p)
        if path.exists():
            return pd.read_csv(path, dtype=str)
    raise FileNotFoundError(f"None of these files exist: {list(paths)}")


def load_behavioral() -> pd.DataFrame:
    df = load_csv_candidate(
        [
            "results/BW_RES_P1_behavioral_sweep.csv",
            "results/behavioral_sweep.csv",
            "data/BW_RES_P1_behavioral_sweep.csv",
            "data/behavioral_sweep.csv",
        ]
    )
    if "problem_id" in df.columns:
        df = df[~df["problem_id"].astype(str).str.contains("_W5_TEMP", na=False)].copy()
    if "model" in df.columns:
        df = df[df["model"].astype(str) != "meta-llama/llama-3-8b-instruct"].copy()
        df = df[df["model"].astype(str).str.lower() != "mock"].copy()
    return df


def load_qb() -> pd.DataFrame:
    return pd.read_csv("data/problems/question_bank.csv", dtype=str)


def output_dir() -> Path:
    out = Path("analysis/figures/output")
    out.mkdir(parents=True, exist_ok=True)
    return out


def to_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map({"true": True, "false": False})


def compute_var(df: pd.DataFrame, model: str, variant_type: str, family: str | None = None) -> float | None:
    x = df.copy()
    x = x[x["model"].astype(str) == model]
    x = x[x["variant_type"].astype(str).str.upper() == variant_type.upper()]
    if family is not None:
        x = x[x["problem_family"].astype(str) == family]
    if len(x) == 0:
        return None
    vals = to_bool_series(x["behavioral_correct"]).dropna()
    if len(vals) == 0:
        return None
    return float(vals.mean())


def bootstrap_ci_bool(values: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05) -> tuple[float, float]:
    if len(values) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    samples = values[idx].mean(axis=1)
    lo = float(np.quantile(samples, alpha / 2))
    hi = float(np.quantile(samples, 1 - alpha / 2))
    return lo, hi


def add_problem_family_from_qb(df: pd.DataFrame) -> pd.DataFrame:
    qb = load_qb()
    base = qb[qb["variant_type"].astype(str).str.upper() == "CANONICAL"][
        ["problem_id", "problem_subtype"]
    ].copy()
    base = base.rename(columns={"problem_subtype": "problem_family"})
    merged = df.merge(base, on="problem_id", how="left")
    merged["problem_family"] = merged["problem_family"].fillna("")
    return merged


def pearson_r_p(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # Lightweight approx p-value with t-stat (sufficient for plotting annotation).
    if len(x) < 3:
        return (np.nan, np.nan)
    r = float(np.corrcoef(x, y)[0, 1])
    if np.isnan(r):
        return (np.nan, np.nan)
    n = len(x)
    if abs(r) >= 1.0:
        return (r, 0.0)
    t = r * np.sqrt((n - 2) / (1 - r * r))
    # Normal approximation for two-tailed p as fallback to avoid scipy dependency.
    # p ≈ 2 * (1 - Phi(|t|))
    from math import erf, sqrt

    phi = 0.5 * (1 + erf(abs(t) / sqrt(2)))
    p = 2 * (1 - phi)
    return (r, float(max(min(p, 1.0), 0.0)))
