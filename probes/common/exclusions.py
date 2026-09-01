"""Instrument exclusions: drop a row only for a documented reason, never the verdict."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EXCLUSIONS_PATH = REPO_ROOT / "results/derived/variant_exclusions.csv"


@lru_cache(maxsize=1)
def load_exclusions() -> pd.DataFrame:
    if not EXCLUSIONS_PATH.exists():
        return pd.DataFrame(columns=["family", "problem_id", "variant", "reason"])
    df = pd.read_csv(EXCLUSIONS_PATH, dtype=str).fillna("")
    df["family"] = df["family"].astype(str).str.strip().str.upper()
    df["problem_id"] = df["problem_id"].astype(str).str.strip()
    df["variant"] = df["variant"].astype(str).str.strip()
    return df


def exclusion_index() -> set[tuple[str, str, str]]:
    df = load_exclusions()
    return {
        (str(r.family), str(r.problem_id), str(r.variant))
        for r in df.itertuples(index=False)
    }


def filter_excluded(
    df: pd.DataFrame,
    *,
    family: str,
    problem_col: str = "problem_id",
    variant_col: str = "variant_type",
) -> pd.DataFrame:
    """Drop rows listed in variant_exclusions.csv for this family."""
    idx = exclusion_index()
    if not idx or df.empty:
        return df
    fam = str(family).upper()
    vt = df[variant_col].astype(str).str.strip()
    vt = vt.where(~vt.str.lower().eq("canonical"), "canonical")
    vt = vt.where(~vt.str.lower().str.fullmatch(r"w[1-6]"), vt.str.upper())
    pid = df[problem_col].astype(str).str.strip()
    drop = [(fam, p, v) in idx for p, v in zip(pid, vt)]
    return df.loc[~pd.Series(drop, index=df.index)].copy()
