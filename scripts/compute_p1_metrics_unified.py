#!/usr/bin/env python3
"""Unified Probe 1 metrics: CAS, CSS, RCS, W6_Gap, per-variant accuracy."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REAL_MODELS = {
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "meta-llama/llama-3.1-8b-instruct",
}

BW_GSM_SWEEP = REPO_ROOT / "results/raw/BW_P1_behavioral.csv"
ALGO_SWEEPS = [
    REPO_ROOT / "results/raw/ALGO_P1_behavioral_claude.csv",
    REPO_ROOT / "results/raw/ALGO_P1_behavioral_gpt4o.csv",
    REPO_ROOT / "results/raw/ALGO_P1_behavioral_llama.csv",
]
ALGO_BANK = REPO_ROOT / "data/problems/question_bank_algo.csv"
OUT_FAMILY = REPO_ROOT / "results/derived/P1_metrics_by_model_family.csv"
OUT_SUBTYPE = REPO_ROOT / "results/derived/P1_metrics_by_model_subtype.csv"
OUT_PER_PROBLEM = REPO_ROOT / "results/derived/P1_per_problem_var.csv"

SURFACE_VARIANTS = ("W1", "W2", "W3", "W4", "W6")
ALL_VARIANTS = ("canonical", "W1", "W2", "W3", "W4", "W5", "W6")
VARIANT_COLS = [f"acc_{v}" for v in ALL_VARIANTS]
AGG_COLS = [
    "model", "family", "CAS", "CSS", "RCS", "W6_Gap", "VAR_mean",
    *VARIANT_COLS, "n_problems",
]


def _to_bool(val: object) -> bool:
    return str(val).strip().lower() in {"true", "1", "yes"}


def _norm_variant(v: object) -> str:
    s = str(v).strip()
    if not s:
        return s
    low = s.lower()
    if low == "canonical":
        return "canonical"
    if re.fullmatch(r"w[1-6]", low):
        return f"W{low[1]}"
    return s


def _filter_models(df: pd.DataFrame) -> pd.DataFrame:
    m = df["model"].astype(str)
    keep = m.isin(REAL_MODELS) & ~m.str.lower().str.contains("mock", na=False)
    keep &= ~m.str.contains("The answer", na=False, regex=False)
    return df.loc[keep].copy()


def _bw_subtype(problem_id: str) -> str:
    return "mystery_blocksworld" if str(problem_id).strip().upper().startswith("MBW") else "blocksworld"


def _load_bw_gsm() -> pd.DataFrame:
    if not BW_GSM_SWEEP.exists():
        raise FileNotFoundError(BW_GSM_SWEEP)
    raw = pd.read_csv(BW_GSM_SWEEP, dtype=str).fillna("")
    raw = _filter_models(raw)
    raw["correct"] = raw["behavioral_correct"].map(_to_bool)
    raw["variant_type"] = raw["variant_type"].map(_norm_variant)
    fam = raw["problem_family"].astype(str).str.strip().str.lower()
    parts: list[pd.DataFrame] = []
    bw = raw.loc[fam == "planning_suite"].copy()
    if not bw.empty:
        bw["family"], bw["subtype"] = "BW", bw["problem_id"].map(_bw_subtype)
        parts.append(bw)
    gsm = raw.loc[fam == "arithmetic_reasoning"].copy()
    if not gsm.empty:
        gsm["family"], gsm["subtype"] = "GSM", "gsm_symbolic"
        parts.append(gsm)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)[
        ["problem_id", "variant_type", "model", "family", "subtype", "correct"]
    ]


def _load_algo() -> pd.DataFrame:
    frames = [pd.read_csv(p, dtype=str).fillna("") for p in ALGO_SWEEPS if p.exists()]
    if len(frames) != len(ALGO_SWEEPS):
        missing = [p for p in ALGO_SWEEPS if not p.exists()]
        raise FileNotFoundError(f"Missing ALGO sweeps: {missing}")
    raw = pd.concat(frames, ignore_index=True)
    raw = _filter_models(raw)
    raw["correct"] = raw["verified"].map(_to_bool)
    raw["variant_type"] = raw["variant_type"].map(_norm_variant)
    raw["family"] = "ALGO"
    bank = pd.read_csv(ALGO_BANK, dtype=str).fillna("")
    bank["variant_type"] = bank["variant_type"].map(_norm_variant)
    raw = raw.merge(
        bank[["problem_id", "variant_type", "problem_subtype"]].drop_duplicates(),
        on=["problem_id", "variant_type"],
        how="left",
    )
    raw["subtype"] = raw["problem_subtype"].astype(str).str.strip().str.lower()
    if raw["subtype"].eq("").any():
        raise ValueError("ALGO rows missing problem_subtype after bank join")
    return raw[["problem_id", "variant_type", "model", "family", "subtype", "correct"]]


def _problem_table(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["problem_id", "model", "family", "subtype", "variant_type"], dropna=False)["correct"]
        .mean()
        .reset_index(name="acc")
    )


def _per_problem(pt: pd.DataFrame) -> pd.DataFrame:
    idx = ["problem_id", "model", "family", "subtype"]
    wide = pt.pivot_table(index=idx, columns="variant_type", values="acc", aggfunc="first").reset_index()
    for v in ALL_VARIANTS:
        wide[f"acc_{v}"] = wide[v] if v in wide.columns else np.nan
        if v in wide.columns:
            wide = wide.drop(columns=[v])
    var_vals, rcs_vals = [], []
    for _, row in wide.iterrows():
        surf = [float(row[f"acc_{v}"]) for v in SURFACE_VARIANTS if pd.notna(row.get(f"acc_{v}", np.nan))]
        if len(surf) >= 2:
            var_vals.append(max(surf) - min(surf))
        elif len(surf) == 1:
            var_vals.append(0.0)
        else:
            var_vals.append(np.nan)
        canon = row.get("acc_canonical", np.nan)
        rcs_vals.append((float(canon) - min(surf)) / (float(canon) + 1e-8) if pd.notna(canon) and surf else np.nan)
    wide["VAR"], wide["RCS_i"] = var_vals, rcs_vals
    return wide[["problem_id", "model", "family", "subtype", "VAR", "RCS_i", *VARIANT_COLS]]


def _aggregate(df: pd.DataFrame, pp: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["model", *group_cols]
    for key, g in df.groupby(keys, dropna=False):
        key = (key,) if not isinstance(key, tuple) else key
        row = dict(zip(keys, key))
        canon = g[g["variant_type"] == "canonical"]["correct"]
        w6 = g[g["variant_type"] == "W6"]["correct"]
        row["CAS"] = float(canon.mean()) if len(canon) else np.nan
        row["W6_Gap"] = (float(canon.mean()) - float(w6.mean())) if len(canon) and len(w6) else np.nan
        for v in ALL_VARIANTS:
            sub = g[g["variant_type"] == v]["correct"]
            row[f"acc_{v}"] = float(sub.mean()) if len(sub) else np.nan
        mask = pp["model"] == row["model"]
        for col in group_cols:
            mask &= pp[col] == row[col]
        pg = pp.loc[mask]
        vv, rv = pg["VAR"].dropna(), pg["RCS_i"].dropna()
        row["CSS"] = row["VAR_mean"] = float(vv.mean()) if len(vv) else np.nan
        row["RCS"] = float(rv.mean()) if len(rv) else np.nan
        row["n_problems"] = int(pg.loc[pg["VAR"].notna(), "problem_id"].nunique())
        rows.append(row)
    out = pd.DataFrame(rows)
    base = ["model", "family"] + (["subtype"] if "subtype" in out.columns else [])
    return out[base + [c for c in AGG_COLS if c in out.columns and c not in base]]


def _print_summary(family_df: pd.DataFrame) -> None:
    show = family_df.sort_values(["family", "CSS"], ascending=[True, False], na_position="last")
    print("\n=== P1 metrics summary (by model × family) ===")
    print(f"{'Model':<42} {'Family':<6} {'CAS':>7} {'CSS':>7} {'RCS':>7} {'W6_Gap':>8} {'N':>6}")
    print("-" * 88)
    for _, r in show.iterrows():
        fmt = lambda x: f"{x:7.4f}" if pd.notna(x) else "    nan"
        print(f"{r['model']:<42} {r['family']:<6} {fmt(r['CAS'])} {fmt(r['CSS'])} {fmt(r['RCS'])} {fmt(r['W6_Gap'])} {int(r['n_problems']):6d}")


def main() -> None:
    df = pd.concat([_load_bw_gsm(), _load_algo()], ignore_index=True)
    if df.empty:
        raise ValueError("No rows after filtering to REAL_MODELS")
    pt = _problem_table(df)
    pp = _per_problem(pt)
    by_family = _aggregate(df, pp, ["family"])
    by_subtype = _aggregate(df, pp, ["family", "subtype"])
    OUT_FAMILY.parent.mkdir(parents=True, exist_ok=True)
    by_family.to_csv(OUT_FAMILY, index=False)
    by_subtype.to_csv(OUT_SUBTYPE, index=False)
    pp.drop(columns=["RCS_i"]).to_csv(OUT_PER_PROBLEM, index=False)
    print(f"Wrote {OUT_FAMILY} ({len(by_family)} rows)")
    print(f"Wrote {OUT_SUBTYPE} ({len(by_subtype)} rows)")
    print(f"Wrote {OUT_PER_PROBLEM} ({len(pp)} rows)")
    print(f"Input rows (filtered): {len(df)}")
    _print_summary(by_family)


if __name__ == "__main__":
    main()
