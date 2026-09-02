#!/usr/bin/env python3
"""M1: correlate per-instance canonical-minus-W6 gap with contamination_score.

Within-family only (BW and ALGO separately; do not pool — Infini-gram windows
differ). Spearman rho per model with cluster-bootstrap CIs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.clones import cluster_ids_for  # noqa: E402
from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
RAW = REPO_ROOT / "results" / "raw"
OUT = DER / "M1_w6_gap_vs_contamination.csv"

PAPER_MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
N_BOOT = 5000
SEED = 42
FAMILIES = ("BW", "ALGO")


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _load_p1() -> pd.DataFrame:
    parts = []
    for path in sorted(DER.glob("*_P1_*rescored.csv")):
        if "review" in path.name.lower():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "included" not in df.columns:
            continue
        df = df[_is_true(df["included"])].copy()
        if path.name.startswith("ALGO_"):
            fam = "ALGO"
        elif path.name.startswith("BW_"):
            fam = "BW"
        else:
            continue
        df = filter_excluded(df, family=fam)
        df["family"] = fam
        df["model_short"] = df["model"].map(PAPER_MODELS).fillna(df["model"])
        df["variant"] = df["variant_type"].map(normalize_variant)
        ok = df["rescored_correct"] if "rescored_correct" in df.columns else df.get("verified", "")
        df["ok"] = _is_true(ok)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["family", "problem_id", "variant", "model_short"], keep="last")


def _load_contamination() -> pd.DataFrame:
    rows = []
    for fam, path in [("BW", RAW / "BW_P3_contamination.csv"), ("ALGO", RAW / "ALGO_P3_contamination.csv")]:
        df = pd.read_csv(path, dtype=str).fillna("")
        df["family"] = fam
        df["contamination_score"] = pd.to_numeric(df["contamination_score"], errors="coerce")
        rows.append(df[["family", "problem_id", "contamination_score"]])
    return pd.concat(rows, ignore_index=True).drop_duplicates(["family", "problem_id"])


def _instance_frame(p1: pd.DataFrame, contam: pd.DataFrame) -> pd.DataFrame:
    can = p1[p1["variant"] == "canonical"][["family", "problem_id", "model_short", "ok"]].rename(
        columns={"ok": "canonical_ok"},
    )
    w6 = p1[p1["variant"] == "W6"][["family", "problem_id", "model_short", "ok"]].rename(
        columns={"ok": "w6_ok"},
    )
    merged = can.merge(w6, on=["family", "problem_id", "model_short"], how="inner")
    merged["w6_gap"] = merged["canonical_ok"].astype(int) - merged["w6_ok"].astype(int)
    merged = merged.merge(contam, on=["family", "problem_id"], how="left")
    merged = merged[merged["contamination_score"].notna()].copy()
    merged["cluster_id"] = merged["problem_id"].astype(str)
    algo_mask = merged["family"] == "ALGO"
    if algo_mask.any():
        merged.loc[algo_mask, "cluster_id"] = cluster_ids_for(
            merged.loc[algo_mask, "problem_id"].astype(str).tolist(),
        )
    return merged


def _cluster_bootstrap_spearman(sub: pd.DataFrame) -> tuple[float, float, float]:
    x = sub["contamination_score"].astype(float)
    y = sub["w6_gap"].astype(float)
    if len(sub) < 5 or x.nunique() < 2 or y.nunique() < 2:
        rho, _ = stats.spearmanr(x, y)
        return float(rho), float("nan"), float("nan")
    rho, _ = stats.spearmanr(x, y)
    clusters = sorted(sub["cluster_id"].astype(str).unique())
    grouped = {c: sub[sub["cluster_id"].astype(str) == c] for c in clusters}
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        draw = rng.choice(clusters, size=len(clusters), replace=True)
        chunk = pd.concat([grouped[c] for c in draw], ignore_index=True)
        if len(chunk) < 5 or chunk["contamination_score"].nunique() < 2 or chunk["w6_gap"].nunique() < 2:
            boots[i] = float("nan")
        else:
            boots[i], _ = stats.spearmanr(
                chunk["contamination_score"].astype(float),
                chunk["w6_gap"].astype(float),
            )
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return float(rho), float("nan"), float("nan")
    return float(rho), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    p1 = _load_p1()
    contam = _load_contamination()
    frame = _instance_frame(p1, contam)

    rows = []
    for fam in FAMILIES:
        for model in sorted(frame.loc[frame["family"] == fam, "model_short"].dropna().unique()):
            sub = frame[(frame["family"] == fam) & (frame["model_short"] == model)].copy()
            if sub.empty:
                continue
            rho, lo, hi = _cluster_bootstrap_spearman(sub)
            _, p = stats.spearmanr(sub["contamination_score"], sub["w6_gap"])
            rows.append(
                {
                    "family": fam,
                    "model": model,
                    "n": len(sub),
                    "spearman_rho": round(rho, 4) if rho == rho else "",
                    "ci_low": round(lo, 4) if lo == lo else "",
                    "ci_high": round(hi, 4) if hi == hi else "",
                    "p_value": round(float(p), 4) if p == p else "",
                    "contamination_column": "contamination_score",
                    "gap_definition": "canonical_ok_minus_w6_ok",
                    "bootstrap": "cluster_by_clone_family" if fam == "ALGO" else "cluster_by_problem_id",
                    "n_boot": N_BOOT,
                    "seed": SEED,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(out)} rows)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
