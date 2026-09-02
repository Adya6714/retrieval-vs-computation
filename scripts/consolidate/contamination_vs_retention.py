#!/usr/bin/env python3
"""N5: Spearman contamination vs W3 retention (per family, per model).

Tests the field claim that contaminated instances show lower W3 retention.
Retention per instance = W3 correct when canonical correct (else excluded).
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
OUT = DER / "N5_contamination_vs_retention.csv"

PAPER_MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
N_BOOT = 5000
SEED = 42
FAMILIES = ("GSM", "ALGO", "BW")


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
        elif path.name.startswith("GSM_"):
            fam = "GSM"
        else:
            continue
        df = filter_excluded(df, family=fam)
        df["family"] = fam
        df["model_short"] = df["model"].map(PAPER_MODELS).fillna(df["model"])
        df["variant"] = df["variant_type"].map(normalize_variant)
        ok = df["rescored_correct"] if "rescored_correct" in df.columns else df.get("verified", "")
        df["ok"] = _is_true(ok)
        parts.append(df)
    return pd.concat(parts, ignore_index=True).drop_duplicates(
        ["family", "problem_id", "variant", "model_short"], keep="last",
    )


def _load_contamination() -> pd.DataFrame:
    rows = []
    for fam, path in [
        ("GSM", RAW / "GSM_P3_contamination.csv"),
        ("BW", RAW / "BW_P3_contamination.csv"),
        ("ALGO", RAW / "ALGO_P3_contamination.csv"),
    ]:
        df = pd.read_csv(path, dtype=str).fillna("")
        df["family"] = fam
        df["contamination_score"] = pd.to_numeric(df["contamination_score"], errors="coerce")
        rows.append(df[["family", "problem_id", "contamination_score"]])
    return pd.concat(rows, ignore_index=True).drop_duplicates(["family", "problem_id"])


def _instance_frame(p1: pd.DataFrame, contam: pd.DataFrame) -> pd.DataFrame:
    can = p1[p1["variant"] == "canonical"][["family", "problem_id", "model_short", "ok"]].rename(
        columns={"ok": "canonical_ok"},
    )
    w3 = p1[p1["variant"] == "W3"][["family", "problem_id", "model_short", "ok"]].rename(
        columns={"ok": "w3_ok"},
    )
    merged = can.merge(w3, on=["family", "problem_id", "model_short"], how="inner")
    merged = merged[merged["canonical_ok"]].copy()
    merged["retained_w3"] = merged["w3_ok"].astype(int)
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
    y = sub["retained_w3"].astype(float)
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
        if len(chunk) < 5 or chunk["contamination_score"].nunique() < 2 or chunk["retained_w3"].nunique() < 2:
            boots[i] = float("nan")
        else:
            boots[i], _ = stats.spearmanr(
                chunk["contamination_score"].astype(float),
                chunk["retained_w3"].astype(float),
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
            if sub.empty or len(sub) < 5:
                continue
            if sub["retained_w3"].nunique() < 2 or sub["contamination_score"].nunique() < 2:
                rows.append(
                    {
                        "family": fam,
                        "model": model,
                        "n": len(sub),
                        "spearman_rho": "",
                        "ci_low": "",
                        "ci_high": "",
                        "p_value": "",
                        "subset": "canonical_correct_only",
                        "y": "w3_retained",
                        "contamination_column": "contamination_score",
                        "bootstrap": "cluster_by_clone_family" if fam == "ALGO" else "cluster_by_problem_id",
                        "n_boot": N_BOOT,
                        "seed": SEED,
                        "note": "insufficient variation in retention or contamination",
                    }
                )
                continue
            rho, lo, hi = _cluster_bootstrap_spearman(sub)
            _, p = stats.spearmanr(sub["contamination_score"], sub["retained_w3"])
            rows.append(
                {
                    "family": fam,
                    "model": model,
                    "n": len(sub),
                    "spearman_rho": round(rho, 4) if rho == rho else "",
                    "ci_low": round(lo, 4) if lo == lo else "",
                    "ci_high": round(hi, 4) if hi == hi else "",
                    "p_value": round(float(p), 4) if p == p else "",
                    "subset": "canonical_correct_only",
                    "y": "w3_retained",
                    "contamination_column": "contamination_score",
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
