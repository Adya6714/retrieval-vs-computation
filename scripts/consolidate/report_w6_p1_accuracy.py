#!/usr/bin/env python3
"""K1: Report pooled W6 accuracy (ALGO cluster-bootstrap CI; BW iid)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.clones import cluster_ids_for  # noqa: E402
from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.stats import cluster_bootstrap_ci  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
OUT = DER / "P1_w6_accuracy.csv"
BUG = DER / "K1_wis_w6_generator_bug.csv"

PAPER_MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "google/gemini-2.5-flash": "Gemini",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
DEEPSEEK = "deepseek/deepseek-r1-distill-llama-70b"
ALGO_TAGS = ["claude", "gpt4o", "gemini", "llama", "o1mini"]
BW_FILES = [
    "BW_P1_behavioral_rescored.csv",
    "BW_P1_behavioral_gemini_rescored.csv",
    "BW_P1_behavioral_o1mini_rescored.csv",
]


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _load_algo() -> pd.DataFrame:
    parts = []
    for tag in ALGO_TAGS:
        path = DER / f"ALGO_P1_behavioral_{tag}_rescored.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        df = df[_is_true(df["included"])].copy()
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["variant_type"] = out["variant_type"].map(normalize_variant)
    out = filter_excluded(out, family="ALGO")
    ok = out["rescored_correct"] if "rescored_correct" in out.columns else out.get("verified", "")
    out["ok"] = _is_true(ok).astype(float)
    out["model_short"] = out["model"].map(PAPER_MODELS).fillna(out["model"])
    return out.drop_duplicates(["problem_id", "variant_type", "model"], keep="last")


def _load_bw() -> pd.DataFrame:
    bank = pd.read_csv(REPO_ROOT / "data/problems/question_bank_bw.csv", dtype=str)
    ids = set(
        bank.loc[bank["variant_type"].str.strip().str.lower() == "canonical", "problem_id"].astype(str)
    )
    parts = []
    for name in BW_FILES:
        path = DER / name
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        df = df[_is_true(df["included"])].copy()
        df = df[df["problem_id"].isin(ids)]
        df["variant_type"] = df["variant_type"].map(normalize_variant)
        df = filter_excluded(df, family="BW")
        ok = df["rescored_correct"] if "rescored_correct" in df.columns else df.get("verified", "")
        df["ok"] = _is_true(ok).astype(float)
        df["model_short"] = df["model"].map(PAPER_MODELS).fillna(df["model"])
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates(
        ["problem_id", "variant_type", "model"], keep="last"
    )


def _w6_row(family: str, model: str, sub: pd.DataFrame) -> dict:
    vals = sub["ok"].astype(float).tolist()
    pids = sub["problem_id"].astype(str).tolist()
    n = len(vals)
    acc = float(sub["ok"].mean()) if n else float("nan")
    if family == "ALGO":
        lo, hi = cluster_bootstrap_ci(vals, cluster_ids_for(pids), n_resamples=5000, seed=42)
        n_clusters = len(set(cluster_ids_for(pids)))
        ci_method = "cluster_bootstrap"
    else:
        arr = np.array(vals, dtype=float)
        rng = np.random.default_rng(42)
        boots = [float(arr[rng.integers(0, n, size=n)].mean()) for _ in range(5000)]
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        n_clusters = n
        ci_method = "iid_bootstrap"
    return {
        "family": family,
        "model": model,
        "variant": "W6",
        "accuracy": round(acc, 3) if n else "",
        "ci_low": round(lo, 3) if n else "",
        "ci_high": round(hi, 3) if n else "",
        "n": n,
        "n_clusters": n_clusters,
        "ci_method": ci_method,
    }


def _emit_wis_bug_report() -> None:
    bank = pd.read_csv(REPO_ROOT / "data/problems/question_bank_algo.csv", dtype=str).fillna("")
    bank["variant"] = bank["variant_type"].map(normalize_variant)
    can = bank[bank["variant"] == "canonical"].set_index("problem_id")
    w6 = bank[bank["variant"] == "W6"].set_index("problem_id")
    rows = []
    for pid in sorted(set(can.index) & set(w6.index)):
        c, w = can.loc[pid], w6.loc[pid]
        if c["problem_text"] != w["problem_text"]:
            continue
        if c["correct_answer"] == w["correct_answer"]:
            continue
        rows.append(
            {
                "problem_id": pid,
                "problem_subtype": c.get("problem_subtype", ""),
                "byte_identical_text": True,
                "answer_differs": True,
                "canonical_answer": c["correct_answer"],
                "w6_answer": w["correct_answer"],
                "w6_notes": w.get("notes", ""),
                "generator": "scripts/ALGO_PX_SCR_generate_w6.py",
                "bug": "render_wis_text_with_weights lacks Interval N: start=, end=, weight= pattern; "
                "updates difficulty_params/answer but leaves problem_text unchanged",
            }
        )
    pd.DataFrame(rows).to_csv(BUG, index=False)
    print(f"Wrote {BUG} ({len(rows)} rows)")


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    _emit_wis_bug_report()
    rows: list[dict] = []

    algo = _load_algo()
    w6_algo = algo[algo["variant_type"] == "W6"]
    for model in sorted(w6_algo["model_short"].dropna().unique()):
        sub = w6_algo[w6_algo["model_short"] == model]
        rows.append(_w6_row("ALGO", model, sub))

    bw = _load_bw()
    w6_bw = bw[bw["variant_type"] == "W6"]
    for model in sorted(w6_bw["model_short"].dropna().unique()):
        sub = w6_bw[w6_bw["model_short"] == model]
        rows.append(_w6_row("BW", model, sub))
    ds = w6_bw[w6_bw["model"] == DEEPSEEK]
    if not ds.empty:
        rows.append(_w6_row("BW", "DeepSeek", ds))

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
