#!/usr/bin/env python3
"""M2 / L1 extension: matched ALGO canonical vs W6 structural difficulty + accuracy."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402
from probes.contamination.algo_instance_metrics import extract_algo_metrics  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
BANK = REPO_ROOT / "data/problems/question_bank_algo.csv"
INST_OUT = DER / "K3_algo_canonical_w6_instances.csv"
SUM_OUT = DER / "K3_algo_canonical_w6_summary.csv"
MATCH_OUT = DER / "K3_algo_canonical_w6_matched_report.csv"

MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
ALGO_FILES = [
    "ALGO_P1_behavioral_claude_rescored.csv",
    "ALGO_P1_behavioral_gpt4o_rescored.csv",
    "ALGO_P1_behavioral_gemini_rescored.csv",
    "ALGO_P1_behavioral_llama_rescored.csv",
    "ALGO_P1_behavioral_o1mini_rescored.csv",
]

METRIC_SPECS = [
    ("CC", "n_denominations", "w6_lower"),
    ("CC", "target", "w6_lower"),
    ("CC", "optimal_coin_count", "w6_lower"),
    ("SP", "n_nodes", "w6_lower"),
    ("SP", "n_edges", "w6_lower"),
    ("SP", "optimal_path_length", "w6_lower"),
    ("WIS", "n_intervals", "w6_lower"),
    ("WIS", "optimal_weight", "w6_lower"),
]


def _load_scores() -> pd.DataFrame:
    parts = []
    for name in ALGO_FILES:
        path = DER / name
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        df = df[df["included"].str.strip().str.lower().eq("true")].copy()
        df["variant"] = df["variant_type"].map(normalize_variant)
        df = filter_excluded(df, family="ALGO")
        df["ok"] = df["rescored_correct"].str.strip().str.lower().eq("true")
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["model_short"] = out["model"].map(MODELS).fillna(out["model"])
    return out.drop_duplicates(["problem_id", "variant", "model_short"], keep="last")


def _metric_keys(metrics: dict) -> dict[str, float | int | None]:
    subtype = str(metrics.get("subtype", "")).upper()
    if subtype == "CC":
        return {k: metrics.get(k) for k in ("n_denominations", "target", "optimal_coin_count")}
    if subtype == "SP":
        return {k: metrics.get(k) for k in ("n_nodes", "n_edges", "optimal_path_length")}
    if subtype == "WIS":
        return {k: metrics.get(k) for k in ("n_intervals", "optimal_weight")}
    return {}


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    bank = pd.read_csv(BANK, dtype=str).fillna("")
    bank["variant"] = bank["variant_type"].map(normalize_variant)
    can = bank[bank["variant"] == "canonical"].set_index("problem_id")
    w6 = bank[bank["variant"] == "W6"].set_index("problem_id")
    common = sorted(set(can.index) & set(w6.index))

    scores = _load_scores()
    score_idx = (
        scores.set_index(["problem_id", "variant", "model_short"])["ok"]
        if not scores.empty
        else pd.Series(dtype=bool)
    )

    inst_rows: list[dict] = []
    for pid in common:
        c, w = can.loc[pid], w6.loc[pid]
        cm = extract_algo_metrics(
            c["difficulty_params"],
            problem_subtype=c.get("problem_subtype", ""),
            verifier_function=c.get("verifier_function", ""),
        )
        wm = extract_algo_metrics(
            w["difficulty_params"],
            problem_subtype=w.get("problem_subtype", ""),
            verifier_function=w.get("verifier_function", ""),
        )
        row: dict = {
            "problem_id": pid,
            "problem_subtype": c.get("problem_subtype", ""),
            "byte_identical_text": c["problem_text"] == w["problem_text"],
            "identical_answer": c["correct_answer"] == w["correct_answer"],
        }
        for prefix, metrics in (("canonical", cm), ("w6", wm)):
            for key, val in _metric_keys(metrics).items():
                row[f"{prefix}_{key}"] = val
        for model in MODELS.values():
            row[f"{model}_canonical_correct"] = bool(score_idx.get((pid, "canonical", model), False))
            row[f"{model}_w6_correct"] = bool(score_idx.get((pid, "W6", model), False))
        inst_rows.append(row)

    inst = pd.DataFrame(inst_rows)
    # WIS W6 re-seeds difficulty_params while keeping template text; include when answer/structure differs.
    valid = inst[(~inst["byte_identical_text"]) | (~inst["identical_answer"])].copy()
    inst.to_csv(INST_OUT, index=False)
    print(f"Wrote {INST_OUT} ({len(inst)} rows, {len(valid)} valid W6 pairs)")

    report_rows: list[dict] = []
    all_metrics = sorted(
        {
            col.replace("canonical_", "")
            for col in valid.columns
            if col.startswith("canonical_") and col not in {"canonical_ok"}
        }
    )
    for metric in all_metrics:
        ccol, wcol = f"canonical_{metric}", f"w6_{metric}"
        if ccol not in valid.columns:
            continue
        sub = valid[[ccol, wcol]].dropna()
        if sub.empty:
            continue
        c_mean = float(sub[ccol].astype(float).mean())
        w_mean = float(sub[wcol].astype(float).mean())
        delta = w_mean - c_mean
        report_rows.append(
            {
                "section": "structural_matched",
                "model": "--",
                "metric": metric,
                "n_pairs": len(sub),
                "canonical_mean": round(c_mean, 3),
                "w6_mean": round(w_mean, 3),
                "delta_w6_minus_canonical": round(delta, 3),
                "w6_easier_if": "w6_lower",
                "direction": "w6_easier" if delta < -1e-9 else ("w6_harder" if delta > 1e-9 else "matched"),
            }
        )

    summary_rows: list[dict] = []
    for model in MODELS.values():
        ccol = f"{model}_canonical_correct"
        wcol = f"{model}_w6_correct"
        sub = valid[[ccol, wcol]].copy()
        n = len(sub)
        if n == 0:
            continue
        acc_c = float(sub[ccol].mean())
        acc_w = float(sub[wcol].mean())
        row = {
            "section": "accuracy_matched",
            "model": model,
            "metric": "accuracy",
            "n_pairs": n,
            "canonical_mean": round(acc_c, 3),
            "w6_mean": round(acc_w, 3),
            "delta_w6_minus_canonical": round(acc_w - acc_c, 3),
            "w6_easier_if": "w6_higher",
            "direction": "w6_easier" if acc_w > acc_c else ("w6_harder" if acc_w < acc_c else "matched"),
        }
        summary_rows.append(row)
        report_rows.append(row)

    summary = pd.DataFrame(summary_rows + [r for r in report_rows if r["section"] == "structural_matched"])
    summary.to_csv(SUM_OUT, index=False)
    pd.DataFrame(report_rows).to_csv(MATCH_OUT, index=False)
    print(f"Wrote {SUM_OUT}")
    print(f"Wrote {MATCH_OUT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
