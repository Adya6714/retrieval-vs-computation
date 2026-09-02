#!/usr/bin/env python3
"""L1: Matched BW canonical vs W6 difficulty controls + accuracy (all paper models)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402
from probes.contamination.bw_instance_metrics import extract_bw_metrics  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
BANK = REPO_ROOT / "data/problems/question_bank_bw.csv"
INST_OUT = DER / "K3_bw_canonical_w6_instances.csv"
SUM_OUT = DER / "K3_bw_canonical_w6_summary.csv"
MATCH_OUT = DER / "K3_bw_canonical_w6_matched_report.csv"

MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
    "deepseek/deepseek-r1-distill-llama-70b": "DeepSeek",
}
BW_FILES = [
    "BW_P1_behavioral_rescored.csv",
    "BW_P1_behavioral_gemini_rescored.csv",
    "BW_P1_behavioral_o1mini_rescored.csv",
]
METRICS = [
    "num_blocks",
    "n_goal_clauses",
    "goal_tower_depth",
    "init_tower_depth",
    "fd_optimal_plan_length",
    "gold_plan_length",
]


def _plan_length(answer: str) -> int:
    return len([ln for ln in str(answer or "").splitlines() if ln.strip()])


def _load_scores() -> pd.DataFrame:
    bank_ids = set(
        pd.read_csv(BANK, dtype=str)
        .loc[lambda d: d["variant_type"].str.strip().str.lower() == "canonical", "problem_id"]
        .astype(str)
    )
    parts = []
    for name in BW_FILES:
        path = DER / name
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        df = df[df["included"].str.strip().str.lower().eq("true")].copy()
        df = df[df["problem_id"].isin(bank_ids)].copy()
        df["variant"] = df["variant_type"].map(normalize_variant)
        df = filter_excluded(df, family="BW")
        df["ok"] = df["rescored_correct"].str.strip().str.lower().eq("true")
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["model_short"] = out["model"].map(MODELS).fillna(out["model"])
    return out.drop_duplicates(["problem_id", "variant", "model"], keep="last")


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
        if str(c.get("problem_subtype", "")).lower() != "blocksworld":
            continue
        cm = extract_bw_metrics(c["problem_text"], f"{pid}_canonical")
        wm = extract_bw_metrics(w["problem_text"], f"{pid}_W6")
        row: dict = {
            "problem_id": pid,
            "problem_subtype": c.get("problem_subtype", ""),
            "byte_identical_text": c["problem_text"] == w["problem_text"],
            "identical_answer": c["correct_answer"] == w["correct_answer"],
            "canonical_gold_plan_length": _plan_length(c["correct_answer"]),
            "w6_gold_plan_length": _plan_length(w["correct_answer"]),
        }
        for prefix, metrics in (("canonical", cm), ("w6", wm)):
            for key in [
                "num_blocks",
                "n_goal_clauses",
                "goal_tower_depth",
                "init_tower_depth",
                "fd_optimal_plan_length",
                "fd_status",
            ]:
                row[f"{prefix}_{key}"] = metrics.get(key)
        row["gold_plan_length_canonical"] = row["canonical_gold_plan_length"]
        row["gold_plan_length_w6"] = row["w6_gold_plan_length"]
        for model in MODELS.values():
            row[f"{model}_canonical_correct"] = bool(score_idx.get((pid, "canonical", model), False))
            row[f"{model}_w6_correct"] = bool(score_idx.get((pid, "W6", model), False))
        inst_rows.append(row)

    inst = pd.DataFrame(inst_rows)
    valid = inst[~inst["byte_identical_text"]].copy()
    inst.to_csv(INST_OUT, index=False)
    print(f"Wrote {INST_OUT} ({len(inst)} rows, {len(valid)} valid W6 pairs)")

    report_rows: list[dict] = []
    for metric in METRICS:
        ccol = f"canonical_{metric}" if metric != "gold_plan_length" else "canonical_gold_plan_length"
        wcol = f"w6_{metric}" if metric != "gold_plan_length" else "w6_gold_plan_length"
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
                "w6_easier_if": "w6_lower" if metric.endswith("length") or "depth" in metric or metric == "num_blocks" else "w6_lower",
                "direction": "w6_easier" if delta < -1e-9 else ("w6_harder" if delta > 1e-9 else "matched"),
            }
        )

    summary_rows: list[dict] = []
    for model in MODELS.values():
        ccol = f"{model}_canonical_correct"
        wcol = f"{model}_w6_correct"
        sub = valid[[ccol, wcol]].copy()
        sub = sub[sub[ccol].notna() & sub[wcol].notna()]
        n = len(sub)
        if n == 0:
            continue
        acc_c = float(sub[ccol].mean())
        acc_w = float(sub[wcol].mean())
        summary_rows.append(
            {
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
        )
        report_rows.append(summary_rows[-1])

    summary = pd.DataFrame(summary_rows + [r for r in report_rows if r["section"] == "structural_matched"])
    summary.to_csv(SUM_OUT, index=False)
    pd.DataFrame(report_rows).to_csv(MATCH_OUT, index=False)
    print(f"Wrote {SUM_OUT}")
    print(f"Wrote {MATCH_OUT}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
