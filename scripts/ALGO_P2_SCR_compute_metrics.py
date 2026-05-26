#!/usr/bin/env python3
"""Compute ALGO Probe-2 metrics from Phase1/Phase2 outputs (offline only).

Removed metrics (paper refactor — not named paper outputs):
  ADC (derived columns; raw stated_algorithm and greedy_assessment_correct stay in
  phase1 CSVs), CPP, SC, FDI, RDI, RTDA summary statistic columns.

Primary CSV outputs: CCI, TEP (same definitions as Probe 2 run scripts).
Phase2 CSVs retain raw reasoning_type for RTDA prose; no RTDA aggregate columns here.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.stats import bootstrap_ci


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _to_bool(x: Any) -> bool | None:
    s = str(x).strip().lower()
    if s == "true":
        return True
    if s == "false":
        return False
    return None


def _warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def _normalize_step_base(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    mins = out.groupby(["problem_id", "model"])["step_index_int"].transform("min")
    out["step_index_int"] = out["step_index_int"] - mins
    return out


def _phase1_intent(stated_algorithm: str) -> str:
    s = str(stated_algorithm).lower()
    if any(k in s for k in ["dynamic programming", "dp", "dijkstra", "optimal", "memoization", "subproblem"]):
        return "dp_like"
    if any(k in s for k in ["greedy", "largest", "nearest", "closest", "highest weight"]):
        return "greedy_like"
    return "unknown"


def _first_step_decision(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    d = df.sort_values("step_index_int").iloc[0]["parsed_decision"]
    return str(d).strip()


def _normalize_decision(subtype: str, text: str) -> str:
    s = str(text).strip()
    if subtype == "wis":
        m = re.search(r"\b(SELECT|RULE OUT)\s+(-?\d+)\b", s, flags=re.IGNORECASE)
        return f"{m.group(1).upper()} {int(m.group(2))}" if m else s.upper()
    m = re.search(r"-?\d+", s)
    return str(int(m.group(0))) if m else s


def _cc_optimal_sequence(correct_answer: str) -> list[str]:
    m = re.search(r"\[([^\]]*)\]", str(correct_answer))
    if not m:
        raise ValueError(f"CC correct_answer missing coin list: {correct_answer!r}")
    nums = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    if not nums:
        raise ValueError(f"CC correct_answer has empty coin list: {correct_answer!r}")
    return [str(n) for n in nums]


def _sp_optimal_sequence(correct_answer: str) -> list[str]:
    m = re.search(r"path\s*:\s*(.+?)\s*,\s*cost\s*:", str(correct_answer), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"SP correct_answer missing path: {correct_answer!r}")
    nodes = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    if len(nodes) < 2:
        raise ValueError(f"SP correct_answer path too short: {correct_answer!r}")
    return [str(n) for n in nodes[1:]]


def _wis_optimal_set(correct_answer: str) -> set[int]:
    m = re.search(r"\{([^}]*)\}", str(correct_answer))
    if not m:
        raise ValueError(f"WIS correct_answer missing selected set: {correct_answer!r}")
    return {int(x) for x in re.findall(r"-?\d+", m.group(1))}


def _optimal_for_step(subtype: str, correct_answer: str, step_index_int: int, parsed_decision: str) -> bool:
    if subtype == "coin_change":
        seq = _cc_optimal_sequence(correct_answer)
        if step_index_int < 0 or step_index_int >= len(seq):
            return False
        return _normalize_decision(subtype, parsed_decision) == seq[step_index_int]
    if subtype == "shortest_path":
        seq = _sp_optimal_sequence(correct_answer)
        if step_index_int < 0 or step_index_int >= len(seq):
            return False
        return _normalize_decision(subtype, parsed_decision) == seq[step_index_int]
    if subtype == "wis":
        norm = _normalize_decision(subtype, parsed_decision)
        m = re.search(r"\b(SELECT|RULE OUT)\s+(-?\d+)\b", norm, flags=re.IGNORECASE)
        if not m:
            return False
        action = m.group(1).upper()
        idx = int(m.group(2))
        selected = _wis_optimal_set(correct_answer)
        if action == "SELECT":
            return idx in selected
        return idx not in selected
    raise ValueError(f"Unknown subtype: {subtype}")


def _metric_rows(
    *,
    model: str,
    subtype: str,
    metric_name: str,
    values: list[float],
    bootstrap_n: int,
) -> dict[str, Any]:
    if not values:
        return {
            "model": model,
            "subtype": subtype,
            "metric_name": metric_name,
            "metric_value": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        }
    lo, hi = bootstrap_ci(values, n_resamples=bootstrap_n)
    return {
        "model": model,
        "subtype": subtype,
        "metric_name": metric_name,
        "metric_value": float(np.mean(values)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute ALGO Probe2 metrics from phase outputs.")
    parser.add_argument("--phase1", nargs="+", required=True, help="Phase1 CSVs (one or more).")
    parser.add_argument("--phase2-normal", required=True)
    parser.add_argument("--phase2-injected", required=True)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--per-instance-output",
        default="results/derived/ALGO_P2_per_instance_cci.csv",
        help="Per (problem_id, model) CCI components for triangulation.",
    )
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    args = parser.parse_args()

    np.random.seed(42)

    p1_frames = [pd.read_csv(Path(p), dtype=str).fillna("") for p in args.phase1]
    phase1 = pd.concat(p1_frames, ignore_index=True)
    phase2_normal = pd.read_csv(Path(args.phase2_normal), dtype=str).fillna("")
    phase2_injected = pd.read_csv(Path(args.phase2_injected), dtype=str).fillna("")
    phase1 = phase1[phase1["model"].astype(str).str.lower() != "mock"].copy()
    phase2_normal = phase2_normal[phase2_normal["model"].astype(str).str.lower() != "mock"].copy()
    phase2_injected = phase2_injected[phase2_injected["model"].astype(str).str.lower() != "mock"].copy()
    bank = pd.read_csv(Path(args.bank), dtype=str).fillna("")

    _require_columns(
        phase1,
        {
            "problem_id",
            "model",
            "subtype",
            "instance_type",
            "stated_algorithm",
            "predicted_first_decision",
            "phase1_parseable",
        },
        "phase1",
    )
    _require_columns(
        phase2_normal,
        {
            "problem_id",
            "model",
            "subtype",
            "instance_type",
            "step_index",
            "response_type",
            "parsed_decision",
            "reasoning_type",
        },
        "phase2_normal",
    )
    _require_columns(
        phase2_injected,
        {
            "problem_id",
            "model",
            "subtype",
            "instance_type",
            "step_index",
            "critical_step_index",
            "response_type",
            "parsed_decision",
            "reasoning_type",
        },
        "phase2_injected",
    )
    _require_columns(
        bank,
        {"problem_id", "variant_type", "problem_subtype", "correct_answer", "difficulty_params"},
        "bank",
    )

    bank = bank[bank["variant_type"].str.strip().str.lower() == "canonical"].copy()
    if bank["problem_id"].duplicated().any():
        dups = sorted(bank.loc[bank["problem_id"].duplicated(), "problem_id"].unique().tolist())
        raise ValueError(f"Canonical bank has duplicate problem_id rows: {dups}")

    def parse_params(s: str) -> dict[str, Any]:
        try:
            return json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid difficulty_params JSON: {e}")

    bank["params"] = bank["difficulty_params"].map(parse_params)
    bank["instance_type_bank"] = bank["params"].map(lambda p: str(p.get("instance_type", "")).strip().lower())
    bank["critical_step_index_bank"] = bank["params"].map(
        lambda p: int(p.get("critical_step_index", -1)) if str(p.get("critical_step_index", "")).strip() != "" else -1
    )
    bank_small = bank[
        ["problem_id", "problem_subtype", "correct_answer", "instance_type_bank", "critical_step_index_bank"]
    ].copy()

    if phase1.duplicated(subset=["problem_id", "model"]).any():
        dup = phase1.loc[phase1.duplicated(subset=["problem_id", "model"], keep=False), ["problem_id", "model"]]
        raise ValueError(f"Phase1 has duplicate (problem_id,model) rows:\n{dup.to_string(index=False)}")

    phase1 = phase1.merge(bank_small, on="problem_id", how="inner", validate="many_to_one")
    if phase1.empty:
        raise ValueError("No overlap between phase1 and bank by problem_id.")

    mismatch = phase1[phase1["subtype"].str.strip().str.lower() != phase1["problem_subtype"].str.strip().str.lower()]
    if not mismatch.empty:
        _warn(f"{len(mismatch)} Phase1 rows subtype mismatch with bank; using bank subtype.")
    phase1["subtype"] = phase1["problem_subtype"].str.strip().str.lower()

    phase1["phase1_parseable_bool"] = phase1["phase1_parseable"].map(_to_bool)
    for _, r in phase1.iterrows():
        if r["phase1_parseable_bool"] is not True:
            _warn(f"phase1_parseable missing/false for {r['problem_id']} {r['model']}")

    for df_name, df in [("phase2_normal", phase2_normal), ("phase2_injected", phase2_injected)]:
        df["subtype"] = df["subtype"].str.strip().str.lower()
        df["instance_type"] = df["instance_type"].str.strip().str.lower()
        df["step_index_int"] = pd.to_numeric(df["step_index"], errors="coerce")
        if df["step_index_int"].isna().any():
            bad = df[df["step_index_int"].isna()][["problem_id", "model", "step_index"]].head(10)
            raise ValueError(f"{df_name} has non-numeric step_index rows (sample):\n{bad.to_string(index=False)}")
        df["step_index_int"] = df["step_index_int"].astype(int)

    phase2_normal = _normalize_step_base(phase2_normal)
    phase2_injected = _normalize_step_base(phase2_injected)

    p1_pairs = set(zip(phase1["problem_id"], phase1["model"]))
    n_pairs = set(zip(phase2_normal["problem_id"], phase2_normal["model"]))
    i_pairs = set(zip(phase2_injected["problem_id"], phase2_injected["model"]))
    if not n_pairs.issubset(p1_pairs):
        miss = sorted(list(n_pairs - p1_pairs))[:10]
        raise ValueError(f"phase2_normal has pairs missing in phase1 (sample): {miss}")
    if not i_pairs.issubset(p1_pairs):
        miss = sorted(list(i_pairs - p1_pairs))[:10]
        raise ValueError(f"phase2_injected has pairs missing in phase1 (sample): {miss}")

    out_rows: list[dict[str, Any]] = []
    per_instance_rows: list[dict[str, Any]] = []
    models = sorted(phase1["model"].unique().tolist())
    subtypes = sorted(phase1["subtype"].unique().tolist())

    for model in models:
        p1m = phase1[phase1["model"] == model].copy()

        for subtype in subtypes:
            p1s = p1m[p1m["subtype"] == subtype].copy()
            if p1s.empty:
                continue

            adv = p1s[p1s["instance_type_bank"] == "adversarial"]
            p2n = phase2_normal[(phase2_normal["model"] == model) & (phase2_normal["subtype"] == subtype)].copy()
            p2i = phase2_injected[(phase2_injected["model"] == model) & (phase2_injected["subtype"] == subtype)].copy()
            p2n = p2n.merge(
                p1s[["problem_id", "correct_answer", "critical_step_index_bank", "predicted_first_decision", "stated_algorithm"]],
                on="problem_id",
                how="inner",
                validate="many_to_one",
            )
            p2i = p2i.merge(
                p1s[["problem_id", "correct_answer", "critical_step_index_bank", "predicted_first_decision", "stated_algorithm"]],
                on="problem_id",
                how="inner",
                validate="many_to_one",
            )

            cci_vals: list[float] = []
            tep_vals: list[float] = []

            for pid in sorted(set(adv["problem_id"].tolist())):
                a_n = p2n[(p2n["problem_id"] == pid)].sort_values("step_index_int")
                a_i = p2i[(p2i["problem_id"] == pid)].sort_values("step_index_int")
                if a_n.empty:
                    raise ValueError(f"Missing phase2_normal steps for adversarial pair: {pid} {model}")
                p1row = adv[adv["problem_id"] == pid].iloc[0]
                expected_critical = int(p1row["critical_step_index_bank"])
                if expected_critical < 0:
                    _warn(f"adversarial row has invalid critical_step_index={expected_critical}: {pid} {model}")

                intent = _phase1_intent(str(p1row["stated_algorithm"]))
                rtypes = [str(x).strip().lower() for x in a_n["reasoning_type"].tolist() if str(x).strip()]
                if not rtypes:
                    _warn(f"missing reasoning_type for {pid} {model}")
                    cci_alg = 0.0
                else:
                    forward_algo = sum(t in {"forward_simulation", "algorithm_invocation"} for t in rtypes) / len(rtypes)
                    local = sum(t == "local_greedy" for t in rtypes) / len(rtypes)
                    if intent == "dp_like":
                        cci_alg = 1.0 if forward_algo >= local else 0.0
                    elif intent == "greedy_like":
                        cci_alg = 1.0 if local >= forward_algo else 0.0
                    else:
                        cci_alg = 0.0

                first_norm = _first_step_decision(a_n)
                pred = str(p1row["predicted_first_decision"])
                match_first = (
                    1.0
                    if _normalize_decision(subtype, first_norm) == _normalize_decision(subtype, pred)
                    else 0.0
                )

                crit_row = a_n[a_n["step_index_int"] == expected_critical]
                if crit_row.empty:
                    _warn(f"critical step {expected_critical} missing in phase2_normal for {pid} {model}")
                    cci_crit = 0.0
                else:
                    d = str(crit_row.iloc[0]["parsed_decision"])
                    cci_crit = (
                        1.0
                        if _optimal_for_step(subtype, str(p1row["correct_answer"]), expected_critical, d)
                        else 0.0
                    )
                cci_composite = float(np.mean([cci_alg, match_first, cci_crit]))
                cci_vals.append(cci_composite)
                per_instance_rows.append(
                    {
                        "problem_id": pid,
                        "model": model,
                        "cci_alg": cci_alg,
                        "cci_crit": cci_crit,
                        "match_first": match_first,
                        "cci_composite": cci_composite,
                    }
                )

                merged_steps = a_n.merge(
                    a_i[["step_index_int", "parsed_decision", "response_type"]],
                    on="step_index_int",
                    how="inner",
                    suffixes=("_n", "_i"),
                )
                post = merged_steps[merged_steps["step_index_int"] > expected_critical]
                post = post[(post["response_type_n"] == "compliant") & (post["response_type_i"] == "compliant")]
                if not post.empty:
                    diff = (
                        post.apply(
                            lambda rr: _normalize_decision(subtype, rr["parsed_decision_n"])
                            != _normalize_decision(subtype, rr["parsed_decision_i"]),
                            axis=1,
                        )
                        .astype(float)
                        .tolist()
                    )
                    tep_vals.append(float(np.mean(diff)))

            out_rows.append(_metric_rows(model=model, subtype=subtype, metric_name="CCI", values=cci_vals, bootstrap_n=args.bootstrap_n))
            out_rows.append(_metric_rows(model=model, subtype=subtype, metric_name="TEP", values=tep_vals, bootstrap_n=args.bootstrap_n))

    out = pd.DataFrame(out_rows, columns=["model", "subtype", "metric_name", "metric_value", "ci_lower", "ci_upper"])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote metrics: {out_path} ({len(out)} rows)")

    per_path = Path(args.per_instance_output)
    per_path.parent.mkdir(parents=True, exist_ok=True)
    per_cols = ["problem_id", "model", "cci_alg", "cci_crit", "match_first", "cci_composite"]
    per_out = pd.DataFrame(per_instance_rows, columns=per_cols)
    if per_out.duplicated(subset=["problem_id", "model"]).any():
        raise ValueError("Per-instance CCI has duplicate (problem_id, model) rows.")
    per_out.to_csv(per_path, index=False)
    print(f"Wrote per-instance CCI: {per_path} ({len(per_out)} rows)")


if __name__ == "__main__":
    main()
