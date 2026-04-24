#!/usr/bin/env python3
"""Compute ALGO Probe-2 metrics from Phase1/Phase2 outputs (offline only)."""

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


def _parse_int(text: Any) -> int | None:
    m = re.search(r"-?\d+", str(text))
    return int(m.group(0)) if m else None


def _normalize_step_base(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize step_index_int to 0-based per (problem_id, model) run."""
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
        # Rule-outs are treated as suboptimal if they remove optimal items.
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
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    args = parser.parse_args()

    np.random.seed(42)  # deterministic bootstrap draws

    p1_frames = [pd.read_csv(Path(p), dtype=str).fillna("") for p in args.phase1]
    phase1 = pd.concat(p1_frames, ignore_index=True)
    phase2_normal = pd.read_csv(Path(args.phase2_normal), dtype=str).fillna("")
    phase2_injected = pd.read_csv(Path(args.phase2_injected), dtype=str).fillna("")
    bank = pd.read_csv(Path(args.bank), dtype=str).fillna("")

    _require_columns(
        phase1,
        {
            "problem_id",
            "model",
            "subtype",
            "instance_type",
            "stated_algorithm",
            "greedy_assessment_correct",
            "predicted_first_decision",
            "critical_point_identified",
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
            "final_answer_correct",
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
            "post_injection_correct",
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
    bank["greedy_succeeds"] = bank["params"].map(lambda p: p.get("greedy_succeeds", None))
    bank["critical_step_index_bank"] = bank["params"].map(lambda p: int(p.get("critical_step_index", -1)) if str(p.get("critical_step_index", "")).strip() != "" else -1)
    bank_small = bank[
        ["problem_id", "problem_subtype", "correct_answer", "instance_type_bank", "greedy_succeeds", "critical_step_index_bank"]
    ].copy()

    if phase1.duplicated(subset=["problem_id", "model"]).any():
        dup = phase1.loc[phase1.duplicated(subset=["problem_id", "model"], keep=False), ["problem_id", "model"]]
        raise ValueError(f"Phase1 has duplicate (problem_id,model) rows:\n{dup.to_string(index=False)}")

    phase1 = phase1.merge(bank_small, on="problem_id", how="inner", validate="many_to_one")
    if phase1.empty:
        raise ValueError("No overlap between phase1 and bank by problem_id.")

    # Validate subtype consistency (warn for mismatch, prefer bank subtype).
    mismatch = phase1[phase1["subtype"].str.strip().str.lower() != phase1["problem_subtype"].str.strip().str.lower()]
    if not mismatch.empty:
        _warn(f"{len(mismatch)} Phase1 rows subtype mismatch with bank; using bank subtype.")
    phase1["subtype"] = phase1["problem_subtype"].str.strip().str.lower()

    # Parse bools and step indices.
    phase1["greedy_assessment_correct_bool"] = phase1["greedy_assessment_correct"].map(_to_bool)
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

    # Basic join integrity check on (problem_id, model).
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
    models = sorted(phase1["model"].unique().tolist())
    subtypes = sorted(phase1["subtype"].unique().tolist())

    for model in models:
        p1m = phase1[phase1["model"] == model].copy()

        for subtype in subtypes:
            p1s = p1m[p1m["subtype"] == subtype].copy()
            if p1s.empty:
                continue

            # ADC standard/adversarial from structured field only.
            std = p1s[p1s["instance_type_bank"] == "standard"]
            adv = p1s[p1s["instance_type_bank"] == "adversarial"]
            adc_std_vals = [
                1.0 if v is True else 0.0
                for v in std["greedy_assessment_correct_bool"].tolist()
                if v is not None
            ]
            adc_adv_vals = [
                1.0 if v is True else 0.0
                for v in adv["greedy_assessment_correct_bool"].tolist()
                if v is not None
            ]
            out_rows.append(_metric_rows(model=model, subtype=subtype, metric_name="ADC_standard", values=adc_std_vals, bootstrap_n=args.bootstrap_n))
            out_rows.append(_metric_rows(model=model, subtype=subtype, metric_name="ADC_adversarial", values=adc_adv_vals, bootstrap_n=args.bootstrap_n))

            # CPP adversarial only.
            cpp_vals: list[float] = []
            for _, r in adv.iterrows():
                expected = int(r["critical_step_index_bank"])
                got = _parse_int(r["critical_point_identified"])
                if expected == -1:
                    _warn(f"missing critical_step_index (-1) for adversarial {r['problem_id']}")
                cpp_match = False
                if got is not None:
                    # Accept both 0-based and 1-based mentions from free-form Q4 text.
                    cpp_match = (got == expected) or (got - 1 == expected)
                cpp_vals.append(1.0 if cpp_match else 0.0)
            out_rows.append(_metric_rows(model=model, subtype=subtype, metric_name="CPP", values=cpp_vals, bootstrap_n=args.bootstrap_n))

            # Gather Phase2 rows for this model/subtype.
            p2n = phase2_normal[(phase2_normal["model"] == model) & (phase2_normal["subtype"] == subtype)].copy()
            p2i = phase2_injected[(phase2_injected["model"] == model) & (phase2_injected["subtype"] == subtype)].copy()
            p2n = p2n.merge(
                p1s[["problem_id", "instance_type_bank", "correct_answer", "critical_step_index_bank", "predicted_first_decision", "stated_algorithm"]],
                on="problem_id",
                how="inner",
                validate="many_to_one",
            )
            p2i = p2i.merge(
                p1s[["problem_id", "instance_type_bank", "correct_answer", "critical_step_index_bank", "predicted_first_decision", "stated_algorithm"]],
                on="problem_id",
                how="inner",
                validate="many_to_one",
            )

            # CCI components per (problem_id, model), adversarial only.
            cci_alg_vals: list[float] = []
            cci_first_vals: list[float] = []
            cci_critical_vals: list[float] = []
            cci_comp_vals: list[float] = []
            fdi0_vals: list[float] = []
            fdi_later_vals: list[float] = []
            sc_vals: list[float] = []
            tep_vals: list[float] = []

            adv_pids = sorted(set(adv["problem_id"].tolist()))
            for pid in adv_pids:
                a_n = p2n[(p2n["problem_id"] == pid)].sort_values("step_index_int")
                a_i = p2i[(p2i["problem_id"] == pid)].sort_values("step_index_int")
                if a_n.empty:
                    raise ValueError(f"Missing phase2_normal steps for adversarial pair: {pid} {model}")
                p1row = adv[adv["problem_id"] == pid]
                if p1row.empty:
                    raise ValueError(f"Missing phase1 row for adversarial pair: {pid} {model}")
                p1row = p1row.iloc[0]
                expected_critical = int(p1row["critical_step_index_bank"])
                if expected_critical < 0:
                    _warn(f"adversarial row has invalid critical_step_index={expected_critical}: {pid} {model}")

                # CCI_algorithm
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
                cci_alg_vals.append(cci_alg)

                # CCI_first_decision and FDI step0
                first_norm = _first_step_decision(a_n)
                pred = str(p1row["predicted_first_decision"])
                match_first = 1.0 if _normalize_decision(subtype, first_norm) == _normalize_decision(subtype, pred) else 0.0
                cci_first_vals.append(match_first)
                fdi0_vals.append(match_first)

                # CCI_critical from normal at critical step.
                crit_row = a_n[a_n["step_index_int"] == expected_critical]
                if crit_row.empty:
                    _warn(f"critical step {expected_critical} missing in phase2_normal for {pid} {model}")
                    cci_crit = 0.0
                else:
                    d = str(crit_row.iloc[0]["parsed_decision"])
                    cci_crit = 1.0 if _optimal_for_step(subtype, str(p1row["correct_answer"]), expected_critical, d) else 0.0
                cci_critical_vals.append(cci_crit)
                cci_comp_vals.append(float(np.mean([cci_alg, match_first, cci_crit])))

                # SC: if step0 incorrect, cascade on later steps.
                if len(a_n) >= 3:
                    first_idx = int(a_n.iloc[0]["step_index_int"])
                    first_ok = _optimal_for_step(subtype, str(p1row["correct_answer"]), first_idx, str(a_n.iloc[0]["parsed_decision"]))
                    if not first_ok:
                        later = a_n.iloc[1:]
                        later_eval = []
                        for _, rr in later.iterrows():
                            later_eval.append(
                                0.0 if _optimal_for_step(subtype, str(p1row["correct_answer"]), int(rr["step_index_int"]), str(rr["parsed_decision"])) else 1.0
                            )
                        if later_eval:
                            sc_vals.append(float(np.mean(later_eval)))

                # FDI later: compare injected vs normal post-first steps on compliant pairs.
                merged_steps = a_n.merge(
                    a_i[["step_index_int", "parsed_decision", "response_type"]],
                    on="step_index_int",
                    how="inner",
                    suffixes=("_n", "_i"),
                )
                later = merged_steps[merged_steps["step_index_int"] > merged_steps["step_index_int"].min()]
                later = later[
                    (later["response_type_n"] == "compliant")
                    & (later["response_type_i"] == "compliant")
                ]
                if not later.empty:
                    same = (
                        later.apply(
                            lambda rr: _normalize_decision(subtype, rr["parsed_decision_n"]) == _normalize_decision(subtype, rr["parsed_decision_i"]),
                            axis=1,
                        )
                        .astype(float)
                        .tolist()
                    )
                    fdi_later_vals.append(float(np.mean(same)))

                # TEP refined on compliant rows after critical step, adversarial only.
                post = merged_steps[merged_steps["step_index_int"] > expected_critical]
                post = post[
                    (post["response_type_n"] == "compliant")
                    & (post["response_type_i"] == "compliant")
                ]
                if not post.empty:
                    diff = (
                        post.apply(
                            lambda rr: _normalize_decision(subtype, rr["parsed_decision_n"]) != _normalize_decision(subtype, rr["parsed_decision_i"]),
                            axis=1,
                        )
                        .astype(float)
                        .tolist()
                    )
                    tep_vals.append(float(np.mean(diff)))

            out_rows.append(_metric_rows(model=model, subtype=subtype, metric_name="CCI_algorithm", values=cci_alg_vals, bootstrap_n=args.bootstrap_n))
            out_rows.append(_metric_rows(model=model, subtype=subtype, metric_name="CCI_first_decision", values=cci_first_vals, bootstrap_n=args.bootstrap_n))
            out_rows.append(_metric_rows(model=model, subtype=subtype, metric_name="CCI_critical", values=cci_critical_vals, bootstrap_n=args.bootstrap_n))
            out_rows.append(_metric_rows(model=model, subtype=subtype, metric_name="CCI_composite", values=cci_comp_vals, bootstrap_n=args.bootstrap_n))
            out_rows.append(_metric_rows(model=model, subtype=subtype, metric_name="TEP_refined", values=tep_vals, bootstrap_n=args.bootstrap_n))
            out_rows.append(_metric_rows(model=model, subtype=subtype, metric_name="FDI_step0", values=fdi0_vals, bootstrap_n=args.bootstrap_n))
            out_rows.append(_metric_rows(model=model, subtype=subtype, metric_name="FDI_later_stability", values=fdi_later_vals, bootstrap_n=args.bootstrap_n))
            out_rows.append(_metric_rows(model=model, subtype=subtype, metric_name="SC", values=sc_vals, bootstrap_n=args.bootstrap_n))

            # RDI on normal condition: standard vs adversarial.
            p2n_sub = p2n.copy()
            def _rdi_values(df: pd.DataFrame) -> list[float]:
                vals = []
                for pid, g in df.groupby("problem_id"):
                    rt = [str(x).strip().lower() for x in g["reasoning_type"].tolist() if str(x).strip()]
                    if not rt:
                        _warn(f"missing reasoning_type for RDI: {pid} {model}")
                        continue
                    num = sum(t in {"forward_simulation", "algorithm_invocation"} for t in rt)
                    den = sum(t in {"forward_simulation", "algorithm_invocation", "local_greedy", "unclear"} for t in rt)
                    if den == 0:
                        continue
                    vals.append(num / den)
                return vals
            rdi_std = _rdi_values(p2n_sub[p2n_sub["instance_type_bank"] == "standard"])
            rdi_adv = _rdi_values(p2n_sub[p2n_sub["instance_type_bank"] == "adversarial"])
            out_rows.append(_metric_rows(model=model, subtype=subtype, metric_name="RDI_standard", values=rdi_std, bootstrap_n=args.bootstrap_n))
            out_rows.append(_metric_rows(model=model, subtype=subtype, metric_name="RDI_adversarial", values=rdi_adv, bootstrap_n=args.bootstrap_n))

        # RTDA across all subtypes for this model on normal run.
        p2m = phase2_normal[phase2_normal["model"] == model].copy()
        p2m = p2m.merge(
            phase1[["problem_id", "model", "instance_type_bank"]],
            on=["problem_id", "model"],
            how="inner",
            validate="many_to_one",
        )
        tracked = ["local_greedy", "forward_simulation", "algorithm_invocation", "unclear"]
        for inst in ["standard", "adversarial"]:
            dfi = p2m[p2m["instance_type_bank"] == inst].copy()
            if dfi.empty:
                for rtype in tracked:
                    out_rows.append(
                        {
                            "model": model,
                            "subtype": "all",
                            "metric_name": f"RTDA_{inst}_{rtype}",
                            "metric_value": np.nan,
                            "ci_lower": np.nan,
                            "ci_upper": np.nan,
                        }
                    )
                continue
            rt = dfi["reasoning_type"].astype(str).str.strip().str.lower().tolist()
            for rtype in tracked:
                vals = [1.0 if x == rtype else 0.0 for x in rt]
                out_rows.append(
                    _metric_rows(
                        model=model,
                        subtype="all",
                        metric_name=f"RTDA_{inst}_{rtype}",
                        values=vals,
                        bootstrap_n=args.bootstrap_n,
                    )
                )

    out = pd.DataFrame(out_rows, columns=["model", "subtype", "metric_name", "metric_value", "ci_lower", "ci_upper"])
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote metrics: {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
