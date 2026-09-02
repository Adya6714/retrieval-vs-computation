#!/usr/bin/env python3
"""N2: Offline ALGO Probe 2 CCI from phase1 declared steps + phase2 execution.

Ports GSM step-alignment CCI to ALGO adversarial instances:
  declared plan  = phase1 predicted first decision + optimal tail from bank
  executed plan  = phase2 normal parsed decisions (compliant steps)
Emits per-instance CCI, per-step intermediates, and optional TEP from injection.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.algo.decision_normalize import (  # noqa: E402
    normalize_phase1_decision,
    normalize_phase2_decision,
)
from probes.behavioral.cci import compute_cci  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
RAW = REPO_ROOT / "results" / "raw"
BANK = REPO_ROOT / "data/problems/question_bank_algo.csv"
OUT = DER / "ALGO_P2_cci.csv"
STEPS_OUT = DER / "ALGO_P2_cci_steps.csv"

PHASE1_FILES = [
    RAW / "ALGO_P2_phase1_claude_new.csv",
    RAW / "ALGO_P2_phase1_gpt4o_new.csv",
    RAW / "ALGO_P2_phase1_llama_new.csv",
    RAW / "ALGO_P2_phase1_gemini.csv",
]
PHASE2_NORMAL = RAW / "ALGO_P2_phase2_normal.csv"
PHASE2_INJECTED = RAW / "ALGO_P2_phase2_injected.csv"


def _to_bool(s: Any) -> bool:
    return str(s).strip().lower() in {"true", "1", "yes"}


def _cc_optimal_sequence(correct_answer: str) -> list[str]:
    m = re.search(r"\[([^\]]*)\]", str(correct_answer))
    if not m:
        raise ValueError(f"CC correct_answer missing coin list: {correct_answer!r}")
    nums = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    return [str(n) for n in nums]


def _sp_optimal_sequence(correct_answer: str) -> list[str]:
    m = re.search(r"path\s*:\s*(.+?)\s*,\s*cost\s*:", str(correct_answer), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"SP correct_answer missing path: {correct_answer!r}")
    nodes = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
    if len(nodes) < 2:
        raise ValueError(f"SP path too short: {correct_answer!r}")
    return [str(n) for n in nodes[1:]]


def _wis_optimal_set(correct_answer: str) -> set[int]:
    m = re.search(r"\{([^}]*)\}", str(correct_answer))
    if not m:
        raise ValueError(f"WIS correct_answer missing selected set: {correct_answer!r}")
    return {int(x) for x in re.findall(r"-?\d+", m.group(1))}


def _wis_optimal_step(correct_answer: str, step_index: int, parsed_decision: str) -> str:
    selected = _wis_optimal_set(correct_answer)
    norm = normalize_phase2_decision("wis", parsed_decision)
    m = re.search(r"\b(SELECT|RULE OUT)\s+(-?\d+)\b", norm, flags=re.IGNORECASE)
    if not m:
        return ""
    action = m.group(1).upper()
    idx = int(m.group(2))
    if action == "SELECT":
        optimal = "SELECT" if idx in selected else "RULE OUT"
    else:
        optimal = "RULE OUT" if idx not in selected else "SELECT"
    return f"{optimal} {idx}"


def _wis_declared_plan(correct_answer: str, n_intervals: int) -> list[str]:
    """Greedy scan: at each interval index, emit optimal SELECT/RULE OUT."""
    selected = _wis_optimal_set(correct_answer)
    plan: list[str] = []
    for i in range(n_intervals):
        action = "SELECT" if i in selected else "RULE OUT"
        plan.append(f"{action} {i}")
    return plan


def _optimal_plan(subtype: str, correct_answer: str, n_intervals: int = 0) -> list[str]:
    st = subtype.strip().lower()
    if st == "coin_change":
        return _cc_optimal_sequence(correct_answer)
    if st == "shortest_path":
        return _sp_optimal_sequence(correct_answer)
    if st == "wis":
        return _wis_declared_plan(correct_answer, n_intervals)
    raise ValueError(f"Unknown subtype: {subtype}")


def _declared_plan(
    subtype: str,
    correct_answer: str,
    predicted_first: str,
    *,
    n_intervals: int = 0,
    phase1_parseable: bool,
) -> list[str]:
    optimal = _optimal_plan(subtype, correct_answer, n_intervals=n_intervals)
    if not optimal:
        return optimal
    if phase1_parseable:
        first = normalize_phase1_decision(subtype, predicted_first)
        if first:
            return [first] + optimal[1:]
    return optimal


def _executed_plan(subtype: str, steps: pd.DataFrame) -> list[str]:
    steps = steps.sort_values("step_index_int")
    steps = steps[steps["response_type"].astype(str).str.strip().str.lower() == "compliant"]
    return [
        normalize_phase2_decision(subtype, str(r["parsed_decision"]))
        for _, r in steps.iterrows()
        if str(r["parsed_decision"]).strip()
    ]


def _load_phase1() -> pd.DataFrame:
    parts = [pd.read_csv(p, dtype=str).fillna("") for p in PHASE1_FILES if p.exists()]
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out[out["model"].astype(str).str.lower() != "mock"]
    return out.drop_duplicates(["problem_id", "model"], keep="last")


def _normalize_step_base(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["step_index_int"] = pd.to_numeric(out["step_index"], errors="coerce").astype("Int64")
    mins = out.groupby(["problem_id", "model"])["step_index_int"].transform("min")
    out["step_index_int"] = out["step_index_int"] - mins
    return out


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    phase1 = _load_phase1()
    p2n = _normalize_step_base(pd.read_csv(PHASE2_NORMAL, dtype=str).fillna(""))
    p2i = _normalize_step_base(pd.read_csv(PHASE2_INJECTED, dtype=str).fillna(""))
    bank = pd.read_csv(BANK, dtype=str).fillna("")
    bank = bank[bank["variant_type"].str.strip().str.lower() == "canonical"].copy()
    bank = bank.drop_duplicates(["problem_id"])

    def parse_params(s: str) -> dict[str, Any]:
        return json.loads(s) if str(s).strip() else {}

    bank["params"] = bank["difficulty_params"].map(parse_params)
    bank["critical_step_index"] = bank["params"].map(
        lambda p: int(p.get("critical_step_index", -1)) if str(p.get("critical_step_index", "")).strip() else -1,
    )
    bank["n_intervals"] = bank["params"].map(
        lambda p: len(p.get("intervals") or []) if isinstance(p, dict) else 0,
    )
    bank_small = bank[
        ["problem_id", "problem_subtype", "correct_answer", "critical_step_index", "n_intervals"]
    ].copy()

    phase1 = phase1.merge(bank_small, on="problem_id", how="inner")
    phase1 = phase1[phase1["instance_type"].str.strip().str.lower() == "adversarial"].copy()
    phase1["subtype"] = phase1["problem_subtype"].str.strip().str.lower()

    rows: list[dict] = []
    step_rows: list[dict] = []

    for _, p1 in phase1.iterrows():
        pid = str(p1["problem_id"])
        model = str(p1["model"])
        subtype = str(p1["subtype"])
        n_int = int(p1["n_intervals"] or 0)
        parseable = _to_bool(p1.get("phase1_parseable", ""))

        declared = _declared_plan(
            subtype,
            str(p1["correct_answer"]),
            str(p1.get("predicted_first_decision", "")),
            n_intervals=n_int,
            phase1_parseable=parseable,
        )
        exec_steps_df = p2n[(p2n["problem_id"] == pid) & (p2n["model"] == model)].copy()
        executed = _executed_plan(subtype, exec_steps_df)

        cci = compute_cci(pid, declared, executed)
        cci_score = cci["cci"]
        matched = int(cci["matched_steps"])
        total = int(cci["total_steps_compared"])

        # TEP: post-critical compliant divergence between normal and injected
        inj_df = p2i[(p2i["problem_id"] == pid) & (p2i["model"] == model)].copy()
        crit = int(p1["critical_step_index"])
        tep_score = ""
        if not exec_steps_df.empty and not inj_df.empty and crit >= 0:
            merged = exec_steps_df.merge(
                inj_df[["step_index_int", "response_type", "parsed_decision"]],
                on="step_index_int",
                how="inner",
                suffixes=("_n", "_i"),
            )
            post = merged[
                (merged["step_index_int"] > crit)
                & (merged["response_type_n"].str.lower() == "compliant")
                & (merged["response_type_i"].str.lower() == "compliant")
            ]
            if not post.empty:
                diff = (
                    post.apply(
                        lambda r: normalize_phase2_decision(subtype, r["parsed_decision_n"])
                        != normalize_phase2_decision(subtype, r["parsed_decision_i"]),
                        axis=1,
                    )
                    .astype(float)
                    .tolist()
                )
                tep_score = round(sum(diff) / len(diff), 4)

        rows.append(
            {
                "problem_id": pid,
                "model": model,
                "subtype": subtype,
                "instance_type": "adversarial",
                "phase1_predicted_first_decision": p1.get("predicted_first_decision", ""),
                "phase1_declared_first_normalized": normalize_phase1_decision(
                    subtype, str(p1.get("predicted_first_decision", "")),
                ),
                "phase1_parseable": parseable,
                "declared_plan_json": json.dumps(declared),
                "executed_plan_json": json.dumps(executed),
                "cci_score": cci_score if cci_score is not None else "",
                "cci_matched": matched,
                "cci_total": total,
                "tep_score": tep_score,
                "critical_step_index": crit,
                "correct_answer": p1["correct_answer"],
            }
        )

        n_steps = max(len(declared), len(executed))
        for i in range(n_steps):
            d_step = declared[i] if i < len(declared) else ""
            e_step = executed[i] if i < len(executed) else ""
            step_rows.append(
                {
                    "problem_id": pid,
                    "model": model,
                    "subtype": subtype,
                    "step_index": i,
                    "declared_step": d_step,
                    "executed_step": e_step,
                    "step_match": d_step == e_step if d_step and e_step else False,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    pd.DataFrame(step_rows).to_csv(STEPS_OUT, index=False)
    print(f"Wrote {OUT} ({len(out)} rows)")
    print(f"Wrote {STEPS_OUT} ({len(step_rows)} rows)")


if __name__ == "__main__":
    main()
