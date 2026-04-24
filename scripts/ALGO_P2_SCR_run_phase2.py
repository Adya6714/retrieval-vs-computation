#!/usr/bin/env python3
"""Run Probe 2 Phase-2 stepwise execution (normal and injected conditions)."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.behavioral.openai_client import OpenRouterClient
from probes.contamination.verify_algo import verify_algo

load_dotenv()


NORMAL_COLUMNS = [
    "problem_id",
    "model",
    "subtype",
    "instance_type",
    "step_index",
    "current_state_prompt",
    "raw_response",
    "response_type",
    "parsed_decision",
    "parse_status",
    "reasoning_text",
    "reasoning_type",
    "state_after_step",
    "final_answer_correct",
]

INJECTED_COLUMNS = [
    "problem_id",
    "model",
    "subtype",
    "instance_type",
    "step_index",
    "injection_applied",
    "critical_step_index",
    "true_state",
    "injected_state",
    "raw_response",
    "response_type",
    "parsed_decision",
    "parse_status",
    "reasoning_text",
    "reasoning_type",
    "diverged_from_normal",
    "post_injection_correct",
]


def classify_reasoning_type(reason_text: str, subtype: str) -> str:
    reason_lower = str(reason_text).lower()
    forward_keywords = [
        "leaves",
        "remaining will be",
        "then i can",
        "which means",
        "because then",
        "allows me to",
        "future",
        "next step",
    ]
    greedy_keywords = [
        "largest",
        "biggest",
        "smallest",
        "nearest",
        "closest",
        "highest weight",
        "maximum",
        "greedy",
        "best available",
    ]
    algorithm_keywords = [
        "dynamic programming",
        "dp",
        "dijkstra",
        "optimal substructure",
        "table",
        "memoization",
        "subproblem",
    ]
    backtrack_keywords = ["actually", "wait", "instead", "better to", "reconsider"]
    if any(k in reason_lower for k in backtrack_keywords):
        return "backtracking"
    if any(k in reason_lower for k in algorithm_keywords):
        return "algorithm_invocation"
    if any(k in reason_lower for k in forward_keywords):
        return "forward_simulation"
    if any(k in reason_lower for k in greedy_keywords):
        return "local_greedy"
    return "unclear"


def parse_decision_reason(raw: str, subtype: str) -> tuple[str, str, str, str]:
    """Return (response_type, decision, parse_status, reasoning_text)."""
    text = str(raw or "").strip()
    lower = text.lower()

    has_decision = re.search(r"^\s*decision\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)
    has_reason = re.search(r"^\s*reason\s*:\s*(.+)$", text, flags=re.IGNORECASE | re.MULTILINE)

    response_type = "format_ignored"
    if re.search(r"\b(i cannot|i can't|unable|won't|refuse)\b", lower):
        response_type = "refusal"
    elif has_decision and has_reason:
        # If contains long additional blocks, classify as partial.
        nonempty_lines = [ln for ln in text.splitlines() if ln.strip()]
        if len(nonempty_lines) <= 3:
            response_type = "compliant"
        else:
            response_type = "partial_compliance"
    elif has_decision:
        response_type = "partial_compliance"

    # full_solution_dump detection by subtype.
    if subtype == "coin_change" and re.search(r"(?:coins?|scoops?)\s*:\s*\[[^\]]+\]", lower):
        response_type = "full_solution_dump"
    if subtype == "shortest_path" and re.search(r"path\s*:\s*.+(?:->|→).+cost", lower):
        response_type = "full_solution_dump"
    if subtype == "wis" and re.search(r"selected\s*:\s*\{[^}]+\}\s*,\s*total\s*:", lower):
        response_type = "full_solution_dump"

    decision = ""
    reason = ""
    if has_decision:
        decision = has_decision.group(1).strip()
    if has_reason:
        reason = has_reason.group(1).strip()

    if not decision and response_type in {"format_ignored", "partial_compliance"}:
        # best-effort fallback extraction, keeps parse from failing on mild format drift.
        first_line = text.splitlines()[0].strip() if text else ""
        if first_line:
            decision = first_line
    if not reason and response_type in {"format_ignored", "partial_compliance"}:
        if len(text.splitlines()) > 1:
            reason = text.splitlines()[1].strip()

    parse_status = "parsed_clean" if (decision and reason and response_type == "compliant") else "parsed_with_normalization"
    if not decision:
        parse_status = "parse_failed"
    return response_type, decision, parse_status, reason


def _cc_prompt(denoms: list[int], target: int, coins: list[int], remaining: int) -> str:
    return (
        "Coin change problem.\n"
        f"Denominations: {denoms}\n"
        f"Target: {target}\n"
        f"Coins chosen so far: {coins}\n"
        f"Remaining: {remaining}\n\n"
        "Make your next single decision.\n"
        "Decision: [write only the denomination value]\n"
        "Reason: [write one sentence explaining why you chose this]"
    )


def _sp_prompt(edges: list[tuple[int, int, int]], source: int, target: int, cur: int, path: list[int], cost: int) -> str:
    neighbors = [(v, w) for u, v, w in edges if u == cur]
    return (
        "Shortest path problem.\n"
        f"Graph edges with weights: {edges}\n"
        f"Source: {source}, Target: {target}\n"
        f"Current position: {cur}\n"
        f"Path so far: {path}, cost so far: {cost}\n"
        f"Unvisited neighbors of current node: {neighbors}\n\n"
        "Make your next single decision.\n"
        "Decision: [write only the next node to move to]\n"
        "Reason: [write one sentence explaining why you chose this node]"
    )


def _wis_prompt(intervals: list[dict], selected: list[int], ruled_out: list[int]) -> str:
    all_ids = [int(x["id"]) for x in intervals]
    available = [i for i in all_ids if i not in selected and i not in ruled_out]
    cur_w = sum(int(intervals[i]["weight"]) for i in selected if 0 <= i < len(intervals))
    interval_list = [(int(x["id"]), int(x["start"]), int(x["end"]), int(x["weight"])) for x in intervals]
    return (
        "Weighted interval scheduling problem.\n"
        f"Intervals: {interval_list}\n"
        f"Selected so far: {selected}, total weight: {cur_w}\n"
        f"Ruled out (overlap): {ruled_out}\n"
        f"Available (not selected, not ruled out): {available}\n\n"
        "Make your next single decision.\n"
        "Decision: [write \"SELECT X\" or \"RULE OUT X\" where X is the interval index]\n"
        "Reason: [write one sentence explaining why]"
    )


def _parse_cc_decision(decision: str) -> int | None:
    m = re.search(r"-?\d+", str(decision))
    return int(m.group(0)) if m else None


def _parse_sp_decision(decision: str) -> int | None:
    m = re.search(r"-?\d+", str(decision))
    return int(m.group(0)) if m else None


def _parse_wis_decision(decision: str) -> tuple[str, int] | None:
    m = re.search(r"\b(SELECT|RULE OUT)\s+(-?\d+)\b", str(decision), flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).upper(), int(m.group(2))


def _canonical_rows(bank: pd.DataFrame, instance_type_filter: str | None) -> pd.DataFrame:
    df = bank[bank["variant_type"] == "canonical"].copy()
    df = df[df["problem_id"].str.match(r"^(CC|SP|WIS)_")]
    if instance_type_filter:
        def match_type(p: str) -> bool:
            params = json.loads(p)
            return str(params.get("instance_type", "")).strip().lower() == instance_type_filter
        df = df[df["difficulty_params"].map(match_type)]
    return df


def _done_pairs(output_path: Path, condition: str) -> set[tuple[str, str]]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(output_path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return set()
    needed = {"problem_id", "model"}
    if not needed.issubset(df.columns):
        return set()
    # Consider a pair done if it has at least one row already.
    return {(r["problem_id"], r["model"]) for _, r in df.iterrows()}


def run_one(
    *,
    row: pd.Series,
    model: str,
    client: OpenRouterClient,
    condition: str,
) -> list[dict]:
    pid = str(row["problem_id"])
    subtype = str(row["problem_subtype"]).strip().lower()
    params = json.loads(str(row["difficulty_params"]))
    instance_type = str(params.get("instance_type", "")).strip().lower()
    if condition == "injected" and instance_type != "adversarial":
        return []

    critical_step = int(params.get("critical_step_index", -1))
    gt = str(row["correct_answer"])
    rows: list[dict] = []

    if subtype == "coin_change":
        denoms = [int(x) for x in params["denominations"]]
        target = int(params["target"])
        gt_coins = [int(x) for x in re.findall(r"\d+", re.search(r"\[([^\]]*)\]", gt).group(1))]
        state = {"coins": [], "remaining": target}
        injected = {"coins": [], "remaining": target}
        max_steps = max(1, len(gt_coins))
        for step in range(max_steps):
            use_state = injected if (condition == "injected" and critical_step == step) else state
            true_state = dict(state)
            prompt = _cc_prompt(denoms, target, use_state["coins"], use_state["remaining"])
            raw = client.complete(pid, prompt).get("response", "")
            rtype, dec_raw, pstatus, reason = parse_decision_reason(raw, subtype)
            decision = _parse_cc_decision(dec_raw) if pstatus != "parse_failed" else None
            if decision is not None and decision in denoms and decision <= state["remaining"]:
                state["coins"].append(decision)
                state["remaining"] -= decision
            if condition == "injected" and critical_step == step:
                # simple injected perturbation: report a shifted remaining to the model.
                injected["coins"] = list(state["coins"])
                injected["remaining"] = max(0, state["remaining"] + 1)
            final_ok = False
            if step == max_steps - 1:
                ans = f"Count: {len(state['coins'])}\nCoins: [{', '.join(str(c) for c in state['coins'])}]"
                final_ok, _reason, _meta = verify_algo(
                    pid, ans, gt, subtype, "canonical", params
                )
            rec = {
                "problem_id": pid,
                "model": model,
                "subtype": subtype,
                "instance_type": instance_type,
                "step_index": step,
                "raw_response": raw,
                "response_type": rtype,
                "parsed_decision": "" if decision is None else str(decision),
                "parse_status": pstatus,
                "reasoning_text": reason,
                "reasoning_type": classify_reasoning_type(reason, subtype),
            }
            if condition == "normal":
                rec.update(
                    {
                        "current_state_prompt": prompt,
                        "state_after_step": json.dumps(state),
                        "final_answer_correct": str(bool(final_ok)),
                    }
                )
            else:
                rec.update(
                    {
                        "injection_applied": str(step == critical_step and critical_step != -1),
                        "critical_step_index": critical_step,
                        "true_state": json.dumps(true_state),
                        "injected_state": json.dumps(injected if step == critical_step and critical_step != -1 else true_state),
                        "diverged_from_normal": str(step >= critical_step and critical_step != -1),
                        "post_injection_correct": str(bool(final_ok)) if step == max_steps - 1 else "",
                    }
                )
            rows.append(rec)

    elif subtype == "shortest_path":
        edges = [(int(e["u"]), int(e["v"]), int(e["w"])) for e in params["graph"]]
        source = int(params["source"])
        target = int(params["target"])
        gt_path = [int(x) for x in re.findall(r"\d+", re.search(r"Path:\s*(.+?)\s*,\s*Cost", gt).group(1))]
        state = {"cur": source, "path": [source], "cost": 0}
        injected_state = dict(state)
        max_steps = max(1, len(gt_path) - 1)
        for step in range(max_steps):
            use = injected_state if (condition == "injected" and critical_step == step) else state
            true_state = dict(state)
            prompt = _sp_prompt(edges, source, target, use["cur"], use["path"], use["cost"])
            raw = client.complete(pid, prompt).get("response", "")
            rtype, dec_raw, pstatus, reason = parse_decision_reason(raw, subtype)
            decision = _parse_sp_decision(dec_raw) if pstatus != "parse_failed" else None
            if decision is not None:
                # apply if edge exists
                w = next((w for u, v, w in edges if u == state["cur"] and v == decision), None)
                if w is not None:
                    state["cur"] = decision
                    state["path"].append(decision)
                    state["cost"] += w
            if condition == "injected" and critical_step == step:
                # injected perturbation: pretend cost is off by +1.
                injected_state = {"cur": state["cur"], "path": list(state["path"]), "cost": state["cost"] + 1}
            final_ok = False
            if step == max_steps - 1:
                ans = f"Path: {' -> '.join(str(x) for x in state['path'])}, Cost: {state['cost']}"
                final_ok, _reason, _meta = verify_algo(
                    pid, ans, gt, subtype, "canonical", params
                )
            rec = {
                "problem_id": pid,
                "model": model,
                "subtype": subtype,
                "instance_type": instance_type,
                "step_index": step,
                "raw_response": raw,
                "response_type": rtype,
                "parsed_decision": "" if decision is None else str(decision),
                "parse_status": pstatus,
                "reasoning_text": reason,
                "reasoning_type": classify_reasoning_type(reason, subtype),
            }
            if condition == "normal":
                rec.update(
                    {
                        "current_state_prompt": prompt,
                        "state_after_step": json.dumps(state),
                        "final_answer_correct": str(bool(final_ok)),
                    }
                )
            else:
                rec.update(
                    {
                        "injection_applied": str(step == critical_step and critical_step != -1),
                        "critical_step_index": critical_step,
                        "true_state": json.dumps(true_state),
                        "injected_state": json.dumps(injected_state if step == critical_step and critical_step != -1 else true_state),
                        "diverged_from_normal": str(step >= critical_step and critical_step != -1),
                        "post_injection_correct": str(bool(final_ok)) if step == max_steps - 1 else "",
                    }
                )
            rows.append(rec)

    elif subtype == "wis":
        intervals = params["intervals"]
        if intervals and isinstance(intervals[0], dict):
            iv = [{"id": int(x["id"]), "start": int(x["start"]), "end": int(x["end"]), "weight": int(x["weight"])} for x in intervals]
        else:
            iv = [{"id": i, "start": int(x[0]), "end": int(x[1]), "weight": int(x[2])} for i, x in enumerate(intervals)]
        gt_selected = [int(x) for x in re.findall(r"\d+", re.search(r"\{([^}]*)\}", gt).group(1))]
        selected: list[int] = []
        ruled_out: set[int] = set()
        injected_selected: list[int] = []
        injected_ruled_out: set[int] = set()
        max_steps = max(1, len(gt_selected))
        for step in range(max_steps):
            use_sel = injected_selected if (condition == "injected" and critical_step == step) else selected
            use_ruled = injected_ruled_out if (condition == "injected" and critical_step == step) else ruled_out
            true_state = {"selected": list(selected), "ruled_out": sorted(ruled_out)}
            prompt = _wis_prompt(iv, use_sel, sorted(use_ruled))
            raw = client.complete(pid, prompt).get("response", "")
            rtype, dec_raw, pstatus, reason = parse_decision_reason(raw, subtype)
            parsed = _parse_wis_decision(dec_raw) if pstatus != "parse_failed" else None
            parsed_decision = ""
            if parsed is not None:
                action, idx = parsed
                parsed_decision = f"{action} {idx}"
                if action == "SELECT" and idx not in ruled_out and idx not in selected:
                    selected.append(idx)
                    s = next((x for x in iv if x["id"] == idx), None)
                    if s is not None:
                        for y in iv:
                            if y["id"] == idx:
                                continue
                            overlap = not (s["end"] <= y["start"] or y["end"] <= s["start"])
                            if overlap and y["id"] not in selected:
                                ruled_out.add(y["id"])
                elif action == "RULE OUT":
                    ruled_out.add(idx)
            if condition == "injected" and critical_step == step:
                injected_selected = list(selected)
                injected_ruled_out = set(ruled_out)
                # perturb by ruling out one extra available item if any
                avail = [x["id"] for x in iv if x["id"] not in injected_selected and x["id"] not in injected_ruled_out]
                if avail:
                    injected_ruled_out.add(avail[0])
            final_ok = False
            if step == max_steps - 1:
                total = sum(next(x["weight"] for x in iv if x["id"] == i) for i in selected if any(x["id"] == i for x in iv))
                ans = f"Selected: {{{', '.join(str(i) for i in sorted(selected))}}}, Total: {total}"
                final_ok, _reason, _meta = verify_algo(
                    pid, ans, gt, subtype, "canonical", params
                )
            rec = {
                "problem_id": pid,
                "model": model,
                "subtype": subtype,
                "instance_type": instance_type,
                "step_index": step,
                "raw_response": raw,
                "response_type": rtype,
                "parsed_decision": parsed_decision,
                "parse_status": pstatus,
                "reasoning_text": reason,
                "reasoning_type": classify_reasoning_type(reason, subtype),
            }
            if condition == "normal":
                rec.update(
                    {
                        "current_state_prompt": prompt,
                        "state_after_step": json.dumps({"selected": sorted(selected), "ruled_out": sorted(ruled_out)}),
                        "final_answer_correct": str(bool(final_ok)),
                    }
                )
            else:
                rec.update(
                    {
                        "injection_applied": str(step == critical_step and critical_step != -1),
                        "critical_step_index": critical_step,
                        "true_state": json.dumps(true_state),
                        "injected_state": json.dumps(
                            {"selected": sorted(injected_selected), "ruled_out": sorted(injected_ruled_out)}
                            if (step == critical_step and critical_step != -1)
                            else true_state
                        ),
                        "diverged_from_normal": str(step >= critical_step and critical_step != -1),
                        "post_injection_correct": str(bool(final_ok)) if step == max_steps - 1 else "",
                    }
                )
            rows.append(rec)

    else:
        raise ValueError(f"{pid}: unsupported subtype {subtype!r}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ALGO Probe2 phase2 step-by-step execution.")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--condition", required=True, choices=["normal", "injected"])
    parser.add_argument("--instance-type", default=None, choices=["adversarial", "standard"])
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    bank = pd.read_csv(Path(args.bank), dtype=str).fillna("")
    rows = _canonical_rows(bank, args.instance_type)
    if rows.empty:
        raise ValueError("No canonical rows matched filters.")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = NORMAL_COLUMNS if args.condition == "normal" else INJECTED_COLUMNS
    done = _done_pairs(out_path, args.condition) if args.resume else set()

    write_header = not out_path.exists() or out_path.stat().st_size == 0
    with out_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if write_header:
            writer.writeheader()

        for model in args.models:
            client = OpenRouterClient(model=model)
            for _, row in rows.iterrows():
                pid = str(row["problem_id"])
                if args.resume and (pid, model) in done:
                    continue
                out_rows = run_one(row=row, model=model, client=client, condition=args.condition)
                for r in out_rows:
                    writer.writerow(r)
                f.flush()
                done.add((pid, model))

    print(f"Wrote phase2 output: {out_path}")


if __name__ == "__main__":
    main()
