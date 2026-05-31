#!/usr/bin/env python3
"""Run dedicated Probe 1 Phase-1 planning sessions (no answer solving)."""

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


load_dotenv()

OUTPUT_COLUMNS = [
    "problem_id",
    "model",
    "subtype",
    "instance_type",
    "raw_response",
    "stated_algorithm",
    "greedy_assessment_correct",
    "greedy_assessment_text",
    "predicted_first_decision",
    "critical_point_identified",
    "unparseable_q2",
    "phase1_parseable",
]


def _parse_phase1_fields(raw: str, params: dict) -> tuple[dict, list[str]]:
    answers = _split_answers(raw)
    q1 = answers.get(1, "")
    q2 = answers.get(2, "")
    q3 = answers.get(3, "")
    q4 = answers.get(4, "")

    stated_algorithm = _extract_algorithm(q1)
    yes_no = _parse_yes_no(q2)
    greedy_correct = ""
    unparseable_q2 = "False"
    if "greedy_succeeds" in params:
        if yes_no is None:
            unparseable_q2 = "True"
        else:
            expected_yes = bool(params["greedy_succeeds"])
            greedy_correct = str((yes_no == "yes") == expected_yes)

    critical_step = _extract_critical_step_index(params)
    critical_identified = _critical_match(q4, critical_step)

    failed_steps: list[str] = []
    if not stated_algorithm:
        failed_steps.append("Q1 extraction")
    if yes_no is None:
        failed_steps.append("Q2 extraction")
    if not q3.strip():
        failed_steps.append("Q3 extraction")
    if not q4.strip():
        failed_steps.append("Q4 extraction")

    phase1_parseable = str(len(failed_steps) == 0)
    parsed = {
        "stated_algorithm": stated_algorithm,
        "greedy_assessment_correct": greedy_correct,
        "greedy_assessment_text": q2.strip(),
        "predicted_first_decision": q3.strip(),
        "critical_point_identified": critical_identified,
        "unparseable_q2": unparseable_q2,
        "phase1_parseable": phase1_parseable,
    }
    return parsed, failed_steps


def _prompt_for_subtype(subtype: str, problem_text: str) -> str:
    if subtype == "coin_change":
        return (
            "You will be given a coin change problem. Do NOT compute the final answer yet.\n\n"
            f"Problem: {problem_text}\n\n"
            "Before solving, describe your approach:\n"
            "1. What algorithm will you use and why?\n"
            "2. Will a greedy approach (always picking the largest coin that fits) work here? \n"
            "   Answer yes or no and explain why in one sentence.\n"
            "3. What will your first coin choice be, and what is your reasoning for that choice?\n"
            "4. What is the key decision point in this problem — the step where a wrong choice \n"
            "   would lead to a suboptimal solution?\n\n"
            "Format your response as exactly four numbered paragraphs starting with 1., 2., 3., 4. "
            "Do not use headers, bold text, or markdown formatting.\n"
            "Do not give the final answer. Only describe your approach."
        )
    if subtype == "shortest_path":
        return (
            "You will be given a shortest path problem. Do NOT compute the final answer yet.\n\n"
            f"Problem: {problem_text}\n\n"
            "Before solving, describe your approach:\n"
            "1. What algorithm will you use and why?\n"
            "2. Will a greedy approach (always moving to the nearest unvisited neighbor) work here?\n"
            "   Answer yes or no and explain why in one sentence.\n"
            "3. What will your first move be from the source node, and why?\n"
            "4. Is there any part of the graph where a locally short edge leads to a globally \n"
            "   longer path? If yes, where?\n\n"
            "Format your response as exactly four numbered paragraphs starting with 1., 2., 3., 4. "
            "Do not use headers, bold text, or markdown formatting.\n"
            "Do not give the final answer. Only describe your approach."
        )
    if subtype == "wis":
        return (
            "You will be given a weighted interval scheduling problem. Do NOT compute the final \n"
            "answer yet.\n\n"
            f"Problem: {problem_text}\n\n"
            "Before solving, describe your approach:\n"
            "1. What algorithm will you use and why?\n"
            "2. Will a greedy approach (always picking the highest-weight available interval) \n"
            "   work here? Answer yes or no and explain why in one sentence.\n"
            "3. What will your first interval selection be, and why?\n"
            "4. Which interval, if incorrectly selected early, would most damage the optimal \n"
            "   total weight?\n\n"
            "Format your response as exactly four numbered paragraphs starting with 1., 2., 3., 4. "
            "Do not use headers, bold text, or markdown formatting.\n"
            "Do not give the final answer. Only describe your approach."
        )
    raise ValueError(f"Unsupported subtype for phase1 prompt: {subtype}")


def _split_answers(raw: str) -> dict[int, str]:
    text = str(raw or "").strip()
    # Extract numbered blocks 1..4 conservatively.
    parts = re.split(r"(?=\n?\s*[1-4][\).\:]\s*)", text)
    out: dict[int, str] = {}
    for part in parts:
        m = re.match(r"\s*([1-4])[\).\:]\s*(.*)", part, flags=re.DOTALL)
        if not m:
            continue
        idx = int(m.group(1))
        body = m.group(2).strip()
        body = re.sub(r"\n?\s*[1-4][\).\:]\s.*$", "", body, flags=re.DOTALL).strip()
        out[idx] = body
    # Fallback: if no numbered structure, use coarse sentence chunks.
    if not out:
        lines = [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()]
        if lines:
            out[1] = lines[0]
        if len(lines) > 1:
            out[2] = lines[1]
        if len(lines) > 2:
            out[3] = lines[2]
        if len(lines) > 3:
            out[4] = lines[3]
    return out


def _parse_yes_no(text: str) -> str | None:
    t = str(text).lower()
    yes_patterns = [
        r"\byes\b",
        r"\bgreedy (?:approach )?works?\b",
        r"\bgreedy (?:approach )?will work\b",
        r"\bgreedy succeeds?\b",
        r"\bgreedy is sufficient\b",
        r"\bgreedy gives optimal\b",
    ]
    no_patterns = [
        r"\bno\b",
        r"\bgreedy fails?\b",
        r"\bgreedy does not work\b",
        r"\bgreedy won'?t work\b",
        r"\bgreedy is not sufficient\b",
        r"\bgreedy gives suboptimal\b",
    ]
    for pat in yes_patterns:
        if re.search(pat, t):
            return "yes"
    for pat in no_patterns:
        if re.search(pat, t):
            return "no"
    return None


def _extract_algorithm(q1: str) -> str:
    text = str(q1).strip()
    if not text:
        return ""
    m = re.search(
        r"\b(dynamic programming|dijkstra|bellman-?ford|shortest path|greedy|backtracking|branch and bound)\b",
        text,
        flags=re.IGNORECASE,
    )
    return m.group(1) if m else text[:200]


def _extract_critical_step_index(params: dict) -> int | None:
    for key in ("critical_step", "critical_step_index", "critical_decision_step"):
        if key in params:
            try:
                return int(params[key])
            except Exception:
                return None
    return None


def _critical_match(q4: str, critical_step: int | None) -> str:
    if critical_step is None:
        return ""
    nums = [int(x) for x in re.findall(r"\d+", str(q4))]
    if not nums:
        return "False"
    return str(critical_step in nums)


def _existing_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except pd.errors.EmptyDataError:
        return set()
    req = {"problem_id", "model"}
    if not req.issubset(df.columns):
        return set()
    return {(str(r["problem_id"]).strip(), str(r["model"]).strip()) for _, r in df.iterrows()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Probe 1 Phase-1 planning sessions.")
    parser.add_argument("--bank", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--debug-unparseable", action="store_true")
    parser.add_argument("--rerun-unparseable", action="store_true")
    args = parser.parse_args()

    bank_path = Path(args.bank)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not bank_path.exists():
        raise FileNotFoundError(f"Bank not found: {bank_path}")

    df = pd.read_csv(bank_path, dtype=str).fillna("")
    df = df[df["variant_type"] == "canonical"].copy()
    df = df[df["problem_id"].str.match(r"^(CC|SP|WIS)_")]
    if args.limit is not None:
        df = df.head(args.limit)

    # Build canonical lookup by problem_id for parser replay and metadata refresh.
    bank_lookup = {}
    for _, r in df.iterrows():
        bank_lookup[str(r["problem_id"]).strip()] = {
            "subtype": str(r["problem_subtype"]).strip().lower(),
            "instance_type": str(json.loads(str(r["difficulty_params"])).get("instance_type", "")).strip(),
            "params": json.loads(str(r["difficulty_params"])),
            "problem_text": str(r["problem_text"]),
        }

    if args.debug_unparseable:
        if not output_path.exists():
            raise FileNotFoundError(
                f"--debug-unparseable requested but output does not exist: {output_path}"
            )
        out_df = pd.read_csv(output_path, dtype=str).fillna("")
        out_df = out_df[out_df["model"] == args.model]
        for _, r in out_df.iterrows():
            pid = str(r["problem_id"]).strip()
            if pid not in bank_lookup:
                continue
            parsed, failed_steps = _parse_phase1_fields(
                str(r.get("raw_response", "")), bank_lookup[pid]["params"]
            )
            if parsed["phase1_parseable"] == "False":
                print(f"\n[{pid}] failed steps: {', '.join(failed_steps)}")
                print(str(r.get("raw_response", "")))
        return

    if args.rerun_unparseable:
        if not output_path.exists():
            raise FileNotFoundError(
                f"--rerun-unparseable requested but output does not exist: {output_path}"
            )
        out_df = pd.read_csv(output_path, dtype=str).fillna("")
        mask = (out_df["model"] == args.model) & (out_df["phase1_parseable"] == "False")
        idxs = out_df[mask].index.tolist()
        if not idxs:
            print("No unparseable rows found for rerun.")
            return
        client = OpenRouterClient(model=args.model)
        for idx in idxs:
            pid = str(out_df.at[idx, "problem_id"]).strip()
            if pid not in bank_lookup:
                continue
            b = bank_lookup[pid]
            prompt = _prompt_for_subtype(b["subtype"], b["problem_text"])
            raw = client.complete(pid, prompt).get("response", "")
            parsed, _failed = _parse_phase1_fields(raw, b["params"])
            out_df.at[idx, "raw_response"] = raw
            out_df.at[idx, "subtype"] = b["subtype"]
            out_df.at[idx, "instance_type"] = b["instance_type"]
            for key, value in parsed.items():
                out_df.at[idx, key] = value
        out_df.to_csv(output_path, index=False)
        print(f"Reran unparseable rows and updated: {output_path} ({len(idxs)} rows)")
        return

    client = OpenRouterClient(model=args.model)
    done = _existing_keys(output_path) if args.resume else set()
    write_header = not output_path.exists() or output_path.stat().st_size == 0

    # If resuming and output exists, refresh metadata columns from current bank/parse.
    if args.resume and output_path.exists() and output_path.stat().st_size > 0:
        out_df = pd.read_csv(output_path, dtype=str).fillna("")
        touched = 0
        for idx, r in out_df.iterrows():
            pid = str(r.get("problem_id", "")).strip()
            if pid not in bank_lookup:
                continue
            b = bank_lookup[pid]
            parsed, _failed = _parse_phase1_fields(str(r.get("raw_response", "")), b["params"])
            for key, value in parsed.items():
                if str(out_df.at[idx, key]) != str(value):
                    out_df.at[idx, key] = value
                    touched += 1
            if str(out_df.at[idx, "subtype"]) != b["subtype"]:
                out_df.at[idx, "subtype"] = b["subtype"]
                touched += 1
            if str(out_df.at[idx, "instance_type"]) != b["instance_type"]:
                out_df.at[idx, "instance_type"] = b["instance_type"]
                touched += 1
        if touched > 0:
            out_df.to_csv(output_path, index=False)
            print(f"Refreshed {touched} metadata fields in existing output before resume.")

    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()

        for _, row in df.iterrows():
            pid = str(row["problem_id"]).strip()
            model = str(args.model)
            if args.resume and (pid, model) in done:
                continue

            subtype = str(row["problem_subtype"]).strip().lower()
            params = json.loads(str(row["difficulty_params"]))
            instance_type = str(params.get("instance_type", "")).strip()
            if instance_type not in {"standard", "adversarial"}:
                raise ValueError(f"{pid}: invalid/missing instance_type in difficulty_params")

            prompt = _prompt_for_subtype(subtype, str(row["problem_text"]))
            raw = client.complete(pid, prompt).get("response", "")

            parsed, _failed = _parse_phase1_fields(raw, params)

            writer.writerow(
                {
                    "problem_id": pid,
                    "model": model,
                    "subtype": subtype,
                    "instance_type": instance_type,
                    "raw_response": raw,
                    "stated_algorithm": parsed["stated_algorithm"],
                    "greedy_assessment_correct": parsed["greedy_assessment_correct"],
                    "greedy_assessment_text": parsed["greedy_assessment_text"],
                    "predicted_first_decision": parsed["predicted_first_decision"],
                    "critical_point_identified": parsed["critical_point_identified"],
                    "unparseable_q2": parsed["unparseable_q2"],
                    "phase1_parseable": parsed["phase1_parseable"],
                }
            )
            f.flush()
            done.add((pid, model))

    print(f"Wrote phase1 output: {output_path}")


if __name__ == "__main__":
    main()
