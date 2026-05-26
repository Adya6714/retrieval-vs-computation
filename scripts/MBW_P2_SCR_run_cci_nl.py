"""MBW Phase-2A runner with NL-tolerant action parsing.

Mirror of scripts/BW_P2_SCR_run_cci_nl.py but using the MBW action grammar
(attack/succumb/overcome/feast) and predicate set (harmony/province/planet/
pain/craves).

Output:
    results/raw/MBW_P2_cci_nl.csv

Usage:
    python scripts/MBW_P2_SCR_run_cci_nl.py --dry-run
    python scripts/MBW_P2_SCR_run_cci_nl.py --smoke --problem-ids MBW_010 \\
        --models anthropic/claude-sonnet-4
    python scripts/MBW_P2_SCR_run_cci_nl.py --models anthropic/claude-sonnet-4 \\
        openai/gpt-4o meta-llama/llama-3.1-8b-instruct
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.behavioral.mbw_action_parser_nl import (
    is_preamble_mbw,
    remap_to_canonical_mbw,
)
from probes.behavioral.mbw_pipeline import (
    execute_action_mbw,
    goal_reached_mbw,
    make_followup_prompt_mbw,
    make_turn1_prompt_mbw,
    parse_state_from_text_mbw,
    state_to_narrative_mbw,
)

BANK_CSV = ROOT / "data/problems/question_bank_bw.csv"
OUT_DEFAULT = ROOT / "results/raw/MBW_P2_cci_nl.csv"

MAX_CONSECUTIVE_ERRORS = 2
MAX_SKIPS = 5
LOOP_WINDOW = 8


class DryRunClientMBW:
    """Canned MBW responses for end-to-end dry runs."""

    def __init__(self, model: str):
        self.model = model
        # Solves MBW with attack X / overcome X Y pattern: pick up b, place on f, etc.
        self.scripted = [
            "I will attack b",      # preamble line
            "attack b",             # canonical
            "overcome b on f",      # NL variant
            "attack a",
            "overcome a b",
            "attack e",
            "overcome e a",
            "attack d",
            "overcome d e",
        ]
        self.idx = 0

    def complete(self, prompt: str) -> str:
        if self.idx >= len(self.scripted):
            return "succumb z"  # benign fallback
        r = self.scripted[self.idx]
        self.idx += 1
        return r


def parse_action_nl_mbw(response_text: str) -> tuple[str | None, str]:
    if response_text is None:
        return None, "unparseable"
    lines = [ln.strip() for ln in str(response_text).split("\n") if ln.strip()]
    saw_preamble = False
    import re
    for line in lines:
        stripped = re.sub(r"^\d+[\.\)\:]\s*", "", line)
        stripped = re.sub(r"^step\s+\d+[\.\:\)]?\s*", "", stripped, flags=re.IGNORECASE).strip()
        if not stripped:
            continue
        if is_preamble_mbw(stripped):
            saw_preamble = True
            continue
        remapped = remap_to_canonical_mbw(stripped)
        if remapped is not None:
            label = "canonical" if remapped == stripped.lower() else "remapped"
            return remapped, label
    return None, ("preamble_only" if saw_preamble else "unparseable")


def _goal_narrative_mbw(goal: dict) -> str:
    parts = [f"craves {x} {y}" for x, y in sorted(goal.items())]
    return ", ".join(parts) + " (all true)"


def run_session_mbw(
    problem_id: str,
    problem_text: str,
    client,
    max_steps: int = 50,
    verbose: bool = False,
) -> dict:
    try:
        objects, state, goal = parse_state_from_text_mbw(problem_text)
    except Exception as exc:
        return {
            "problem_id": problem_id,
            "session_status": f"error: parse_state_from_text_mbw: {exc}",
            "goal_reached": False,
            "executed_length": 0,
            "illegal_action_count": 0,
            "skip_count": 0,
            "first_illegal_step": None,
            "partial_goal_achievement": None,
            "goals_met": 0,
            "executed_steps_json": json.dumps([]),
            "parser_classifications_json": json.dumps({}),
            "preamble_lines": 0,
            "remapped_lines": 0,
            "canonical_lines": 0,
            "unparseable_lines": 0,
        }

    executed_steps: list[str] = []
    illegal_count = 0
    skip_count = 0
    error_count = 0
    last_error: str | None = None
    classifications: Counter = Counter()
    last_action = ""
    session_status = "complete"

    for step in range(max_steps):
        narrative = state_to_narrative_mbw(state, objects)
        goal_narrative = _goal_narrative_mbw(goal)
        if step == 0:
            prompt = make_turn1_prompt_mbw(narrative, goal_narrative)
        else:
            prompt = make_followup_prompt_mbw(narrative, goal_narrative, last_action)
        try:
            response = client.complete(prompt)
        except Exception as exc:
            session_status = f"api_error: {exc}"
            break

        action, label = parse_action_nl_mbw(response)
        classifications[label] += 1

        if action is None:
            current_error = label
            if current_error == last_error:
                error_count += 1
            else:
                error_count = 1
                last_error = current_error
            if error_count >= MAX_CONSECUTIVE_ERRORS:
                executed_steps.append("STEP_SKIP")
                skip_count += 1
                error_count = 0
                last_error = None
                if skip_count > MAX_SKIPS:
                    session_status = "aborted: excessive illegal steps"
                    break
            continue

        try:
            new_state = execute_action_mbw(copy.deepcopy(state), action)
            executed_steps.append(action)
            last_action = action
            state = new_state
            error_count = 0
            last_error = None
            recent = [s for s in executed_steps[-LOOP_WINDOW:] if s != "STEP_SKIP"]
            if len(recent) >= LOOP_WINDOW and len(set(recent)) <= 2:
                session_status = "aborted: planning_loop"
                break
            if goal_reached_mbw(state, goal):
                break
        except ValueError:
            illegal_count += 1
            current_error = f"illegal:{action}"
            if current_error == last_error:
                error_count += 1
            else:
                error_count = 1
                last_error = current_error
            if error_count >= MAX_CONSECUTIVE_ERRORS:
                executed_steps.append("STEP_SKIP")
                skip_count += 1
                error_count = 0
                last_error = None
                if skip_count > MAX_SKIPS:
                    session_status = "aborted: excessive illegal steps"
                    break
            else:
                executed_steps.append(action)
                last_action = action

    goals_total = len(goal)
    goals_met = sum(1 for x, y in goal.items() if state["craves"].get(x) == y) if goals_total else 0
    pga = round(goals_met / goals_total, 4) if goals_total else None

    fis = None
    try:
        _, state_fis, _ = parse_state_from_text_mbw(problem_text)
        for fis_idx, act in enumerate(executed_steps):
            try:
                state_fis = execute_action_mbw(copy.deepcopy(state_fis), act)
            except ValueError:
                fis = fis_idx
                break
    except Exception:
        pass

    return {
        "problem_id": problem_id,
        "session_status": session_status,
        "goal_reached": goal_reached_mbw(state, goal),
        "executed_length": len(executed_steps),
        "illegal_action_count": illegal_count,
        "skip_count": skip_count,
        "first_illegal_step": fis,
        "partial_goal_achievement": pga,
        "goals_met": goals_met,
        "executed_steps_json": json.dumps(executed_steps),
        "parser_classifications_json": json.dumps(dict(classifications)),
        "preamble_lines": classifications.get("preamble_only", 0),
        "remapped_lines": classifications.get("remapped", 0),
        "canonical_lines": classifications.get("canonical", 0),
        "unparseable_lines": classifications.get("unparseable", 0),
    }


FIELDNAMES = [
    "problem_id", "model", "difficulty", "contamination_pole",
    "session_status", "goal_reached", "executed_length",
    "illegal_action_count", "skip_count", "first_illegal_step",
    "partial_goal_achievement", "goals_met",
    "executed_steps_json", "parser_classifications_json",
    "preamble_lines", "remapped_lines", "canonical_lines", "unparseable_lines",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+",
                        default=["anthropic/claude-sonnet-4",
                                 "openai/gpt-4o",
                                 "meta-llama/llama-3.1-8b-instruct"])
    parser.add_argument("--output", default=str(OUT_DEFAULT))
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--problem-ids", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set (and --dry-run not passed). Exiting.",
              file=sys.stderr)
        sys.exit(1)

    bank = pd.read_csv(BANK_CSV)
    mbw_canon = bank[(bank["variant_type"] == "canonical") & bank["problem_id"].str.startswith("MBW")]
    mbw_canon = mbw_canon.set_index("problem_id")

    done: set[tuple[str, str]] = set()
    output_path = Path(args.output)
    if args.resume and output_path.exists():
        existing = pd.read_csv(output_path)
        done = set(zip(existing["problem_id"], existing["model"]))
        print(f"Resuming — {len(done)} pairs already done")

    write_header = not (args.resume and output_path.exists())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(output_path, "a", newline="")
    writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
    if write_header:
        writer.writeheader()

    models_to_run = args.models
    problem_ids = args.problem_ids
    if args.smoke:
        models_to_run = [args.models[0]]
        if problem_ids is None:
            problem_ids = ["MBW_010"]

    all_problem_ids = list(mbw_canon.index)
    if problem_ids:
        all_problem_ids = [p for p in all_problem_ids if str(p) in set(str(x) for x in problem_ids)]
    if args.limit:
        all_problem_ids = all_problem_ids[:args.limit]
    if args.smoke:
        all_problem_ids = all_problem_ids[:1]

    for model_str in models_to_run:
        print(f"\n--- {model_str} | {len(all_problem_ids)} MBW problems ---")
        client = DryRunClientMBW(model_str) if args.dry_run else None
        if client is None:
            from probes.behavioral.model_client import ModelClient
            client = ModelClient(model_str, temperature=0.0)

        for pid in all_problem_ids:
            if (pid, model_str) in done:
                print(f"  {pid} | skipped (resume)")
                continue
            row = mbw_canon.loc[pid]
            problem_text = row["problem_text"]
            print(f"  {pid} | running...", end=" ", flush=True)
            if args.dry_run:
                client = DryRunClientMBW(model_str)
            result = run_session_mbw(
                pid, problem_text, client,
                max_steps=args.max_steps,
                verbose=args.smoke or args.dry_run,
            )
            out_row = {
                "problem_id": pid,
                "model": model_str,
                "difficulty": row.get("difficulty"),
                "contamination_pole": row.get("contamination_pole"),
                **result,
            }
            for f in FIELDNAMES:
                out_row.setdefault(f, None)
            writer.writerow({f: out_row[f] for f in FIELDNAMES})
            out_file.flush()
            print(
                f"status={result['session_status']:<35} | "
                f"goal={result['goal_reached']} | "
                f"exec={result['executed_length']} | "
                f"illegal={result['illegal_action_count']} | "
                f"remapped={result['remapped_lines']} | "
                f"preamble={result['preamble_lines']}"
            )

    out_file.close()
    print(f"\nDone. Results in {output_path}")


if __name__ == "__main__":
    main()
