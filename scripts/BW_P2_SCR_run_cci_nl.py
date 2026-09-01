"""BW Phase-2A re-run with NL-tolerant action parser.

This is a parallel runner to `scripts/BW_P2_SCR_run_cci.py`. The differences:

1. State + goal are derived from the question-bank `problem_text` (the original
   PDDL files referenced in `BW_P2_plans.csv` are not on local disk).
2. The action parser is the new `bw_action_parser_nl.remap_to_canonical`,
   which accepts natural-English variants like "put X on Y" (→ stack X Y),
   "put-down X Y" (→ stack X Y), "pick X from Y" (→ unstack X Y), etc.
3. The runner is gated by --dry-run (no API), --smoke (1 problem × 1 model),
   --resume, and --problem-ids for staged execution.

Output:
    results/raw/BW_P2_cci_nl.csv  (parallel to BW_P2_cci.csv; same schema)

Usage examples:
    python scripts/BW_P2_SCR_run_cci_nl.py --dry-run
    python scripts/BW_P2_SCR_run_cci_nl.py --smoke --models anthropic/claude-sonnet-4 \\
        --problem-ids BW_010
    python scripts/BW_P2_SCR_run_cci_nl.py --models anthropic/claude-sonnet-4 \\
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

from probes.behavioral.bw_action_parser_nl import is_preamble, remap_to_canonical
from probes.behavioral.bw_cci_pipeline import (
    execute_action,
    goal_reached,
    make_followup_prompt,
    make_turn1_prompt,
    parse_state_from_text,
    state_to_narrative,
)
from probes.behavioral.cci import compute_cci

# Question bank (standard BW) — initial state + goal source
BANK_CSV = ROOT / "data/problems/question_bank_bw.csv"

# Existing Phase-1 plans (used for CCI scoring against declared plan)
PLANS_CSV = ROOT / "results/raw/BW_P2_plans.csv"
OUT_DEFAULT = ROOT / "results/raw/BW_P2_cci_nl.csv"

MAX_CONSECUTIVE_ERRORS = 2
MAX_SKIPS = 5


# ---------------------------------------------------------------------------
# Mock client for dry runs
# ---------------------------------------------------------------------------


class DryRunClient:
    """Yields canned responses that exercise the NL-tolerant parser end-to-end."""

    def __init__(self, model: str, scripted: list[str] | None = None):
        self.model = model
        self.scripted = list(scripted) if scripted else [
            "I need to put block d on block e",  # preamble + would be stack d e
            "put d on e",                          # natural English -> stack d e
            "put-down e a",                        # 2-arg put-down -> stack e a
            "put a on b",                          # -> stack a b
            "put b on f",                          # -> stack b f
            "stack b f",                           # canonical
            "pick-up d",                           # canonical (loop guard)
        ]
        self.idx = 0

    def complete(self, prompt: str) -> str:
        if self.idx >= len(self.scripted):
            response = "put-down z"  # benign fallback
        else:
            response = self.scripted[self.idx]
        self.idx += 1
        return response


# ---------------------------------------------------------------------------
# Parsing the model response into a single action (with NL tolerance)
# ---------------------------------------------------------------------------


def parse_action_nl(response_text: str) -> tuple[str | None, str]:
    """Try to parse model output into a canonical action via the NL remap.

    Returns (canonical_action_or_None, classification_label).
    classification_label is one of:
      'canonical'      - already in canonical form, executed unchanged
      'remapped'       - mapped from NL variant
      'preamble_only'  - response contained only preamble (no action)
      'unparseable'    - no line was mappable
    """
    if response_text is None:
        return None, "unparseable"
    lines = [ln.strip() for ln in str(response_text).split("\n") if ln.strip()]
    saw_preamble = False
    for line in lines:
        # Strip leading "Step N:" / "1." / "1)" etc.
        import re
        stripped = re.sub(r"^\d+[\.\)\:]\s*", "", line)
        stripped = re.sub(r"^step\s+\d+[\.\:\)]?\s*", "", stripped, flags=re.IGNORECASE).strip()
        if not stripped:
            continue
        if is_preamble(stripped):
            saw_preamble = True
            continue
        # ALWAYS try remap first — handles `put-down X Y` -> `stack X Y` and
        # other shape-fix cases the executor would otherwise reject.
        remapped = remap_to_canonical(stripped)
        if remapped is not None:
            label = "canonical" if remapped == stripped.lower() else "remapped"
            return remapped, label
    return None, ("preamble_only" if saw_preamble else "unparseable")


def _build_goal_narrative(goal: dict) -> str:
    parts = []
    for top, bot in goal.items():
        if bot is None:
            parts.append(f"block {top} on the table")
        else:
            parts.append(f"block {top} on block {bot}")
    return "; ".join(parts) if parts else "(empty goal)"


def run_session(
    problem_id: str,
    problem_text: str,
    generated_plan: list[str],
    client,
    max_steps: int = 50,
    verbose: bool = False,
) -> dict:
    try:
        objects, state, goal = parse_state_from_text(problem_text)
    except Exception as exc:
        return {
            "problem_id": problem_id,
            "cci": None,
            "matched_steps": 0,
            "total_steps_compared": 0,
            "generated_plan_length": len(generated_plan),
            "executed_length": 0,
            "illegal_action_count": 0,
            "skip_count": 0,
            "session_status": f"error: parse_state_from_text: {exc}",
            "first_illegal_step": None,
            "partial_goal_achievement": None,
            "goals_met": 0,
            "goal_reached": False,
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
    # Loop detector: abort if same action emitted 5+ times in last 8 steps
    LOOP_WINDOW = 8
    LOOP_REPEAT = 5

    for step in range(max_steps):
        narrative = state_to_narrative(state, objects)
        goal_narrative = _build_goal_narrative(goal)
        if step == 0:
            prompt = make_turn1_prompt(narrative, goal_narrative)
        else:
            prompt = make_followup_prompt(narrative, goal_narrative, last_action)

        try:
            response = client.complete(prompt)
        except Exception as exc:
            if verbose:
                print(f"    API error at step {step}: {exc}")
            session_status = f"api_error: {exc}"
            break

        action, label = parse_action_nl(response)
        classifications[label] += 1

        if action is None:
            current_error = label  # 'preamble_only' or 'unparseable'
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
            new_state = execute_action(copy.deepcopy(state), action)
            executed_steps.append(action)
            last_action = action
            state = new_state
            error_count = 0
            last_error = None
            # Loop detector: abort if recent window has <=2 unique actions
            # (catches A-A-A and A-B-A-B alternation patterns).
            recent = [s for s in executed_steps[-LOOP_WINDOW:] if s != "STEP_SKIP"]
            if len(recent) >= LOOP_WINDOW and len(set(recent)) <= 2:
                session_status = "aborted: planning_loop"
                break
            if goal_reached(state, goal):
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

    # CCI vs declared plan
    filtered_steps = [s for s in executed_steps if s != "STEP_SKIP"]
    if session_status.startswith("aborted") or session_status.startswith("api_error") or session_status.startswith("error"):
        cci_result = {"cci": None, "matched_steps": 0, "total_steps_compared": 0}
    else:
        cci_result = compute_cci(problem_id, generated_plan, filtered_steps)

    # Goal proximity
    goals_total = len(goal)
    goals_met = sum(1 for top, bot in goal.items() if state["on"].get(top) == bot) if goals_total else 0
    pga = round(goals_met / goals_total, 4) if goals_total else None

    # First-illegal-step on a fresh state
    fis = None
    try:
        _, state_fis, _ = parse_state_from_text(problem_text)
        for fis_idx, act in enumerate(executed_steps):
            try:
                state_fis = execute_action(copy.deepcopy(state_fis), act)
            except ValueError:
                fis = fis_idx
                break
    except Exception:
        pass

    return {
        "problem_id": problem_id,
        "cci": cci_result["cci"],
        "matched_steps": cci_result["matched_steps"],
        "total_steps_compared": cci_result["total_steps_compared"],
        "generated_plan_length": len(generated_plan),
        "executed_length": len(executed_steps),
        "illegal_action_count": illegal_count,
        "skip_count": skip_count,
        "session_status": session_status,
        "first_illegal_step": fis,
        "partial_goal_achievement": pga,
        "goals_met": goals_met,
        "goal_reached": goal_reached(state, goal),
        "executed_steps_json": json.dumps(executed_steps),
        "parser_classifications_json": json.dumps(dict(classifications)),
        "preamble_lines": classifications.get("preamble_only", 0),
        "remapped_lines": classifications.get("remapped", 0),
        "canonical_lines": classifications.get("canonical", 0),
        "unparseable_lines": classifications.get("unparseable", 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


FIELDNAMES = [
    "problem_id", "model", "difficulty", "contamination_pole",
    "cci", "matched_steps", "total_steps_compared",
    "generated_plan_length", "executed_length",
    "illegal_action_count", "skip_count", "session_status",
    "first_illegal_step", "partial_goal_achievement", "goals_met",
    "goal_reached", "executed_steps_json",
    "parser_classifications_json",
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
    parser.add_argument("--dry-run", action="store_true",
                        help="No API; use canned responses for end-to-end test")
    parser.add_argument("--smoke", action="store_true",
                        help="Limit to 1 model × 1 problem; useful as gate before full sweep")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of problems per model")
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("OPENROUTER_API_KEY"):
        print("OPENROUTER_API_KEY not set (and --dry-run not passed). Exiting.",
              file=sys.stderr)
        sys.exit(1)

    # Load Phase-1 plans for CCI scoring
    plans = pd.read_csv(PLANS_CSV)
    plans = plans[plans["plan_length"] > 0].reset_index(drop=True)
    # Load question bank for problem_text → state derivation
    bank = pd.read_csv(BANK_CSV)
    bank_canon = bank[bank["variant_type"] == "canonical"].set_index("problem_id")["problem_text"].to_dict()

    # Resume tracking
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
    if args.smoke:
        models_to_run = [args.models[0]]
        if args.problem_ids is None:
            args.problem_ids = ["BW_010"]

    for model_str in models_to_run:
        model_tail = model_str.split("/")[-1]
        model_plans = plans[plans["model"].str.contains(model_tail, case=False, na=False)]
        if args.problem_ids:
            allowed = set(str(x) for x in args.problem_ids)
            model_plans = model_plans[model_plans["problem_id"].astype(str).isin(allowed)]
        if args.limit:
            model_plans = model_plans.head(args.limit)
        if args.smoke:
            model_plans = model_plans.head(1)

        if model_plans.empty:
            print(f"  WARN: no plans for '{model_tail}'", file=sys.stderr)
            continue

        print(f"\n--- {model_str} | {len(model_plans)} problems ---")

        client = DryRunClient(model_str) if args.dry_run else None
        if client is None:
            from probes.behavioral.model_client import ModelClient
            client = ModelClient(model_str, temperature=0.0)

        for _, row in model_plans.iterrows():
            pid = row["problem_id"]
            if (pid, model_str) in done:
                print(f"  {pid} | skipped (resume)")
                continue
            problem_text = bank_canon.get(pid)
            if not problem_text:
                print(f"  {pid} | WARN: no canonical problem_text — skipping")
                continue
            try:
                generated_plan = json.loads(row["parsed_plan_json"])
            except Exception:
                generated_plan = []
            print(f"  {pid} | plan_len={len(generated_plan)} | running...", end=" ", flush=True)
            # Dry run: reset the canned-response iterator each problem
            if args.dry_run:
                client = DryRunClient(model_str)
            result = run_session(
                pid, problem_text, generated_plan, client,
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
