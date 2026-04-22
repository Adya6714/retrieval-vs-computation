#!/usr/bin/env python3
"""
CCI + TEP Testing Pipeline for Blocksworld PDDL Instances

Interactive and batch tooling to evaluate plan–execution consistency (CCI)
and trajectory error propagation (TEP) on Blocksworld problems expressed as
PDDL problem files.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# -----------------------------------------------------------------------------
# PDDL PARSER
# -----------------------------------------------------------------------------


def parse_pddl(filepath: str | Path) -> Tuple[List[str], Dict[str, Any], Dict[str, Optional[str]]]:
    """
    Parse a Blocksworld PDDL problem file.

    Returns:
        objects: list of block names
        init_state: dict with keys on, clear, on_table, holding
        goal: mapping top_block -> block_below, or None if that block should be on the table
    """
    path = Path(filepath)
    text = path.read_text(encoding="utf-8")

    obj_m = re.search(
        r"\(:objects\s+([^)]+)\)",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not obj_m:
        raise ValueError(f"No :objects section in {path}")
    raw_objs = obj_m.group(1).strip()
    objects = []
    for tok in raw_objs.split():
        if tok == "-":
            break
        objects.append(tok.lower())

    lo = text.lower()
    i0 = lo.find("(:init")
    g0 = lo.find("(:goal")
    if i0 < 0 or g0 < 0 or g0 <= i0:
        raise ValueError(f"Expected :init before :goal in {path}")
    init_body = text[i0:g0]

    on: Dict[str, str] = {}
    on_table: Set[str] = set()
    clear: Set[str] = set()
    holding: Optional[str] = None

    for m in re.finditer(r"\(\s*on-?table\s+(\w+)\s*\)", init_body, re.IGNORECASE):
        on_table.add(m.group(1).lower())
    for m in re.finditer(r"\(\s*on\s+(\w+)\s+(\w+)\s*\)", init_body, re.IGNORECASE):
        top, bot = m.group(1).lower(), m.group(2).lower()
        on[top] = bot
    for m in re.finditer(r"\(\s*clear\s+(\w+)\s*\)", init_body, re.IGNORECASE):
        clear.add(m.group(1).lower())
    if re.search(r"\(\s*hand-?empty\s*\)", init_body, re.IGNORECASE):
        holding = None
    hm = re.search(r"\(\s*holding\s+(\w+)\s*\)", init_body, re.IGNORECASE)
    if hm:
        holding = hm.group(1).lower()

    init_state: Dict[str, Any] = {
        "on": on,
        "clear": clear,
        "on_table": on_table,
        "holding": holding,
    }

    goal_body = text[g0:]
    goal: Dict[str, Optional[str]] = {}
    for m in re.finditer(r"\(\s*on\s+(\w+)\s+(\w+)\s*\)", goal_body, re.IGNORECASE):
        goal[m.group(1).lower()] = m.group(2).lower()
    for m in re.finditer(r"\(\s*on-?table\s+(\w+)\s*\)", goal_body, re.IGNORECASE):
        goal[m.group(1).lower()] = None

    return objects, init_state, goal


# -----------------------------------------------------------------------------
# STATE TO NATURAL LANGUAGE
# -----------------------------------------------------------------------------


def state_to_narrative(state: Dict[str, Any], objects: List[str]) -> str:
    """Describe the current blocksworld state in natural language."""
    on = state["on"]
    on_table = state["on_table"]
    holding = state["holding"]

    blocks = set(objects) | set(on.keys()) | set(on.values()) | set(on_table)
    sentences: List[str] = []
    for b in sorted(blocks):
        if holding == b:
            continue
        if b in on:
            sentences.append(f"Block {b} is on block {on[b]}.")
        elif b in on_table:
            sentences.append(f"Block {b} is on the table.")

    if holding:
        sentences.append(f"The hand is holding block {holding}.")
    else:
        sentences.append("The hand is empty.")

    return " ".join(sentences)


# -----------------------------------------------------------------------------
# ERROR INJECTION
# -----------------------------------------------------------------------------


def inject_error(state: Dict[str, Any], objects: List[str]) -> Tuple[Dict[str, Any], str]:
    """
    Randomly perturb a stacked relation to produce a false state description
    (for TEP-style probes). Uses global randomness (not reproducible).
    """
    rng = random.Random()
    modified = copy.deepcopy(state)

    if not modified["on"]:
        return modified, "NO ERROR INJECTED (no stacked blocks)"

    stacked_blocks = sorted(modified["on"].keys())
    block_to_move = rng.choice(stacked_blocks)
    original_bottom = modified["on"][block_to_move]

    possible_targets = sorted(
        [
            b
            for b in objects
            if b in modified["clear"]
            and b != block_to_move
            and b != original_bottom
        ]
    )
    if not possible_targets:
        possible_targets = sorted(
            [b for b in objects if b != block_to_move and b != original_bottom]
        )
    if not possible_targets:
        return modified, "NO ERROR INJECTED (could not find target)"

    new_bottom = rng.choice(possible_targets)

    del modified["on"][block_to_move]
    modified["clear"].add(original_bottom)
    modified["on"][block_to_move] = new_bottom
    modified["clear"].discard(new_bottom)

    change_desc = (
        f"INJECTED: {block_to_move.upper()} on {original_bottom.upper()} "
        f"-> {block_to_move.upper()} on {new_bottom.upper()}"
    )
    return modified, change_desc


# -----------------------------------------------------------------------------
# ACTION EXECUTION
# -----------------------------------------------------------------------------


def execute_action(state: Dict[str, Any], action: str) -> Dict[str, Any]:
    """Apply one Blocksworld action string; returns a new state dict."""
    s = copy.deepcopy(state)
    parts = action.strip().split()
    if not parts:
        raise ValueError("Empty action")
    verb = parts[0].lower()

    if verb == "pick-up" and len(parts) == 2:
        x = parts[1].lower()
        if s["holding"] is not None:
            raise ValueError("Hand not empty")
        if x not in s["on_table"] or x not in s["clear"]:
            raise ValueError(f"pick-up {x}: illegal")
        s["on_table"].discard(x)
        s["clear"].discard(x)
        s["holding"] = x
        return s

    if verb == "put-down" and len(parts) == 2:
        x = parts[1].lower()
        if s["holding"] != x:
            raise ValueError(f"put-down {x}: not holding {x}")
        s["holding"] = None
        s["on_table"].add(x)
        s["clear"].add(x)
        return s

    if verb == "stack" and len(parts) == 3:
        x, y = parts[1].lower(), parts[2].lower()
        if s["holding"] != x or y not in s["clear"] or x == y:
            raise ValueError(f"stack {x} {y}: illegal")
        s["holding"] = None
        s["on"][x] = y
        s["clear"].discard(y)
        s["clear"].add(x)
        return s

    if verb == "unstack" and len(parts) == 3:
        x, y = parts[1].lower(), parts[2].lower()
        if s["holding"] is not None:
            raise ValueError("Hand not empty")
        if s["on"].get(x) != y or x not in s["clear"]:
            raise ValueError(f"unstack {x} {y}: illegal")
        del s["on"][x]
        s["clear"].discard(x)
        s["clear"].add(y)
        s["holding"] = x
        return s

    raise ValueError(f"Unknown or malformed action: {action!r}")


def goal_reached(state: Dict[str, Any], goal: Dict[str, Optional[str]]) -> bool:
    """Returns True when all goal on-relations are satisfied."""
    for top, bottom in goal.items():
        if state["on"].get(top) != bottom:
            return False
    return True


def seeded_inject_error(
    state: Dict[str, Any], objects: List[str], seed_str: str
) -> Tuple[Dict[str, Any], str]:
    """
    Deterministic inject_error. Same seed_str always produces the same
    false claim so reruns are reproducible.
    """
    import random as _random

    rng = _random.Random(seed_str)

    modified = copy.deepcopy(state)

    if not modified["on"]:
        return modified, "NO ERROR INJECTED (no stacked blocks)"

    stacked_blocks = sorted(modified["on"].keys())
    block_to_move = rng.choice(stacked_blocks)
    original_bottom = modified["on"][block_to_move]

    possible_targets = sorted(
        [
            b
            for b in objects
            if b in modified["clear"]
            and b != block_to_move
            and b != original_bottom
        ]
    )
    if not possible_targets:
        possible_targets = sorted(
            [b for b in objects if b != block_to_move and b != original_bottom]
        )
    if not possible_targets:
        return modified, "NO ERROR INJECTED (could not find target)"

    new_bottom = rng.choice(possible_targets)

    del modified["on"][block_to_move]
    modified["clear"].add(original_bottom)
    modified["on"][block_to_move] = new_bottom
    modified["clear"].discard(new_bottom)

    change_desc = (
        f"INJECTED: {block_to_move.upper()} on {original_bottom.upper()} "
        f"-> {block_to_move.upper()} on {new_bottom.upper()}"
    )
    return modified, change_desc


# -----------------------------------------------------------------------------
# PROMPT TEMPLATES
# -----------------------------------------------------------------------------


def make_phase1_prompt(narrative: str, goal_narrative: str) -> str:
    """Initial planning prompt: describe world and ask for a plan."""
    return (
        "You are solving a Blocksworld planning task.\n\n"
        f"Current state: {narrative}\n\n"
        f"Goal: {goal_narrative}\n\n"
        "Respond with a numbered list of actions, one per line. "
        "Use only these forms: pick-up X, put-down X, stack X Y, unstack X Y."
    )


def make_turn1_prompt(narrative: str, goal_narrative: str) -> str:
    """First execution turn after a plan exists (shorthand alias)."""
    return make_phase1_prompt(narrative, goal_narrative)


def make_followup_prompt(
    narrative: str,
    goal_narrative: str,
    last_action: str,
    error_note: str = "",
) -> str:
    """Prompt for the next step given the latest state and optional corruption note."""
    extra = f"\nNote: {error_note}\n" if error_note else ""
    return (
        f"Current state after your last action ({last_action}): {narrative}\n\n"
        f"Goal remains: {goal_narrative}\n"
        f"{extra}\n"
        "What is the single next action? Reply with one line only."
    )


# -----------------------------------------------------------------------------
# INTERACTIVE TRACKER
# -----------------------------------------------------------------------------


def run_tracker(
    filepath: str | Path,
    mode: str = "cci",
    inject_at: int = 3,
    max_steps: int = 50,
) -> None:
    """Load a PDDL problem and step through actions interactively (stdin)."""
    objects, state, goal = parse_pddl(filepath)
    goal_parts = []
    for top, bot in goal.items():
        if bot is None:
            goal_parts.append(f"block {top} on the table")
        else:
            goal_parts.append(f"block {top} on block {bot}")
    goal_narrative = "; ".join(goal_parts) if goal_parts else "(empty goal)"

    print("Loaded:", filepath)
    print("Goal:", goal_narrative)
    step = 0
    injected = False

    while step < max_steps:
        narrative = state_to_narrative(state, objects)
        print(f"\n--- Step {step} ---\n{narrative}\n")

        if mode == "tep" and step == inject_at and not injected:
            state, note = inject_error(state, objects)
            injected = True
            print(f"[TEP] {note}")
            narrative = state_to_narrative(state, objects)
            print(narrative)

        if goal_reached(state, goal):
            print("Goal reached.")
            return

        try:
            line = input("Next action (or 'quit'): ").strip()
        except EOFError:
            print()
            return
        if line.lower() in ("quit", "q", "exit"):
            return
        if not line:
            continue
        try:
            state = execute_action(state, line)
        except ValueError as e:
            print(f"Illegal: {e}")
            continue
        step += 1

    print(f"Stopped after {max_steps} steps (limit).")


# -----------------------------------------------------------------------------
# GENERATE + BATCH
# -----------------------------------------------------------------------------


def cmd_generate(args: argparse.Namespace) -> None:
    """Print JSON lines of prompts for each problem file."""
    for fp in args.files:
        objects, state, goal = parse_pddl(fp)
        nar = state_to_narrative(state, objects)
        goal_parts = []
        for top, bot in goal.items():
            if bot is None:
                goal_parts.append(f"block {top} on the table")
            else:
                goal_parts.append(f"block {top} on block {bot}")
        goal_narrative = "; ".join(goal_parts)
        rec = {
            "file": str(fp),
            "phase1": make_phase1_prompt(nar, goal_narrative),
            "turn1": make_turn1_prompt(nar, goal_narrative),
        }
        print(json.dumps(rec, ensure_ascii=False))


def cmd_batch(args: argparse.Namespace) -> None:
    """Batch mode: emit one prompt record per line for downstream tooling."""
    cmd_generate(args)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_track = sub.add_parser("track", help="Interactive stepping")
    p_track.add_argument("pddl", type=Path, help="Path to a PDDL problem file")
    p_track.add_argument(
        "--mode",
        choices=("cci", "tep"),
        default="cci",
        help="cci: normal; tep: inject a false state mid-run",
    )
    p_track.add_argument(
        "--inject-at",
        type=int,
        default=3,
        metavar="N",
        help="Step index for TEP injection (default: 3)",
    )
    p_track.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Maximum interactive steps (default: 50)",
    )

    p_gen = sub.add_parser("generate", help="Emit JSON prompts for files")
    p_gen.add_argument("files", nargs="+", type=Path, help="PDDL problem files")

    p_batch = sub.add_parser("batch", help="Alias for generate (batch JSON lines)")
    p_batch.add_argument("files", nargs="+", type=Path, help="PDDL problem files")

    ns = parser.parse_args(argv)

    if ns.command == "track":
        run_tracker(ns.pddl, mode=ns.mode, inject_at=ns.inject_at, max_steps=ns.max_steps)
    elif ns.command == "generate":
        cmd_generate(ns)
    elif ns.command == "batch":
        cmd_batch(ns)


if __name__ == "__main__":
    main()
