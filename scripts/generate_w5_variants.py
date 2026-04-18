#!/usr/bin/env python3
"""Generate W5 (procedural regeneration) variants for blocksworld canonical rows.

W5 means: same block set, randomly re-drawn initial and goal states, plan
found by Fast Downward.  The generated row is appended directly to
question_bank.csv with a new problem_id of the form ``{source_id}_W5``.

Usage
-----
  python scripts/generate_w5_variants.py \\
      --planbench /path/to/LLMs-Planning \\
      --downward  /path/to/downward/fast-downward.py

  # preview without writing
  python scripts/generate_w5_variants.py \\
      --planbench /path/to/LLMs-Planning \\
      --downward  /path/to/downward/fast-downward.py \\
      --dry-run

Both --planbench and --downward are required for actual execution; they are
not needed with --dry-run (the script will skip FD and only report what it
*would* generate).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from probes.common.io import QUESTION_BANK_PATH, QUESTION_BANK_COLUMNS


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VARIANT_TYPE = "W5"

_BLOCKSWORLD_SUBTYPES = {"blocksworld"}
_MYSTERY_SUBTYPES = {"mystery_blocksworld"}

# Candidate locations for the 4-ops blocksworld domain file, relative to the
# LLMs-Planning repo root (--planbench).
_DOMAIN_CANDIDATES_RELATIVE = [
    "plan-bench/instances/blocksworld/generated_domain.pddl",
    "plan-bench/pddlgenerators/blocksworld/4ops/domain.pddl",
    "plan-bench/pddlgenerators/blocksworld/domain.pddl",
]


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def _seed_from_problem_id(problem_id: str) -> int:
    digest = hashlib.sha256(problem_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


# ---------------------------------------------------------------------------
# Block-name extraction
# ---------------------------------------------------------------------------

def _extract_block_names(problem_text: str) -> List[str]:
    """Return a sorted list of unique single-character block names found in
    the canonical problem_text.

    We look inside the 'Current state:' section (before 'Goal:') to avoid
    picking up names that only appear in the goal.  Falls back to the whole
    text if that section is absent.
    """
    text = problem_text.strip().strip('"')

    # Prefer the current-state section for extraction
    cs_match = re.search(r"Current state:(.*?)(?:Goal:|$)", text, re.IGNORECASE | re.DOTALL)
    search_text = cs_match.group(1) if cs_match else text

    # Block names appear as lowercase word after "block " or in a listing
    # like "Blocks a, b, c, and d are ..."
    raw_names: Set[str] = set()

    for m in re.finditer(r"\bblock\s+([a-z]\w*)\b", search_text, re.IGNORECASE):
        raw_names.add(m.group(1).lower())

    listing = re.search(
        r"[Bb]locks?\s+([\w,\s]+?)\s+are\s+(?:clear\s+and\s+)?on\s+the\s+table",
        search_text,
    )
    if listing:
        for tok in re.findall(r"\b([a-z])\b", listing.group(1)):
            raw_names.add(tok.lower())

    return sorted(raw_names)


# ---------------------------------------------------------------------------
# Random stack generation  (carried over from old script)
# ---------------------------------------------------------------------------

def _random_stacks(
    blocks: Sequence[str], rng: np.random.Generator
) -> Tuple[Tuple[str, ...], ...]:
    """Randomly partition *blocks* into one or more stacks."""
    blocks_list = list(blocks)
    rng.shuffle(blocks_list)
    stack_count = int(rng.integers(1, len(blocks_list) + 1))
    stacks: List[List[str]] = [[] for _ in range(stack_count)]
    for block in blocks_list:
        stacks[int(rng.integers(0, stack_count))].append(block)
    stacks = [s for s in stacks if s]
    rng.shuffle(stacks)
    return tuple(tuple(s) for s in stacks)


def _stacks_to_positions(stacks: Tuple[Tuple[str, ...], ...]) -> Dict[str, str]:
    """Map every block to what it sits on ('table' or another block name)."""
    positions: Dict[str, str] = {}
    for stack in stacks:
        for i, block in enumerate(stack):
            positions[block] = "table" if i == 0 else stack[i - 1]
    return positions


def _states_equal(
    a: Tuple[Tuple[str, ...], ...], b: Tuple[Tuple[str, ...], ...]
) -> bool:
    return _stacks_to_positions(a) == _stacks_to_positions(b)


# ---------------------------------------------------------------------------
# PDDL generation
# ---------------------------------------------------------------------------

def _pddl_facts_for_stacks(stacks: Tuple[Tuple[str, ...], ...]) -> List[str]:
    """Return PDDL predicates describing the given stacks."""
    facts: List[str] = []
    for stack in stacks:
        for i, block in enumerate(stack):
            if i == 0:
                facts.append(f"(ontable {block})")
            else:
                facts.append(f"(on {block} {stack[i - 1]})")
        facts.append(f"(clear {stack[-1]})")  # top block is clear
    return facts


def _write_pddl_problem(
    problem_name: str,
    blocks: List[str],
    init_stacks: Tuple[Tuple[str, ...], ...],
    goal_stacks: Tuple[Tuple[str, ...], ...],
    dest: Path,
) -> None:
    """Write a PDDL 1.2 blocksworld problem file to *dest*."""
    objects_str = " ".join(blocks)
    init_facts = ["(handempty)"] + _pddl_facts_for_stacks(init_stacks)
    goal_facts = _pddl_facts_for_stacks(goal_stacks)

    init_str = "\n    ".join(init_facts)
    goal_str = "\n    ".join(goal_facts)

    content = f"""\
(define (problem {problem_name})
  (:domain blocksworld-4ops)
  (:objects {objects_str})
  (:init
    {init_str})
  (:goal (and
    {goal_str}))
)
"""
    dest.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Natural-language state descriptions
# ---------------------------------------------------------------------------

def _english_list(items: List[str]) -> str:
    """Join a list with Oxford comma: 'a, b, and c' or 'a and b'."""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _describe_init_state(stacks: Tuple[Tuple[str, ...], ...]) -> str:
    """Generate a natural-language description of the initial state matching
    the canonical prompt format.

    Examples
    --------
    All on table:
        "Blocks a, b, c, d, and e are clear and on the table. The hand is empty."

    Mixed:
        "Block b is on block a. Block a is on the table. Block c is on the
         table. Block d is on block e. Block e is on the table. The hand is
         empty."
    """
    positions = _stacks_to_positions(stacks)

    # Check if everything is just flat on the table
    all_on_table = all(v == "table" for v in positions.values())
    if all_on_table:
        block_list = _english_list([f"block {b}" for b in sorted(positions.keys())])
        # capitalise first letter
        sentence = f"Blocks {_english_list(sorted(positions.keys()))} are clear and on the table."
        return sentence + " The hand is empty."

    # Mixed state: list every block's position
    sentences: List[str] = []
    # on-block relationships first, then on-table
    for block in sorted(positions.keys()):
        support = positions[block]
        if support == "table":
            sentences.append(f"Block {block} is on the table.")
        else:
            sentences.append(f"Block {block} is on block {support}.")
    sentences.append("The hand is empty.")
    return " ".join(sentences)


def _describe_goal_state(stacks: Tuple[Tuple[str, ...], ...]) -> str:
    """Generate a natural-language description of the goal state.

    Uses the same comma-joined ON-relationships format as the canonical rows:
    "Block b is on block a, block a is on the table."
    """
    positions = _stacks_to_positions(stacks)

    # Collect stacking chains
    clauses: List[str] = []
    for block in sorted(positions.keys()):
        support = positions[block]
        if support == "table":
            clauses.append(f"block {block} is on the table")
        else:
            clauses.append(f"block {block} is on block {support}")

    if not clauses:
        return ""
    # Capitalise first word and join
    joined = ", ".join(clauses)
    return joined[0].upper() + joined[1:] + "."


def _build_problem_text(
    init_stacks: Tuple[Tuple[str, ...], ...],
    goal_stacks: Tuple[Tuple[str, ...], ...],
) -> str:
    """Assemble the full canonical-format problem prompt."""
    init_desc = _describe_init_state(init_stacks)
    goal_desc = _describe_goal_state(goal_stacks)

    return (
        "You are a robot arm. "
        "Available actions: "
        "pick-up X (X must be clear and on the table, hand must be empty), "
        "put-down X (place X on the table), "
        "stack X Y (place X on Y; Y must be clear, you must be holding X), "
        "unstack X Y (pick up X from Y; X must be clear, hand must be empty). "
        "You can hold one block at a time. "
        f"Current state: {init_desc} "
        f"Goal: {goal_desc} "
        "Respond with a numbered list of actions only. "
        "Each action must be exactly one of: "
        "pick-up X / put-down X / stack X Y / unstack X Y. "
        "No explanation. No extra text."
    )


# ---------------------------------------------------------------------------
# Fast Downward integration  (adapted from fill_correct_answers.py)
# ---------------------------------------------------------------------------

def run_fast_downward(
    domain_path: Path, problem_path: Path, fd_path: Path
) -> Optional[List[str]]:
    """Run Fast Downward; return list of raw action strings or None on failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        plan_file = Path(tmpdir) / "plan.txt"
        result = subprocess.run(
            [
                str(fd_path),
                "--plan-file", str(plan_file),
                str(domain_path),
                str(problem_path),
                "--search", "astar(lmcut())",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if not plan_file.exists():
            print(f"    No plan found. FD stderr tail: {result.stderr[-300:]}")
            return None

        lines = plan_file.read_text().splitlines()
        return [
            line.strip().strip("()").strip()
            for line in lines
            if line.strip() and not line.strip().startswith(";")
        ]


def pddl_action_to_natural(action: str) -> str:
    """Convert a raw FD action string to the natural-language format used in
    the question bank (e.g. 'pick-up a', 'stack b a', 'unstack c b')."""
    parts = action.lower().split()
    if not parts:
        return action
    verb, args = parts[0], parts[1:]
    if verb == "pick-up" and len(args) == 1:
        return f"pick-up {args[0]}"
    if verb == "put-down" and len(args) == 1:
        return f"put-down {args[0]}"
    if verb == "stack" and len(args) == 2:
        return f"stack {args[0]} {args[1]}"
    if verb == "unstack" and len(args) == 2:
        return f"unstack {args[0]} {args[1]}"
    return action


def _find_domain_file(planbench_root: Path) -> Optional[Path]:
    """Locate the 4-ops blocksworld domain file inside the planbench repo."""
    for rel in _DOMAIN_CANDIDATES_RELATIVE:
        candidate = planbench_root / rel
        if candidate.exists():
            return candidate
    return None


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _append_rows(
    path: Path, fieldnames: List[str], new_rows: List[Dict[str, str]]
) -> None:
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writerows(new_rows)


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def _generate_w5_for_row(
    row: Dict[str, str],
    domain_path: Optional[Path],
    fd_path: Optional[Path],
    rng: np.random.Generator,
    seed_value: int,
    dry_run: bool,
) -> Optional[Dict[str, str]]:
    """Attempt to generate a W5 row for a canonical blocksworld row.

    Returns the new row dict on success, or None on failure.
    """
    pid = row["problem_id"].strip()

    # Extract block names from canonical problem_text
    blocks = _extract_block_names(row.get("problem_text", ""))
    if len(blocks) < 2:
        print(f"[FAIL]  {pid} — could not extract ≥2 block names from problem_text")
        return None

    # Generate random init / goal states (retry until they differ)
    max_attempts = 30
    init_stacks: Optional[Tuple[Tuple[str, ...], ...]] = None
    goal_stacks: Optional[Tuple[Tuple[str, ...], ...]] = None

    for _ in range(max_attempts):
        cand_init = _random_stacks(blocks, rng)
        cand_goal = _random_stacks(blocks, rng)
        if not _states_equal(cand_init, cand_goal):
            init_stacks = cand_init
            goal_stacks = cand_goal
            break

    if init_stacks is None or goal_stacks is None:
        print(f"[FAIL]  {pid} — could not generate distinct init/goal after {max_attempts} attempts")
        return None

    problem_text = _build_problem_text(init_stacks, goal_stacks)

    if dry_run:
        print(f"[DRY]   {pid}_W5 — blocks={blocks}, seed={seed_value}")
        print(f"          init : {init_stacks}")
        print(f"          goal : {goal_stacks}")
        print(f"          text : {problem_text[:120]}...")
        return None  # don't build a real row in dry-run

    # --- Write temp PDDL and call Fast Downward ---
    if domain_path is None or fd_path is None:
        print(f"[FAIL]  {pid} — --planbench / --downward required for real run")
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        problem_file = Path(tmpdir) / f"{pid}_W5_problem.pddl"
        _write_pddl_problem(
            problem_name=f"{pid.lower()}-w5",
            blocks=blocks,
            init_stacks=init_stacks,
            goal_stacks=goal_stacks,
            dest=problem_file,
        )

        print(f"[RUN]   {pid}_W5 — calling Fast Downward …")
        try:
            raw_actions = run_fast_downward(domain_path, problem_file, fd_path)
        except subprocess.TimeoutExpired:
            print(f"[FAIL]  {pid} — Fast Downward timed out")
            return None
        except Exception as exc:
            print(f"[FAIL]  {pid} — Fast Downward error: {exc}")
            return None

    if raw_actions is None:
        return None

    natural_actions = [pddl_action_to_natural(a) for a in raw_actions]
    correct_answer = "\n".join(natural_actions)
    print(f"          → {len(natural_actions)} steps")

    # Build the new row: start from canonical, override W5-specific fields
    new_row: Dict[str, str] = dict(row)
    new_row["problem_id"] = f"{pid}_W5"
    new_row["variant_type"] = VARIANT_TYPE
    new_row["problem_text"] = problem_text
    new_row["correct_answer"] = correct_answer
    new_row["contamination_pole"] = "low"
    new_row["source"] = f"generated_seed_{seed_value}"
    new_row["notes"] = "W5 procedurally generated"
    return new_row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate W5 (procedural) blocksworld variants into question_bank.csv.",
    )
    parser.add_argument(
        "--csv",
        default=QUESTION_BANK_PATH,
        help=f"Path to question bank CSV (default: {QUESTION_BANK_PATH})",
    )
    parser.add_argument(
        "--planbench",
        default=None,
        help="Path to the LLMs-Planning repo root (required unless --dry-run).",
    )
    parser.add_argument(
        "--downward",
        default=None,
        help="Path to fast-downward.py (required unless --dry-run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be generated without writing to the CSV.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"ERROR: CSV not found: {csv_path}")

    # Validate required args for real runs
    domain_path: Optional[Path] = None
    fd_path: Optional[Path] = None

    if not args.dry_run:
        if not args.planbench:
            sys.exit("ERROR: --planbench is required when not using --dry-run")
        if not args.downward:
            sys.exit("ERROR: --downward is required when not using --dry-run")

        planbench_root = Path(args.planbench)
        fd_path = Path(args.downward)

        if not planbench_root.is_dir():
            sys.exit(f"ERROR: --planbench directory not found: {planbench_root}")
        if not fd_path.exists():
            sys.exit(f"ERROR: fast-downward.py not found: {fd_path}")

        domain_path = _find_domain_file(planbench_root)
        if domain_path is None:
            sys.exit(
                f"ERROR: Could not find blocksworld domain.pddl under {planbench_root}. "
                f"Tried:\n" + "\n".join(f"  {r}" for r in _DOMAIN_CANDIDATES_RELATIVE)
            )
        print(f"Domain: {domain_path}")
        print(f"Planner: {fd_path}")

    # Load CSV
    fieldnames, rows = _load_csv(csv_path)

    missing = set(QUESTION_BANK_COLUMNS) - set(fieldnames)
    if missing:
        sys.exit(f"ERROR: question_bank CSV missing columns: {sorted(missing)}")

    # Build set of problem_ids that already have a W5 row
    existing_w5_ids: Set[str] = {
        row["problem_id"].replace("_W5", "").strip()
        for row in rows
        if row.get("variant_type", "").strip() == VARIANT_TYPE
    }

    generated = 0
    skipped = 0
    failed = 0
    new_rows: List[Dict[str, str]] = []

    for row in rows:
        pid = row.get("problem_id", "").strip()
        vtype = row.get("variant_type", "").strip().lower()
        subtype = row.get("problem_subtype", "").strip().lower()

        # Only process canonical rows
        if vtype != "canonical":
            continue

        # Only blocksworld subtypes
        if subtype in _MYSTERY_SUBTYPES:
            print(f"[SKIP]  {pid} — mystery_blocksworld not supported yet (W5)")
            skipped += 1
            continue

        if subtype not in _BLOCKSWORLD_SUBTYPES:
            skipped += 1
            continue

        # Skip if W5 already exists
        if pid in existing_w5_ids:
            print(f"[SKIP]  {pid} — W5 row already exists")
            skipped += 1
            continue

        seed_value = _seed_from_problem_id(pid)
        rng = np.random.default_rng(seed_value)

        new_row = _generate_w5_for_row(
            row=row,
            domain_path=domain_path,
            fd_path=fd_path,
            rng=rng,
            seed_value=seed_value,
            dry_run=args.dry_run,
        )

        if new_row is not None:
            new_rows.append(new_row)
            generated += 1
        elif not args.dry_run:
            failed += 1
        else:
            # dry_run returns None for successes too — count as generated
            generated += 1

    # Write to CSV (real run only)
    if not args.dry_run and new_rows:
        _append_rows(csv_path, fieldnames, new_rows)
        print(f"\nAppended {len(new_rows)} new W5 rows to {csv_path}")

    label = "Would generate" if args.dry_run else "Generated"
    print(f"\nSummary: {label.lower()}={generated}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
