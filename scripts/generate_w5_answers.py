#!/usr/bin/env python3
"""Generate correct_answer for W5 blocksworld variants.

W5 is the "reverse planning" variant: the initial and goal states of the
canonical problem are swapped.  Consequently, the reversed canonical plan
(with each action semantically inverted) should be a valid plan for the W5
problem.

Inverse action mapping
----------------------
  pick-up X   →  put-down X
  put-down X  →  pick-up X
  stack X Y   →  unstack X Y
  unstack X Y →  stack X Y

Algorithm
---------
1. For each W5 blocksworld row whose correct_answer is empty:
   a. Find the canonical row with the same problem_id.
   b. Reverse the canonical move list and invert each action.
   c. Parse the W5 problem_text to extract the initial state.
   d. Simulate the reversed plan step-by-step, checking preconditions.
   e. If all moves are legal → fill correct_answer.
   f. Otherwise → print WARNING, leave correct_answer empty.
2. Write the modified rows back to the same CSV (unless --dry-run).

Usage
-----
  python scripts/generate_w5_answers.py
  python scripts/generate_w5_answers.py --dry-run
  python scripts/generate_w5_answers.py --csv path/to/other.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from probes.common.io import QUESTION_BANK_PATH, QUESTION_BANK_COLUMNS


# ---------------------------------------------------------------------------
# Action model
# ---------------------------------------------------------------------------

class BlocksworldState:
    """Minimal blocksworld state for plan verification.

    Tracks:
      on_table: set of blocks resting directly on the table
      on:       dict mapping block → block it rests on  (block below it)
      clear:    set of blocks with nothing on top
      holding:  block currently held, or None
      hand_empty: True iff holding is None
    """

    def __init__(
        self,
        on_table: Set[str],
        on: Dict[str, str],
        clear: Set[str],
        holding: Optional[str] = None,
    ) -> None:
        self.on_table = set(on_table)
        self.on = dict(on)
        self.clear = set(clear)
        self.holding = holding

    @property
    def hand_empty(self) -> bool:
        return self.holding is None

    def copy(self) -> "BlocksworldState":
        return BlocksworldState(
            on_table=set(self.on_table),
            on=dict(self.on),
            clear=set(self.clear),
            holding=self.holding,
        )

    # ------------------------------------------------------------------
    # Transitions (return new state or raise ValueError on precondition fail)
    # ------------------------------------------------------------------

    def pick_up(self, x: str) -> "BlocksworldState":
        """pick-up X: X on table, X clear, hand empty → holding X."""
        if x not in self.on_table:
            raise ValueError(f"pick-up {x}: {x} is not on the table")
        if x not in self.clear:
            raise ValueError(f"pick-up {x}: {x} is not clear")
        if not self.hand_empty:
            raise ValueError(f"pick-up {x}: hand is not empty (holding {self.holding})")
        s = self.copy()
        s.on_table.discard(x)
        s.clear.discard(x)
        s.holding = x
        return s

    def put_down(self, x: str) -> "BlocksworldState":
        """put-down X: holding X → X on table, clear, hand empty."""
        if self.holding != x:
            raise ValueError(f"put-down {x}: not holding {x} (holding {self.holding!r})")
        s = self.copy()
        s.holding = None
        s.on_table.add(x)
        s.clear.add(x)
        return s

    def stack(self, x: str, y: str) -> "BlocksworldState":
        """stack X Y: holding X, Y clear → X on Y, hand empty."""
        if self.holding != x:
            raise ValueError(f"stack {x} {y}: not holding {x} (holding {self.holding!r})")
        if y not in self.clear:
            raise ValueError(f"stack {x} {y}: {y} is not clear")
        if x == y:
            raise ValueError(f"stack {x} {y}: cannot stack a block on itself")
        s = self.copy()
        s.holding = None
        s.on[x] = y
        s.clear.discard(y)
        s.clear.add(x)
        return s

    def unstack(self, x: str, y: str) -> "BlocksworldState":
        """unstack X Y: X on Y, X clear, hand empty → holding X, Y clear."""
        if s_on := self.on.get(x):
            if s_on != y:
                raise ValueError(f"unstack {x} {y}: {x} is on {s_on}, not {y}")
        else:
            raise ValueError(f"unstack {x} {y}: {x} is not on any block (it may be on the table)")
        if x not in self.clear:
            raise ValueError(f"unstack {x} {y}: {x} is not clear")
        if not self.hand_empty:
            raise ValueError(f"unstack {x} {y}: hand is not empty (holding {self.holding})")
        s = self.copy()
        del s.on[x]
        s.clear.discard(x)
        s.clear.add(y)
        s.holding = x
        return s

    def apply(self, move: str) -> "BlocksworldState":
        """Apply a single natural-language move string and return the new state."""
        move = move.strip()
        lm = move.lower()

        # pick-up X
        m = re.fullmatch(r"pick-up\s+(\w+)", lm)
        if m:
            return self.pick_up(m.group(1))

        # put-down X
        m = re.fullmatch(r"put-down\s+(\w+)", lm)
        if m:
            return self.put_down(m.group(1))

        # stack X Y
        m = re.fullmatch(r"stack\s+(\w+)\s+(\w+)", lm)
        if m:
            return self.stack(m.group(1), m.group(2))

        # unstack X Y
        m = re.fullmatch(r"unstack\s+(\w+)\s+(\w+)", lm)
        if m:
            return self.unstack(m.group(1), m.group(2))

        raise ValueError(f"Unrecognised move format: {move!r}")


# ---------------------------------------------------------------------------
# State parsing from W5 problem_text
# ---------------------------------------------------------------------------

def _parse_w5_initial_state(problem_text: str) -> BlocksworldState:
    """Extract the initial blocksworld state from a W5 problem_text string.

    The W5 problem_text has "Current state:" section describing which blocks
    are on the table and which are stacked on others, e.g.:

      Current state: Block i is on block f, block f is on block e, ... 
                     Block h is on the table. The hand is empty.

    Or the simpler canonical form:
      Current state: Blocks i, f, e, j, and h are clear and on the table.
    """
    # Strip surrounding quotes that CSV quoting may have preserved
    text = problem_text.strip().strip('"')

    # Locate the "Current state:" section
    cs_match = re.search(r"Current state:(.*?)(?:Goal:|$)", text, re.IGNORECASE | re.DOTALL)
    if not cs_match:
        raise ValueError("Could not find 'Current state:' in problem_text")

    cs_text = cs_match.group(1).strip()

    on_table: Set[str] = set()
    on: Dict[str, str] = set()   # type: ignore[assignment]
    on = {}

    # Pattern: "Block X is on block Y"   (X rests on Y)
    for m in re.finditer(r"[Bb]lock\s+(\w+)\s+is\s+on\s+block\s+(\w+)", cs_text):
        x, y = m.group(1), m.group(2)
        on[x] = y

    # Pattern: "Block X is on the table"
    for m in re.finditer(r"[Bb]lock\s+(\w+)\s+is\s+on\s+the\s+table", cs_text):
        on_table.add(m.group(1))

    # Pattern: "Blocks X, Y, and Z are clear and on the table" / "are on the table"
    multi_table = re.search(
        r"[Bb]locks?\s+([\w,\s]+?)\s+are\s+(?:clear\s+and\s+)?on\s+the\s+table",
        cs_text,
    )
    if multi_table:
        block_list_str = multi_table.group(1)
        for b in re.findall(r"\b([a-zA-Z]\w*)\b", block_list_str):
            if b.lower() not in ("and", "are", "clear", "the", "table"):
                on_table.add(b)

    if not on_table and not on:
        raise ValueError(f"Could not parse any block positions from: {cs_text!r}")

    # All blocks referenced
    all_blocks: Set[str] = on_table | set(on.keys()) | set(on.values())

    # A block is clear iff nothing is on top of it
    has_something_on_top: Set[str] = set(on.values())
    clear = all_blocks - has_something_on_top - {b for b in all_blocks if b not in on_table and b not in on}
    # Blocks in on_table but not supporting anything are clear
    clear = set()
    for b in all_blocks:
        if b not in has_something_on_top:
            clear.add(b)

    return BlocksworldState(on_table=on_table, on=on, clear=clear)


# ---------------------------------------------------------------------------
# Plan manipulation helpers
# ---------------------------------------------------------------------------

_INVERSE_MAP = {
    "pick-up": "put-down",
    "put-down": "pick-up",
    "stack":    "unstack",
    "unstack":  "stack",
}


def _invert_move(move: str) -> str:
    """Return the inverse of a single blocksworld move.

    Inverse action mapping:
      pick-up X   →  put-down X
      put-down X  →  pick-up X
      stack X Y   →  unstack X Y
      unstack X Y →  stack X Y
    """
    move = move.strip()
    lm = move.lower()

    for verb in ("pick-up", "put-down", "stack", "unstack"):
        if lm.startswith(verb + " "):
            rest = move[len(verb):].strip()   # preserve original case for block names
            inverse_verb = _INVERSE_MAP[verb]
            return f"{inverse_verb} {rest}"

    raise ValueError(f"Cannot invert unrecognised move: {move!r}")


def _make_reversed_plan(canonical_moves: List[str]) -> List[str]:
    """Reverse the move list and invert each move."""
    return [_invert_move(m) for m in reversed(canonical_moves)]


# ---------------------------------------------------------------------------
# Plan verification
# ---------------------------------------------------------------------------

def _verify_plan(
    initial_state: BlocksworldState, moves: List[str]
) -> Tuple[bool, Optional[int], Optional[str]]:
    """Simulate the plan from initial_state.

    Returns
    -------
    (ok, bad_step_index, error_message)
      ok is True iff all moves are legal.
      bad_step_index is 0-indexed position of the first illegal move (or None).
    """
    state = initial_state
    for i, move in enumerate(moves):
        try:
            state = state.apply(move)
        except ValueError as exc:
            return False, i, str(exc)
    return True, None, None


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _save_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def _parse_moves(raw_answer: str) -> List[str]:
    """Split newline-separated move list; strip empty lines."""
    lines = [ln.strip() for ln in raw_answer.splitlines()]
    return [ln for ln in lines if ln]


def process(csv_path: Path, dry_run: bool) -> None:
    fieldnames, rows = _load_csv(csv_path)

    missing = set(QUESTION_BANK_COLUMNS) - set(fieldnames)
    if missing:
        sys.exit(f"ERROR: question_bank CSV missing columns: {sorted(missing)}")

    # Build lookup: problem_id → canonical row
    canonical_lookup: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if row.get("variant_type", "").strip().lower() == "canonical":
            pid = row.get("problem_id", "").strip()
            if pid:
                canonical_lookup[pid] = row

    filled = 0
    invalid = 0
    skipped = 0

    for row in rows:
        pid = row.get("problem_id", "").strip()
        vtype = row.get("variant_type", "").strip()
        subtype = row.get("problem_subtype", "").strip().lower()
        answer = row.get("correct_answer", "").strip()

        # Target: W5 blocksworld with missing answer
        if vtype != "W5":
            continue
        if subtype != "blocksworld":
            skipped += 1
            continue
        # Removed skip if answer check so it can override inherited canonical ones.

        # --- Locate canonical row ---
        canonical = canonical_lookup.get(pid)
        if canonical is None:
            print(f"[WARN]  {pid} W5 — no canonical row found; skipping")
            skipped += 1
            continue

        canonical_answer = canonical.get("correct_answer", "").strip()
        if not canonical_answer:
            print(f"[WARN]  {pid} W5 — canonical row has empty correct_answer; skipping")
            skipped += 1
            continue

        canonical_moves = _parse_moves(canonical_answer)
        if not canonical_moves:
            print(f"[WARN]  {pid} W5 — canonical answer has no parseable moves; skipping")
            skipped += 1
            continue

        # --- Build reversed plan ---
        try:
            reversed_plan = _make_reversed_plan(canonical_moves)
        except ValueError as exc:
            print(f"[WARN]  {pid} W5 — could not invert plan: {exc}")
            invalid += 1
            continue

        # --- Parse initial state from W5 problem_text ---
        problem_text = row.get("problem_text", "").strip()
        try:
            initial_state = _parse_w5_initial_state(problem_text)
        except ValueError as exc:
            print(f"[WARN]  {pid} W5 — could not parse initial state: {exc}")
            invalid += 1
            continue

        # --- Verify reversed plan ---
        ok, bad_idx, err_msg = _verify_plan(initial_state, reversed_plan)
        if not ok:
            step_num = (bad_idx or 0) + 1
            bad_move = reversed_plan[bad_idx] if bad_idx is not None else "?"
            print(
                f"[WARN]  {pid} W5 — illegal move at step {step_num} "
                f"({bad_move!r}): {err_msg}"
            )
            invalid += 1
            continue

        # --- All good: fill the answer ---
        plan_str = "\n".join(reversed_plan)
        if dry_run:
            print(f"[DRY]   {pid} W5 — would write {len(reversed_plan)}-step plan:")
            for i, m in enumerate(reversed_plan, 1):
                print(f"          {i}. {m}")
        else:
            row["correct_answer"] = plan_str
            print(f"[FILL]  {pid} W5 — wrote {len(reversed_plan)}-step reversed plan")
        filled += 1

    # --- Persist ---
    if not dry_run:
        _save_csv(csv_path, fieldnames, rows)
        print(f"\nUpdated: {csv_path}")

    label = "Would fill" if dry_run else "Filled"
    print(f"\nSummary: {label.lower()}={filled}, invalid={invalid}, skipped={skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate correct_answer for W5 blocksworld variants by reversing canonical plans.",
    )
    parser.add_argument(
        "--csv",
        default=QUESTION_BANK_PATH,
        help=f"Path to question bank CSV (default: {QUESTION_BANK_PATH})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without modifying the CSV.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"ERROR: CSV not found: {csv_path}")

    process(csv_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
