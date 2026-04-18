"""
Fill correct_answer column in probe1_instances.csv using Fast Downward.
Run from rvc/ root:
  python scripts/fill_correct_answers.py \
    --csv data/problems/probe1_instances.csv \
    --planbench /path/to/LLMs-Planning \
    --downward /path/to/downward/fast-downward.py
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import tempfile
from pathlib import Path


BLOCKSWORLD_FAMILIES = {"blocksworld", "planning_suite"}
MYSTERY_FAMILIES = {"mystery_blocksworld"}


def run_fast_downward(domain_path: Path, problem_path: Path, fd_path: Path) -> list[str] | None:
    """Run Fast Downward and return list of actions, or None if unsolvable/error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [
                str(fd_path),
                "--plan-file", f"{tmpdir}/plan.txt",
                str(domain_path),
                str(problem_path),
                "--search", "astar(lmcut())",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        plan_file = Path(tmpdir) / "plan.txt"
        if not plan_file.exists():
            print(f"  No plan found. FD stdout tail: {result.stdout[-500:]}")
            return None

        lines = plan_file.read_text().splitlines()
        # FD plan files have lines like: (pick-up a)
        # Filter out comment lines starting with ;
        actions = [
            line.strip().strip("()").strip()
            for line in lines
            if line.strip() and not line.startswith(";")
        ]
        return actions


def pddl_action_to_natural(action: str) -> str:
    """Convert PDDL action string to natural language format matching our prompt."""
    # FD outputs: 'pick-up a' or 'stack a b' or 'unstack a b' or 'put-down a'
    parts = action.lower().split()
    if not parts:
        return action
    verb = parts[0]
    args = parts[1:]

    if verb == "pick-up" and len(args) == 1:
        return f"pick-up {args[0]}"
    elif verb == "put-down" and len(args) == 1:
        return f"put-down {args[0]}"
    elif verb == "stack" and len(args) == 2:
        return f"stack {args[0]} {args[1]}"
    elif verb == "unstack" and len(args) == 2:
        return f"unstack {args[0]} {args[1]}"
    # Mystery blocksworld actions
    elif verb == "attack" and len(args) == 1:
        return f"attack {args[0]}"
    elif verb == "succumb" and len(args) == 1:
        return f"succumb {args[0]}"
    elif verb == "overcome" and len(args) == 2:
        return f"overcome {args[0]} {args[1]}"
    elif verb == "feast" and len(args) == 2:
        return f"feast {args[0]} {args[1]}"
    return action


def get_pddl_paths(row: dict, planbench_root: Path) -> tuple[Path, Path] | None:
    """Extract domain + problem PDDL paths from CSV source field."""
    source = row.get("source", "")
    # source field format: "type=... | path=plan-bench/instances/blocksworld/..."
    path_match = re.search(r"path=([^\s|]+)", source)
    if not path_match:
        print(f"  Could not parse path from source: {source}")
        return None

    problem_rel = path_match.group(1).strip()
    problem_path = planbench_root / problem_rel

    if not problem_path.exists():
        print(f"  Problem file not found: {problem_path}")
        return None

    # Domain file is always in the same directory as the problem
    domain_path = problem_path.parent / "domain.pddl"
    if not domain_path.exists():
        # Try one level up
        domain_path = problem_path.parent.parent / "domain.pddl"
    if not domain_path.exists():
        print(f"  Domain file not found near: {problem_path}")
        return None

    return domain_path, problem_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/problems/probe1_instances.csv")
    parser.add_argument("--planbench", required=True, help="Path to LLMs-Planning repo root")
    parser.add_argument("--downward", required=True, help="Path to fast-downward.py")
    parser.add_argument("--dry-run", action="store_true", help="Print paths only, don't run FD")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    planbench_root = Path(args.planbench)
    fd_path = Path(args.downward)

    if not fd_path.exists():
        raise FileNotFoundError(f"fast-downward.py not found at {fd_path}")

    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    filled = 0
    skipped = 0
    failed = 0

    for row in rows:
        pid = row.get("problem_id", "").strip()
        subtype = row.get("problem_subtype", "").strip().lower()
        existing = row.get("correct_answer", "").strip()

        if existing:
            print(f"[SKIP] {pid} — already has answer")
            skipped += 1
            continue

        if subtype not in ("blocksworld", "mystery_blocksworld"):
            print(f"[SKIP] {pid} — subtype '{subtype}' not handled yet")
            skipped += 1
            continue

        paths = get_pddl_paths(row, planbench_root)
        if paths is None:
            print(f"[FAIL] {pid} — could not resolve PDDL paths")
            failed += 1
            continue

        domain_path, problem_path = paths
        print(f"[RUN]  {pid} — {problem_path.name}")

        if args.dry_run:
            print(f"       domain: {domain_path}")
            print(f"       problem: {problem_path}")
            continue

        try:
            actions = run_fast_downward(domain_path, problem_path, fd_path)
        except subprocess.TimeoutExpired:
            print(f"[FAIL] {pid} — Fast Downward timed out after 120s")
            failed += 1
            continue
        except Exception as e:
            print(f"[FAIL] {pid} — error: {e}")
            failed += 1
            continue

        if actions is None:
            print(f"[FAIL] {pid} — no plan returned")
            failed += 1
            continue

        natural_actions = [pddl_action_to_natural(a) for a in actions]
        plan_str = "\n".join(natural_actions)
        row["correct_answer"] = plan_str
        print(f"       → {len(natural_actions)} steps")
        filled += 1

    if not args.dry_run:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nDone. filled={filled}, skipped={skipped}, failed={failed}")
        print(f"Updated: {csv_path}")
    else:
        print(f"\nDry run complete. Would process {len(rows) - skipped} rows.")


if __name__ == "__main__":
    main()
