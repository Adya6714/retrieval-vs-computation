#!/usr/bin/env python3
"""Rewrite Blocksworld gold plans that fail the state-machine verifier.

Some bank rows (notably BW_002 and several W3 variants) store a plan that
stacks from the top of the goal tower and is therefore unexecutable. This
script synthesizes an unstack-to-table then stack-from-bottom plan, keeping
the original verb vocabulary, and logs every row it changes.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.contamination.verify import (  # noqa: E402
    _apply_blocksworld_action,
    _extract_blocksworld_actions_line_based,
    _parse_blocksworld_state,
    parse_action_mapping_from_notes,
    verify_answer,
)

BANK_PATH = REPO_ROOT / "data/problems/question_bank_bw.csv"


def _stack_from_bottom(goal_ons: list[tuple[str, str]]) -> list[str] | None:
    remaining = dict(goal_ons)
    plan: list[str] = []
    while remaining:
        progress = False
        for x, y in list(remaining.items()):
            if x not in remaining or remaining.get(x) != y:
                continue
            if y in remaining:
                continue
            plan.append(f"pick-up {x}")
            plan.append(f"stack {x} {y}")
            del remaining[x]
            progress = True
        if not progress:
            return None
    return plan


def _synthesize_plan(state: set[tuple], goal: set[tuple]) -> list[str] | None:
    st = set(state)
    plan: list[str] = []
    changed = True
    while changed:
        changed = False
        ons = [f for f in list(st) if f[0] == "on"]
        for _pred, x, y in ons:
            action = f"unstack {x} {y}"
            trial = set(st)
            if _apply_blocksworld_action(trial, action):
                st = trial
                plan.append(action)
                if _apply_blocksworld_action(st, f"put-down {x}"):
                    plan.append(f"put-down {x}")
                changed = True
                break
    goal_ons = [(f[1], f[2]) for f in goal if f[0] == "on"]
    rebuild = _stack_from_bottom(goal_ons)
    if rebuild is None:
        return None
    for action in rebuild:
        if not _apply_blocksworld_action(st, action):
            return None
        plan.append(action)
    on_goals = {f for f in goal if f[0] == "on"}
    if not on_goals.issubset(st):
        return None
    return plan


def _restore_case(text: str, existing_gold: str, problem_text: str) -> str:
    casemap: dict[str, str] = {}
    for source in (existing_gold, problem_text):
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_-]*", source):
            casemap.setdefault(tok.lower(), tok)
    out_lines = []
    for line in text.splitlines():
        out_lines.append(" ".join(casemap.get(w.lower(), w) for w in line.split()))
    return "\n".join(out_lines)


def _format_plan(actions: list[str], existing_gold: str, mapping: dict[str, str] | None) -> str:
    sample = existing_gold.strip().splitlines()[0].strip().lower() if existing_gold.strip() else ""
    first = sample.split()[0] if sample else ""
    if mapping:
        out: list[str] = []
        for action in actions:
            parts = action.split()
            verb = mapping.get(parts[0], parts[0])
            if verb == "place" and parts[0] == "stack" and " under " in existing_gold:
                out.append(f"place {parts[1]} under {parts[2]}")
            elif verb == "remove" and parts[0] == "unstack" and " from " in existing_gold:
                out.append(f"remove {parts[1]} from {parts[2]}")
            else:
                out.append(" ".join([verb, *parts[1:]]))
        return "\n".join(out)
    if first == "select" or ("place " in existing_gold and " under " in existing_gold):
        out = []
        for action in actions:
            parts = action.split()
            if parts[0] == "pick-up":
                out.append(f"select {parts[1]}")
            elif parts[0] == "put-down":
                out.append(f"release {parts[1]}")
            elif parts[0] == "stack":
                out.append(f"place {parts[1]} under {parts[2]}")
            elif parts[0] == "unstack":
                out.append(f"remove {parts[1]} from {parts[2]}")
            else:
                out.append(action)
        return "\n".join(out)
    return "\n".join(actions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", default=str(BANK_PATH))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    path = Path(args.bank)

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]

    changed: list[tuple[str, str, str, str]] = []
    for row in rows:
        if str(row.get("problem_subtype", "")).strip() != "blocksworld":
            continue
        mapping = parse_action_mapping_from_notes(row.get("notes"))
        ok = verify_answer(
            row["problem_id"],
            row["correct_answer"],
            row["correct_answer"],
            "blocksworld",
            problem_text=row["problem_text"],
            action_mapping=mapping,
        )
        if ok is True:
            continue
        parsed = _parse_blocksworld_state(row["problem_text"])
        if not parsed:
            print(f"SKIP unparsed {row['problem_id']} {row['variant_type']}")
            continue
        state, goal = parsed
        synthesized = _synthesize_plan(state, goal)
        if not synthesized:
            print(f"SKIP no-plan {row['problem_id']} {row['variant_type']}")
            continue
        new_gold = _restore_case(
            _format_plan(synthesized, row["correct_answer"], mapping),
            row["correct_answer"],
            row["problem_text"],
        )
        check = verify_answer(
            row["problem_id"],
            new_gold,
            new_gold,
            "blocksworld",
            problem_text=row["problem_text"],
            action_mapping=mapping,
        )
        if check is not True:
            print(
                f"SKIP synthesized-still-fails {row['problem_id']} {row['variant_type']} "
                f"{synthesized}"
            )
            continue
        changed.append(
            (row["problem_id"], row["variant_type"], row["correct_answer"], new_gold)
        )
        row["correct_answer"] = new_gold

    print(f"Would change {len(changed)} rows:" if args.dry_run else f"Changed {len(changed)} rows:")
    for pid, vt, old, new in changed:
        print(f"  {pid} {vt}")
        print(f"    old: {old[:80]!r}")
        print(f"    new: {new[:80]!r}")

    if not args.dry_run:
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
