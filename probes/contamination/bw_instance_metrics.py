"""Blocksworld instance metrics from natural-language prompts + Fast Downward."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

from probes.contamination.verify import (
    _infer_bw_current_derived,
    _parse_blocksworld_state,
)
from scripts.generation.utils.variant_utils import load_bw_domain, run_fast_downward


def _collect_blocks(state: set[tuple], goal: set[tuple]) -> set[str]:
    blocks: set[str] = set()
    for fact in state | goal:
        if not fact:
            continue
        if fact[0] == "on" and len(fact) == 3:
            blocks.update([fact[1], fact[2]])
        elif len(fact) >= 2 and fact[0] in {"ontable", "clear", "holding"}:
            blocks.add(fact[1])
    return blocks


def _on_pairs(facts: set[tuple]) -> list[tuple[str, str]]:
    return [(f[1], f[2]) for f in facts if f[0] == "on" and len(f) == 3]


def _ontable_blocks(facts: set[tuple]) -> set[str]:
    return {f[1] for f in facts if f[0] == "ontable" and len(f) == 2}


def tower_height(on_pairs: list[tuple[str, str]], ontable: set[str]) -> int:
    """Max stack height (blocks in the tallest tower). Flat init → 1."""
    if not on_pairs:
        return 1 if ontable else 0
    children = {c for c, _ in on_pairs}
    parents = {p for _, p in on_pairs}
    roots = (parents - children) | ontable
    if not roots:
        roots = parents
    adj: dict[str, list[str]] = defaultdict(list)
    for child, parent in on_pairs:
        adj[parent].append(child)

    def depth(node: str) -> int:
        kids = adj.get(node, [])
        if not kids:
            return 1
        return 1 + max(depth(k) for k in kids)

    return max(depth(r) for r in roots)


def nl_to_bw_pddl(problem_text: str, problem_id: str) -> tuple[str, str] | None:
    """Reconstruct domain + problem PDDL from NL via the released state parser."""
    parsed = _parse_blocksworld_state(problem_text)
    if parsed is None:
        return None
    state, goal = parsed
    state = set(state)
    goal = set(goal)
    _infer_bw_current_derived(state)
    blocks = _collect_blocks(state, goal)
    if not blocks:
        return None

    init_parts: list[str] = []
    for fact in sorted(state, key=str):
        if fact[0] == "on" and len(fact) == 3:
            init_parts.append(f"(on {fact[1]} {fact[2]})")
        elif fact[0] == "ontable" and len(fact) == 2:
            init_parts.append(f"(ontable {fact[1]})")
        elif fact[0] == "clear" and len(fact) == 2:
            init_parts.append(f"(clear {fact[1]})")
        elif fact == ("handempty",):
            init_parts.append("(handempty)")

    on_goal = _on_pairs(goal)
    goal_parts = [f"(on {a} {b})" for a, b in sorted(on_goal)]
    for block in sorted(blocks):
        if ("ontable", block) in goal:
            goal_parts.append(f"(ontable {block})")
        if ("clear", block) in goal:
            goal_parts.append(f"(clear {block})")
    if not goal_parts:
        return None

    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in problem_id)
    problem_pddl = f"""\
(define (problem {safe_id})
  (:domain blocksworld-4ops)
  (:objects {" ".join(sorted(blocks))})
  (:init
    {" ".join(init_parts)})
  (:goal (and
    {" ".join(goal_parts)}))
)
"""
    return load_bw_domain(), problem_pddl


@lru_cache(maxsize=512)
def fd_optimal_plan_length(problem_text: str, problem_id: str) -> tuple[int | None, str]:
    built = nl_to_bw_pddl(problem_text, problem_id)
    if built is None:
        return None, "pddl_build_failed"
    domain, problem = built
    plan, status = run_fast_downward(domain, problem, timeout=60)
    if not plan:
        return None, status
    n = len([ln for ln in plan.splitlines() if ln.strip()])
    return n, status


def extract_bw_metrics(problem_text: str, problem_id: str) -> dict[str, int | str | None]:
    """Structural metrics for a BW NL instance."""
    parsed = _parse_blocksworld_state(problem_text)
    if parsed is None:
        return {
            "num_blocks": None,
            "n_goal_clauses": None,
            "goal_tower_depth": None,
            "init_tower_depth": None,
            "fd_optimal_plan_length": None,
            "gold_plan_length": None,
            "fd_status": "parse_failed",
        }
    state, goal = parsed
    state = set(state)
    goal = set(goal)
    _infer_bw_current_derived(state)
    blocks = _collect_blocks(state, goal)
    on_goal = _on_pairs(goal)
    on_init = _on_pairs(state)
    n_goal_clauses = len(on_goal) + sum(
        1 for f in goal if f[0] in {"ontable", "clear"} and len(f) == 2
    )
    fd_len, fd_status = fd_optimal_plan_length(problem_text, problem_id)
    return {
        "num_blocks": len(blocks),
        "n_goal_clauses": n_goal_clauses,
        "goal_tower_depth": tower_height(on_goal, _ontable_blocks(goal)),
        "init_tower_depth": tower_height(on_init, _ontable_blocks(state)),
        "fd_optimal_plan_length": fd_len,
        "fd_status": fd_status,
    }
