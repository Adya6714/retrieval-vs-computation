#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import networkx as nx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.algo_solvers import (  # noqa: E402
    coin_change_dp,
    coin_change_greedy,
    dijkstra_shortest_path,
    nearest_neighbor_greedy,
    wis_edf_greedy,
    wis_interval_dp,
)
from scripts.generation.utils.algo_solvers import bellman_ford_sp  # noqa: E402
from scripts.generation.utils.duplicate_detector import DuplicateDetector  # noqa: E402


QUESTION_BANK = REPO_ROOT / "data/problems/question_bank_algo.csv"
OUT_CSV = REPO_ROOT / "data/staging/algo_canonical.csv"
SP_STANDARD_TOTAL = 20
WIS_ADVERSARIAL_TOTAL = 15

TARGETS: dict[str, dict[str, int]] = {
    "cc": {"textbook": 8, "novel": 7},
    "sp": {"standard": 15, "adversarial": 12},
    "wis": {"standard": 7, "adversarial": 8},
}


def _seed_for_instance(problem_id: str) -> int:
    digest = hashlib.sha256(problem_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _difficulty_from_numeric(score: int) -> str:
    if score <= 6:
        return "easy"
    if score <= 11:
        return "medium"
    return "hard"


def _next_ids(df: pd.DataFrame) -> dict[str, int]:
    out = {"CC": 0, "SP": 0, "WIS": 0}
    for prefix in out:
        subset = df["problem_id"].astype(str).str.extract(rf"^{prefix}_(\d+)$", expand=False).dropna()
        out[prefix] = int(subset.astype(int).max()) if not subset.empty else 0
    return out


def _format_cc_answer(count: int, coins: list[int]) -> str:
    return f"Count: {count}\nCoins: [{', '.join(str(c) for c in coins)}]"


def _format_sp_answer(path: list[int], cost: int) -> str:
    return f"Path: {' -> '.join(str(x) for x in path)}, Cost: {cost}"


def _format_wis_answer(selected: list[int], total: int) -> str:
    return f"Selected: {{{', '.join(str(x) for x in selected)}}}, Total: {total}"


def _cc_problem_text(denoms: list[int], target: int) -> str:
    return (
        f"You have coins with the following denominations: [{', '.join(str(x) for x in denoms)}]. "
        f"You need to make exact change for {target} using the minimum number of coins. "
        "You may use each denomination any number of times.\n"
        "What is the minimum number of coins needed and which coins do you use?\n"
        "Respond in this exact format:\n"
        "Count: [integer]\n"
        "Coins: [denomination1, denomination2, ...]\n"
        "No explanation. No other text."
    )


def _sp_problem_text(num_nodes: int, edges: list[tuple[int, int, int]], source: int, target: int) -> str:
    lines = [f"- Node {u} to Node {v}: {w}" for u, v, w in edges]
    return (
        f"A route planner has {num_nodes} locations labeled 0 to {num_nodes - 1}. "
        "Roads are one-way and weighted by travel time.\n\n"
        "The road network is:\n"
        f"{chr(10).join(lines)}\n\n"
        f"What is the shortest path from node {source} to node {target} and its total cost?\n\n"
        "Reply with only the path and cost.\n"
        "Format: Path: X -> X -> X, Cost: X"
    )


def _wis_problem_text(intervals: list[dict[str, int]]) -> str:
    lines = [
        f"Interval {it['id']}: start={it['start']}, end={it['end']}, weight={it['weight']}"
        for it in intervals
    ]
    return (
        "You are given weighted intervals on a timeline. Select a non-overlapping subset "
        "with maximum total weight.\n\n"
        "Intervals:\n"
        f"{chr(10).join(lines)}\n\n"
        "Two intervals conflict if their ranges overlap.\n"
        "Reply with only the selected interval ids and the total weight.\n"
        "Format: Selected: {X, X, X, ...}, Total: X"
    )


def _is_multiple_free(values: list[int]) -> bool:
    sorted_vals = sorted(values)
    for i, a in enumerate(sorted_vals):
        for b in sorted_vals[i + 1 :]:
            if b % a == 0:
                return False
    return True


def _generate_cc_standard(problem_id: str, denomination_type: str) -> dict:
    if denomination_type not in {"textbook", "novel"}:
        raise ValueError(f"Unsupported denomination_type={denomination_type!r}")
    rng = random.Random(_seed_for_instance(problem_id))
    textbook_sets = [
        [1, 5, 10, 25],
        [1, 2, 5, 10],
        [1, 10, 50, 100],
        [1, 2, 5, 10, 25],
    ]
    prime_pool = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

    while True:
        target = rng.randint(20, 150)
        if denomination_type == "textbook":
            denoms = sorted(rng.choice(textbook_sets))
        else:
            k_extra = rng.randint(2, 3)
            sampled = sorted(rng.sample(prime_pool, k_extra))
            if not _is_multiple_free(sampled):
                continue
            denoms = [1] + sampled

        dp = coin_change_dp(denoms, target)
        greedy = coin_change_greedy(denoms, target)
        if dp is None or greedy is None:
            continue
        dp_count, dp_coins = dp
        gr_count, gr_coins = greedy
        if gr_count != dp_count:
            continue
        diff_num = min(15, 3 + len(denoms) + target // 30)
        return {
            "problem_text": _cc_problem_text(denoms, target),
            "correct_answer": _format_cc_answer(dp_count, dp_coins),
            "difficulty": _difficulty_from_numeric(diff_num),
            "difficulty_params": {
                "subtype": "CC",
                "denominations": denoms,
                "target": target,
                "greedy_succeeds": True,
                "instance_type": "standard",
                "denomination_type": denomination_type,
                "greedy_answer": _format_cc_answer(gr_count, gr_coins),
                "difficulty_numeric": diff_num,
                "critical_step_index": -1,
            },
        }


def _nx_to_edge_list(g: nx.DiGraph) -> list[tuple[int, int, int]]:
    return sorted((int(u), int(v), int(data["weight"])) for u, v, data in g.edges(data=True))


def _split_positive_sum(rng: random.Random, total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    if total < parts:
        raise ValueError(f"cannot split total={total} into {parts} positive parts")
    weights = [1] * parts
    for _ in range(total - parts):
        weights[rng.randrange(parts)] += 1
    rng.shuffle(weights)
    return weights


def _count_sp_standard(*dfs: pd.DataFrame) -> int:
    count = 0
    for df in dfs:
        if df is None or df.empty:
            continue
        subset = df[
            (df["variant_type"] == "canonical") & (df["problem_subtype"] == "shortest_path")
        ]
        for _, row in subset.iterrows():
            try:
                params = json.loads(str(row["difficulty_params"]))
            except json.JSONDecodeError:
                continue
            if str(params.get("instance_type", "")).strip().lower() == "standard":
                count += 1
    return count


def _generate_sp_standard(problem_id: str) -> dict:
    rng = random.Random(_seed_for_instance(problem_id))
    source = 0
    n_hops = rng.randint(3, 5)
    n = rng.randint(max(6, n_hops + 3), 9)  # room for trap + dead_end off-path nodes
    target = n - 1
    optimal_cost = rng.randint(8, 25)
    path_weights = _split_positive_sum(rng, optimal_cost, n_hops)

    internal = sorted(rng.sample(range(1, target), n_hops - 1))
    path_nodes = [source, *internal, target]

    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for u, v, w in zip(path_nodes[:-1], path_nodes[1:], path_weights):
        g.add_edge(u, v, weight=w)

    off_path = [node for node in range(n) if node not in path_nodes]
    trap_node = off_path[0]
    dead_end = off_path[1]
    filler_nodes = off_path[2:]

    min_path_edge = min(path_weights)
    trap_weight = max(0, min_path_edge - 1)
    g.add_edge(source, trap_node, weight=trap_weight)
    g.add_edge(trap_node, dead_end, weight=20)
    g.add_edge(dead_end, target, weight=optimal_cost + 15)

    def _dijkstra_preserved() -> tuple[list[int], int] | None:
        solved = dijkstra_shortest_path(n, _nx_to_edge_list(g), source, target)
        if solved is None:
            return None
        path, cost = solved
        if (
            int(cost) != optimal_cost
            or path[0] != source
            or path[-1] != target
            or len(path) - 1 < 3
        ):
            return None
        return path, int(cost)

    preserved = _dijkstra_preserved()
    if preserved is None:
        raise RuntimeError(f"SP standard construction failed for {problem_id}")

    path_mid_nodes = [node for node in path_nodes if node not in {source, target}]
    for filler in filler_nodes:
        rng.shuffle(path_mid_nodes)
        attached = False
        for anchor in path_mid_nodes:
            if g.has_edge(anchor, filler):
                continue
            g.add_edge(anchor, filler, weight=rng.randint(optimal_cost + 2, optimal_cost + 8))
            if _dijkstra_preserved() is not None:
                attached = True
                break
            g.remove_edge(anchor, filler)
        if not attached:
            g.add_edge(path_mid_nodes[0], filler, weight=optimal_cost + 5)

    edges = _nx_to_edge_list(g)
    dijkstra = dijkstra_shortest_path(n, edges, source, target)
    if dijkstra is None:
        raise RuntimeError(f"SP standard unsolvable for {problem_id}")
    dijkstra_path, dijkstra_cost = dijkstra

    if g.has_edge(source, target):
        raise RuntimeError(f"SP standard has forbidden direct edge for {problem_id}")
    if len(dijkstra_path) - 1 < 3:
        raise RuntimeError(f"SP standard path too short for {problem_id}")

    greedy = nearest_neighbor_greedy(n, edges, source, target)
    if greedy is None or len(greedy[0]) < 2:
        raise RuntimeError(f"SP standard greedy failed for {problem_id}")
    if greedy[0][1] != trap_node:
        raise RuntimeError(f"SP standard greedy first move is not trap for {problem_id}")
    if greedy[1] <= dijkstra_cost:
        raise RuntimeError(f"SP standard greedy cost not worse than optimal for {problem_id}")

    diff_num = min(15, 4 + n + len(edges) // 5)
    return {
        "problem_text": _sp_problem_text(n, edges, source, target),
        "correct_answer": _format_sp_answer(dijkstra_path, dijkstra_cost),
        "difficulty": _difficulty_from_numeric(diff_num),
        "difficulty_params": {
            "subtype": "SP",
            "directed": True,
            "source": source,
            "target": target,
            "graph": [{"u": u, "v": v, "w": w} for u, v, w in edges],
            "num_vertices": n,
            "instance_type": "standard",
            "greedy_answer": _format_sp_answer(greedy[0], greedy[1]),
            "greedy_succeeds": False,
            "tempting_wrong_first_move": int(trap_node),
            "difficulty_numeric": diff_num,
            "critical_step_index": -1,
        },
    }


def _generate_sp_adversarial_trap(rng: random.Random) -> dict:
    n = rng.randint(6, 9)
    source, target = 0, n - 1
    trap_1 = 1
    trap_2 = 2
    bypass_1 = 3
    bypass_2 = 4
    edges: list[tuple[int, int, int]] = [
        (source, trap_1, 1),
        (trap_1, trap_2, 1),
        (trap_2, target, 24),
        (source, bypass_1, 4),
        (bypass_1, bypass_2, 4),
        (bypass_2, target, 4),
    ]
    # Fill optional extra nodes/edges without breaking trap behavior.
    for node in range(5, n - 1):
        edges.append((source, node, rng.randint(8, 12)))
        edges.append((node, target, rng.randint(8, 12)))
    edges.extend(
        [
            (trap_1, bypass_2, 14),
            (trap_2, bypass_2, 12),
            (bypass_1, trap_2, 11),
        ]
    )
    edges = sorted(set(edges))

    dijkstra = dijkstra_shortest_path(n, edges, source, target)
    greedy = nearest_neighbor_greedy(n, edges, source, target)
    if dijkstra is None or greedy is None or greedy[1] <= dijkstra[1]:
        raise RuntimeError("SP trap adversarial template generation failed")

    diff_num = min(15, 6 + n + (greedy[1] - dijkstra[1]) // 3)
    return {
        "problem_text": _sp_problem_text(n, edges, source, target),
        "correct_answer": _format_sp_answer(dijkstra[0], dijkstra[1]),
        "difficulty": _difficulty_from_numeric(diff_num),
        "difficulty_params": {
            "subtype": "SP",
            "directed": True,
            "source": source,
            "target": target,
            "graph": [{"u": u, "v": v, "w": w} for u, v, w in edges],
            "num_vertices": n,
            "instance_type": "adversarial",
            "greedy_answer": _format_sp_answer(greedy[0], greedy[1]),
            "greedy_succeeds": False,
            "adversarial_pattern": "trap",
            "difficulty_numeric": diff_num,
            "critical_step_index": 0,
        },
    }


def _generate_sp_adversarial_long_cheap(rng: random.Random) -> dict:
    n = rng.randint(6, 9)
    source, target = 0, n - 1
    mid = 1
    a = 2
    b = 3
    edges: list[tuple[int, int, int]] = [
        (source, mid, 3),
        (mid, target, 3),
        (source, a, 3),
        (a, b, 1),
        (b, target, 1),
    ]
    for node in range(4, n - 1):
        edges.append((source, node, rng.randint(7, 11)))
        edges.append((node, target, rng.randint(7, 11)))
    edges.extend(
        [
            (a, target, 8),
            (mid, b, 6),
            (source, b, 9),
        ]
    )
    edges = sorted(set(edges))

    dijkstra = dijkstra_shortest_path(n, edges, source, target)
    greedy = nearest_neighbor_greedy(n, edges, source, target)
    expected_greedy = [source, mid, target]
    expected_opt = [source, a, b, target]
    if (
        dijkstra is None
        or greedy is None
        or dijkstra[0] != expected_opt
        or dijkstra[1] != 5
        or greedy[0] != expected_greedy
        or greedy[1] != 6
    ):
        # Keep this deterministic and fail loud if pattern is broken.
        raise RuntimeError("SP long_cheap adversarial template generation failed")

    diff_num = min(15, 8 + n)
    return {
        "problem_text": _sp_problem_text(n, edges, source, target),
        "correct_answer": _format_sp_answer(dijkstra[0], dijkstra[1]),
        "difficulty": _difficulty_from_numeric(diff_num),
        "difficulty_params": {
            "subtype": "SP",
            "directed": True,
            "source": source,
            "target": target,
            "graph": [{"u": u, "v": v, "w": w} for u, v, w in edges],
            "num_vertices": n,
            "instance_type": "adversarial",
            "greedy_answer": _format_sp_answer(greedy[0], greedy[1]),
            "greedy_succeeds": False,
            "adversarial_pattern": "long_cheap",
            "difficulty_numeric": diff_num,
            "critical_step_index": 0,
        },
    }


def _generate_sp_adversarial_negative_weight(rng: random.Random) -> dict:
    n = rng.randint(6, 9)
    source, target = 0, n - 1
    x = 1
    y = 2
    neg_w = -3
    edges: list[tuple[int, int, int]] = [
        (source, x, 3),
        (source, y, 5),
        (y, x, neg_w),
        (x, target, 1),
        (y, target, 8),
    ]
    for node in range(3, n - 1):
        edges.append((source, node, rng.randint(6, 10)))
        edges.append((node, target, rng.randint(6, 10)))
    edges = sorted(set(edges))

    g = nx.DiGraph()
    g.add_nodes_from(range(n))
    for u, v, w in edges:
        g.add_edge(u, v, weight=w)

    dijkstra = dijkstra_shortest_path(n, edges, source, target)
    bellman_path, bellman_cost = bellman_ford_sp(g, source, target)
    greedy = nearest_neighbor_greedy(n, edges, source, target)
    if (
        dijkstra is None
        or greedy is None
        or not bellman_path
        or bellman_cost == float("inf")
        or dijkstra[1] <= bellman_cost
        or (y, x) not in list(zip(bellman_path[:-1], bellman_path[1:]))
    ):
        raise RuntimeError("SP negative_weight adversarial template generation failed")

    diff_num = min(15, 9 + n + int(dijkstra[1] - bellman_cost))
    return {
        "problem_text": _sp_problem_text(n, edges, source, target),
        "correct_answer": _format_sp_answer([int(xn) for xn in bellman_path], int(bellman_cost)),
        "difficulty": _difficulty_from_numeric(diff_num),
        "difficulty_params": {
            "subtype": "SP",
            "directed": True,
            "source": source,
            "target": target,
            "graph": [{"u": u, "v": v, "w": w} for u, v, w in edges],
            "num_vertices": n,
            "instance_type": "adversarial",
            "greedy_answer": _format_sp_answer(greedy[0], greedy[1]),
            "greedy_succeeds": False,
            "adversarial_pattern": "negative_weight",
            "requires_bellman_ford": True,
            "difficulty_numeric": diff_num,
            "critical_step_index": 0,
        },
    }


def _generate_wis_standard(problem_id: str) -> dict:
    rng = random.Random(_seed_for_instance(problem_id))
    while True:
        n = rng.randint(8, 12)
        intervals: list[dict[str, int]] = []
        for i in range(n):
            start = rng.randint(0, 20)
            duration = rng.randint(2, 6)
            intervals.append(
                {"id": i, "start": start, "end": start + duration, "weight": rng.randint(5, 20)}
            )

        conflicts = 0
        for i in range(n):
            for j in range(i + 1, n):
                a = intervals[i]
                b = intervals[j]
                if int(a["end"]) > int(b["start"]) and int(b["end"]) > int(a["start"]):
                    conflicts += 1
        if conflicts < 4:
            continue

        greedy_ids, greedy_total = wis_edf_greedy(intervals)
        opt_ids, opt_total = wis_interval_dp(intervals)
        if greedy_total != opt_total:
            continue
        if len(opt_ids) >= n - 1:
            continue
        excluded = [it for it in intervals if int(it["id"]) not in set(opt_ids)]
        if not any(int(it["weight"]) > 5 for it in excluded):
            continue

        diff_num = min(15, 4 + n + opt_total // 25)
        return {
            "problem_text": _wis_problem_text(intervals),
            "correct_answer": _format_wis_answer(opt_ids, opt_total),
            "difficulty": _difficulty_from_numeric(diff_num),
            "difficulty_params": {
                "subtype": "WIS",
                "intervals": intervals,
                "greedy_succeeds": True,
                "instance_type": "standard",
                "greedy_answer": _format_wis_answer(greedy_ids, greedy_total),
                "difficulty_numeric": diff_num,
                "critical_step_index": -1,
            },
        }


def _generate_wis_adversarial_anchor(rng: random.Random) -> dict:
    light_count = rng.choice([3, 4])
    intervals: list[dict[str, int]] = []
    cur = 0
    for i in range(light_count):
        intervals.append({"id": i, "start": cur, "end": cur + 2, "weight": rng.randint(7, 9)})
        cur += 2
    # Heavy anchor overlaps all light intervals.
    anchor_id = light_count
    anchor_weight = sum(int(it["weight"]) for it in intervals) + rng.randint(10, 14)
    intervals.append({"id": anchor_id, "start": 1, "end": cur + 1, "weight": anchor_weight})
    # Optional tail interval compatible with either strategy.
    intervals.append({"id": anchor_id + 1, "start": cur + 1, "end": cur + 3, "weight": rng.randint(5, 8)})

    greedy_ids, greedy_total = wis_edf_greedy(intervals)
    opt_ids, opt_total = wis_interval_dp(intervals)
    if opt_total - greedy_total < 10:
        raise RuntimeError("WIS anchor_conflict generation failed: gap < 10")

    diff_num = min(15, 8 + (opt_total - greedy_total) // 2 + len(intervals))
    return {
        "problem_text": _wis_problem_text(intervals),
        "correct_answer": _format_wis_answer(opt_ids, opt_total),
        "difficulty": _difficulty_from_numeric(diff_num),
        "difficulty_params": {
            "subtype": "WIS",
            "intervals": intervals,
            "greedy_succeeds": False,
            "instance_type": "adversarial",
            "greedy_answer": _format_wis_answer(greedy_ids, greedy_total),
            "adversarial_pattern": "anchor_conflict",
            "difficulty_numeric": diff_num,
            "critical_step_index": 0,
        },
    }


def _generate_wis_adversarial_chain_trap(rng: random.Random) -> dict:
    chain_weight = rng.randint(11, 13)
    h1 = rng.randint(27, 30)
    h2 = rng.randint(27, 30)
    intervals = [
        {"id": 0, "start": 0, "end": 2, "weight": chain_weight},
        {"id": 1, "start": 2, "end": 4, "weight": chain_weight},
        {"id": 2, "start": 4, "end": 6, "weight": chain_weight},
        {"id": 3, "start": 6, "end": 8, "weight": chain_weight},
        # Two heavy intervals, each conflicting with ~2 chain nodes, non-overlapping with each other.
        {"id": 4, "start": 1, "end": 5, "weight": h1},
        {"id": 5, "start": 5, "end": 9, "weight": h2},
    ]
    greedy_ids, greedy_total = wis_edf_greedy(intervals)
    opt_ids, opt_total = wis_interval_dp(intervals)
    if opt_total <= greedy_total:
        raise RuntimeError("WIS chain_trap generation failed: dp not better than edf")

    diff_num = min(15, 8 + (opt_total - greedy_total) // 2 + len(intervals))
    return {
        "problem_text": _wis_problem_text(intervals),
        "correct_answer": _format_wis_answer(opt_ids, opt_total),
        "difficulty": _difficulty_from_numeric(diff_num),
        "difficulty_params": {
            "subtype": "WIS",
            "intervals": intervals,
            "greedy_succeeds": False,
            "instance_type": "adversarial",
            "greedy_answer": _format_wis_answer(greedy_ids, greedy_total),
            "adversarial_pattern": "chain_trap",
            "difficulty_numeric": diff_num,
            "critical_step_index": 0,
        },
    }


def _parse_wis_selected_count(correct_answer: str) -> int | None:
    m = re.search(r"Selected:\s*\{([^}]*)\}", str(correct_answer), flags=re.IGNORECASE)
    if not m:
        return None
    return len([x for x in m.group(1).split(",") if x.strip()])


def _is_trivial_wis_standard_row(row: dict[str, str]) -> bool:
    if str(row.get("problem_subtype", "")).strip() != "wis":
        return False
    if str(row.get("variant_type", "")).strip() != "canonical":
        return False
    try:
        params = json.loads(str(row.get("difficulty_params", "")))
    except json.JSONDecodeError:
        return False
    if str(params.get("instance_type", "")).strip().lower() != "standard":
        return False
    intervals = params.get("intervals")
    if not isinstance(intervals, list):
        return False
    selected_count = _parse_wis_selected_count(str(row.get("correct_answer", "")))
    return selected_count is not None and selected_count == len(intervals)


def _count_wis_adversarial(*dfs: pd.DataFrame) -> int:
    total = 0
    for df in dfs:
        if df is None or df.empty:
            continue
        subset = df[
            (df["variant_type"] == "canonical") & (df["problem_subtype"] == "wis")
        ]
        for _, row in subset.iterrows():
            try:
                params = json.loads(str(row["difficulty_params"]))
            except json.JSONDecodeError:
                continue
            if str(params.get("instance_type", "")).strip().lower() == "adversarial":
                total += 1
    return total


def _count_wis_adversarial_by_pattern(df: pd.DataFrame) -> dict[str, int]:
    counts = {"anchor_conflict": 0, "chain_trap": 0}
    if df is None or df.empty:
        return counts
    subset = df[(df["variant_type"] == "canonical") & (df["problem_subtype"] == "wis")]
    for _, row in subset.iterrows():
        try:
            params = json.loads(str(row["difficulty_params"]))
        except json.JSONDecodeError:
            continue
        if str(params.get("instance_type", "")).strip().lower() != "adversarial":
            continue
        pattern = str(params.get("adversarial_pattern", "")).strip().lower()
        if pattern in counts:
            counts[pattern] += 1
    return counts


def _merge_sp_staging_rows(new_rows: list[dict], bank_columns: list[str]) -> pd.DataFrame:
    """Append newly generated SP rows to existing staging content."""
    if OUT_CSV.exists():
        existing = pd.read_csv(OUT_CSV, dtype=str).fillna("")
    else:
        existing = pd.DataFrame(columns=bank_columns)
    out_df = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
    return out_df.reindex(columns=bank_columns, fill_value="")


def _row(problem_id: str, family_subtype: str, verifier: str, contamination: str, payload: dict) -> dict:
    return {
        "problem_id": problem_id,
        "variant_type": "canonical",
        "problem_text": payload["problem_text"],
        "correct_answer": payload["correct_answer"],
        "problem_family": "Algorithmic Suit",
        "problem_subtype": family_subtype,
        "difficulty": payload["difficulty"],
        "contamination_pole": contamination,
        "source": f"procedural_seed_{_seed_for_instance(problem_id)}",
        "verifier_function": verifier,
        "difficulty_params": json.dumps(payload["difficulty_params"], separators=(",", ":")),
        "notes": "stage1_procedural_generation",
    }


def _run_fix_trivial() -> None:
    if not OUT_CSV.exists():
        raise FileNotFoundError(f"Missing staging file: {OUT_CSV}")
    if not QUESTION_BANK.exists():
        raise FileNotFoundError(f"Missing source bank: {QUESTION_BANK}")

    bank_df = pd.read_csv(QUESTION_BANK, dtype=str).fillna("")
    staging_df = pd.read_csv(OUT_CSV, dtype=str).fillna("")

    removed_sp022 = int((staging_df["problem_id"] == "SP_022").sum())
    staging_df = staging_df[staging_df["problem_id"] != "SP_022"].copy()

    trivial_mask = staging_df.apply(
        lambda row: _is_trivial_wis_standard_row(row.to_dict()), axis=1
    )
    removed_trivial = staging_df[trivial_mask].copy()
    staging_df = staging_df[~trivial_mask].copy()

    id_state = _next_ids(bank_df)
    for prefix in ("CC", "SP", "WIS"):
        id_state[prefix] = max(id_state[prefix], _next_ids(staging_df)[prefix])

    duplicate_detector = DuplicateDetector(REPO_ROOT)
    new_rows: list[dict] = []

    replace_needed = len(removed_trivial)
    generated_standard = 0
    guard = 0
    while generated_standard < replace_needed:
        guard += 1
        if guard > replace_needed * 300:
            raise RuntimeError("WIS standard replacement exceeded duplicate-reseed guard")
        id_state["WIS"] += 1
        pid = f"WIS_{id_state['WIS']:03d}"
        payload = _generate_wis_standard(pid)
        row = _row(pid, "wis", "verify_wis", "high", payload)
        dup, reason = duplicate_detector.is_duplicate(row)
        if dup:
            print(f"Skipping duplicate {pid}: {reason}")
            continue
        duplicate_detector.register(row)
        new_rows.append(row)
        generated_standard += 1

    staging_adv = _count_wis_adversarial(staging_df)
    adv_needed = max(0, WIS_ADVERSARIAL_TOTAL - staging_adv)
    anchor_needed = min(4, adv_needed)
    chain_needed = min(3, max(0, adv_needed - anchor_needed))
    adv_generated = {"anchor_conflict": 0, "chain_trap": 0}
    adv_guard = 0
    while adv_generated["anchor_conflict"] < anchor_needed or adv_generated["chain_trap"] < chain_needed:
        adv_guard += 1
        if adv_guard > (anchor_needed + chain_needed) * 300:
            raise RuntimeError("WIS adversarial top-up exceeded duplicate-reseed guard")
        id_state["WIS"] += 1
        pid = f"WIS_{id_state['WIS']:03d}"
        rng = random.Random(_seed_for_instance(pid))
        if adv_generated["anchor_conflict"] < anchor_needed:
            payload = _generate_wis_adversarial_anchor(rng)
            pattern = "anchor_conflict"
        else:
            payload = _generate_wis_adversarial_chain_trap(rng)
            pattern = "chain_trap"
        row = _row(pid, "wis", "verify_wis", "high", payload)
        dup, reason = duplicate_detector.is_duplicate(row)
        if dup:
            print(f"Skipping duplicate {pid}: {reason}")
            continue
        duplicate_detector.register(row)
        new_rows.append(row)
        adv_generated[pattern] += 1

    out_df = pd.concat([staging_df, pd.DataFrame(new_rows)], ignore_index=True)
    cols = list(bank_df.columns)
    out_df = out_df.reindex(columns=cols, fill_value="")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    print("Fix-trivial summary:")
    print(f"- removed SP_022 rows: {removed_sp022}")
    print(f"- removed trivial WIS standard rows: {len(removed_trivial)}")
    if not removed_trivial.empty:
        print("  trivial IDs:", ", ".join(sorted(removed_trivial["problem_id"].unique())))
    print(f"- regenerated WIS standard rows: {generated_standard}")
    print(f"- added WIS adversarial rows: anchor_conflict={adv_generated['anchor_conflict']}, "
          f"chain_trap={adv_generated['chain_trap']}")
    staging_adv_after = _count_wis_adversarial(out_df)
    pattern_counts = _count_wis_adversarial_by_pattern(out_df)
    print(
        f"- WIS adversarial in staging now: {staging_adv_after} "
        f"(target={WIS_ADVERSARIAL_TOTAL}, "
        f"anchor_conflict={pattern_counts['anchor_conflict']}, "
        f"chain_trap={pattern_counts['chain_trap']})"
    )
    print(f"- wrote {len(out_df)} rows to {OUT_CSV}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate stage1 ALGO canonical rows.")
    parser.add_argument("--fix-trivial", action="store_true", help="Repair staging WIS/SP rows")
    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--subtype", choices=["cc", "sp", "wis", "all"], default="all")
    args = parser.parse_args()

    if args.fix_trivial:
        _run_fix_trivial()
        return
    if not args.dry_run and not args.run:
        parser.error("One of --dry-run, --run, or --fix-trivial is required")

    if not QUESTION_BANK.exists():
        raise FileNotFoundError(f"Missing source bank: {QUESTION_BANK}")
    bank_df = pd.read_csv(QUESTION_BANK, dtype=str).fillna("")
    staging_df = (
        pd.read_csv(OUT_CSV, dtype=str).fillna("")
        if OUT_CSV.exists()
        else pd.DataFrame(columns=bank_df.columns)
    )
    id_state = _next_ids(bank_df)
    if not staging_df.empty:
        for prefix in ("CC", "SP", "WIS"):
            staging_max = _next_ids(staging_df)[prefix]
            id_state[prefix] = max(id_state[prefix], staging_max)

    duplicate_detector = DuplicateDetector(REPO_ROOT)
    requested = ["cc", "sp", "wis"] if args.subtype == "all" else [args.subtype]
    rows: list[dict] = []
    sp_standard_skipped_dup = 0

    for sub in requested:
        if sub == "cc":
            for _ in range(TARGETS["cc"]["textbook"]):
                id_state["CC"] += 1
                pid = f"CC_{id_state['CC']:03d}"
                payload = _generate_cc_standard(pid, "textbook")
                rows.append(_row(pid, "coin_change", "verify_coinchange", "high", payload))
            for _ in range(TARGETS["cc"]["novel"]):
                id_state["CC"] += 1
                pid = f"CC_{id_state['CC']:03d}"
                payload = _generate_cc_standard(pid, "novel")
                rows.append(_row(pid, "coin_change", "verify_coinchange", "low", payload))
        elif sub == "sp":
            sp_standard_needed = max(
                0, SP_STANDARD_TOTAL - _count_sp_standard(bank_df, staging_df)
            )
            print(
                f"SP standard target={SP_STANDARD_TOTAL}, "
                f"existing={SP_STANDARD_TOTAL - sp_standard_needed}, "
                f"to_generate={sp_standard_needed}"
            )
            generated_sp_standard = 0
            guard = 0
            while generated_sp_standard < sp_standard_needed:
                guard += 1
                if guard > sp_standard_needed * 200:
                    raise RuntimeError("SP standard generation exceeded duplicate-reseed guard")
                id_state["SP"] += 1
                pid = f"SP_{id_state['SP']:03d}"
                payload = _generate_sp_standard(pid)
                row = _row(pid, "shortest_path", "verify_sp", "high", payload)
                dup, reason = duplicate_detector.is_duplicate(row)
                if dup:
                    sp_standard_skipped_dup += 1
                    print(f"Skipping duplicate {pid}: {reason}")
                    continue
                duplicate_detector.register(row)
                rows.append(row)
                generated_sp_standard += 1
            # 12 adversarial split: 4 trap, 4 long_cheap, 4 negative_weight.
            adv_generated = 0
            adv_guard = 0
            while adv_generated < TARGETS["sp"]["adversarial"]:
                adv_guard += 1
                if adv_guard > TARGETS["sp"]["adversarial"] * 200:
                    raise RuntimeError("SP adversarial generation exceeded duplicate-reseed guard")
                id_state["SP"] += 1
                pid = f"SP_{id_state['SP']:03d}"
                rng = random.Random(_seed_for_instance(pid))
                if adv_generated < 4:
                    payload = _generate_sp_adversarial_trap(rng)
                    verifier = "verify_sp"
                elif adv_generated < 8:
                    payload = _generate_sp_adversarial_long_cheap(rng)
                    verifier = "verify_sp"
                else:
                    payload = _generate_sp_adversarial_negative_weight(rng)
                    verifier = "verify_sp"
                adv_row = _row(pid, "shortest_path", verifier, "high", payload)
                dup, reason = duplicate_detector.is_duplicate(adv_row)
                if dup:
                    print(f"Skipping duplicate {pid}: {reason}")
                    continue
                duplicate_detector.register(adv_row)
                rows.append(adv_row)
                adv_generated += 1
        elif sub == "wis":
            for _ in range(TARGETS["wis"]["standard"]):
                id_state["WIS"] += 1
                pid = f"WIS_{id_state['WIS']:03d}"
                payload = _generate_wis_standard(pid)
                rows.append(_row(pid, "wis", "veryify_WIS", "High", payload))
            # 8 adversarial split: 4 anchor_conflict, 4 chain_trap.
            for pattern_idx in range(TARGETS["wis"]["adversarial"]):
                id_state["WIS"] += 1
                pid = f"WIS_{id_state['WIS']:03d}"
                rng = random.Random(_seed_for_instance(pid))
                if pattern_idx < 4:
                    payload = _generate_wis_adversarial_anchor(rng)
                else:
                    payload = _generate_wis_adversarial_chain_trap(rng)
                rows.append(_row(pid, "wis", "veryify_WIS", "High", payload))

    print(f"Generated rows: {len(rows)}")
    print(f"By prefix: CC={sum(r['problem_id'].startswith('CC_') for r in rows)}, "
          f"SP={sum(r['problem_id'].startswith('SP_') for r in rows)}, "
          f"WIS={sum(r['problem_id'].startswith('WIS_') for r in rows)}")
    if sp_standard_skipped_dup:
        print(f"SP standard skipped as duplicates: {sp_standard_skipped_dup}")
    if rows:
        print("First IDs:", ", ".join(r["problem_id"] for r in rows[:5]))

    if args.run:
        cols = list(bank_df.columns)
        if args.subtype == "sp" and OUT_CSV.exists():
            out_df = _merge_sp_staging_rows(rows, cols)
        else:
            out_df = pd.DataFrame(rows).reindex(columns=cols, fill_value="")
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(OUT_CSV, index=False)
        print(f"Wrote {len(out_df)} rows to {OUT_CSV}")
    else:
        print("Dry-run mode: no files written.")


if __name__ == "__main__":
    main()
