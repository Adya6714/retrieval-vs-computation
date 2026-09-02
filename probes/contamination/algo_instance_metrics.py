"""Structural difficulty metrics for ALGO canonical/W6 matched comparisons (M2/L1)."""

from __future__ import annotations

import json
import re
from typing import Any

import networkx as nx


def _parse_params(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw or "").strip()
    if not text:
        return {}
    return json.loads(text)


def _dp_coin_count(denoms: list[int], target: int) -> int | None:
    inf = 10**9
    dp = [inf] * (target + 1)
    dp[0] = 0
    for amount in range(1, target + 1):
        for coin in denoms:
            if coin <= amount and dp[amount - coin] + 1 < dp[amount]:
                dp[amount] = dp[amount - coin] + 1
    return None if dp[target] >= inf else int(dp[target])


def _sp_optimal_cost(params: dict[str, Any], verifier: str) -> int | None:
    graph_edges = params.get("graph") or []
    if not graph_edges:
        return None
    directed = bool(params.get("directed", False))
    g = nx.DiGraph() if directed else nx.Graph()
    for edge in graph_edges:
        g.add_edge(int(edge["u"]), int(edge["v"]), weight=int(edge["w"]))
    src = int(params.get("source", 0))
    tgt = int(params.get("target", max(g.nodes) if g.nodes else 0))
    use_bf = "bellman" in str(verifier or "").lower()
    try:
        if use_bf:
            return int(nx.bellman_ford_path_length(g, src, tgt, weight="weight"))
        return int(nx.dijkstra_path_length(g, src, tgt, weight="weight"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def _wis_optimal_weight(params: dict[str, Any]) -> int | None:
    intervals = params.get("intervals") or []
    if not isinstance(intervals, list) or not intervals:
        return None
    ordered = sorted(intervals, key=lambda it: (int(it["end"]), int(it["start"])))
    n = len(ordered)
    p = [-1] * n
    for i in range(n):
        for j in range(i - 1, -1, -1):
            if int(ordered[j]["end"]) <= int(ordered[i]["start"]):
                p[i] = j
                break
    dp = [0] * (n + 1)
    take = [False] * n
    for i in range(1, n + 1):
        w = int(ordered[i - 1]["weight"])
        incl = w + dp[p[i - 1] + 1]
        excl = dp[i - 1]
        if incl >= excl:
            dp[i] = incl
            take[i - 1] = True
        else:
            dp[i] = excl
    return int(dp[n])


def extract_algo_metrics(
    difficulty_params: str | dict[str, Any],
    *,
    problem_subtype: str = "",
    verifier_function: str = "",
) -> dict[str, Any]:
    """
    True optimal / structural difficulty for ALGO instances.

    CC: target, denomination count, optimal coin count.
    SP: graph size, optimal shortest-path cost.
    WIS: interval count, optimal total weight.
    """
    params = _parse_params(difficulty_params)
    subtype = str(params.get("subtype") or problem_subtype or "").upper()
    out: dict[str, Any] = {"subtype": subtype}

    if subtype == "CC" or str(problem_subtype).lower() == "coin_change":
        denoms = [int(x) for x in params.get("denominations") or []]
        target = int(params.get("target", 0))
        out["n_denominations"] = len(denoms)
        out["target"] = target
        out["optimal_coin_count"] = _dp_coin_count(denoms, target)
        return out

    if subtype == "SP" or "shortest" in str(problem_subtype).lower():
        graph_edges = params.get("graph") or []
        nodes = set()
        for edge in graph_edges:
            nodes.add(int(edge["u"]))
            nodes.add(int(edge["v"]))
        out["n_nodes"] = len(nodes)
        out["n_edges"] = len(graph_edges)
        out["optimal_path_length"] = _sp_optimal_cost(params, verifier_function)
        return out

    if subtype == "WIS" or "interval" in str(problem_subtype).lower():
        intervals = params.get("intervals") or []
        out["n_intervals"] = len(intervals)
        out["optimal_weight"] = _wis_optimal_weight(params)
        return out

    return out
