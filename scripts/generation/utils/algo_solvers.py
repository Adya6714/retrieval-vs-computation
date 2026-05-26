from __future__ import annotations

import networkx as nx


def bellman_ford_sp(G: nx.DiGraph, source: int, target: int) -> tuple[list, float]:
    try:
        path = nx.bellman_ford_path(G, source=source, target=target, weight="weight")
        cost = nx.bellman_ford_path_length(G, source=source, target=target, weight="weight")
        return path, float(cost)
    except nx.NetworkXUnbounded:
        return [], float("inf")
