#!/usr/bin/env python3
"""Verify SP_005 shortest path using Dijkstra on the canonical edge list."""

from __future__ import annotations

import networkx as nx


EDGES = [
    (0, 1, 5),
    (0, 3, 8),
    (0, 4, 7),
    (1, 2, 4),
    (1, 5, 6),
    (1, 7, 9),
    (2, 6, 5),
    (2, 8, 7),
    (3, 7, 6),
    (3, 9, 8),
    (4, 5, 4),
    (4, 9, 7),
    (5, 6, 3),
    (5, 10, 8),
    (6, 8, 4),
    (6, 10, 6),
    (7, 8, 5),
    (7, 9, 7),
    (8, 10, 5),
    (8, 11, 8),
    (9, 11, 6),
    (9, 12, 9),
    (10, 12, 4),
    (10, 13, 7),
    (11, 13, 5),
    (11, 14, 8),
    (12, 14, 6),
    (12, 15, 10),
    (13, 14, 3),
    (13, 15, 11),
    (14, 15, 4),
]


def main() -> None:
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(EDGES)

    source = 0
    target = 15

    optimal_path = nx.dijkstra_path(graph, source=source, target=target, weight="weight")
    optimal_cost = nx.dijkstra_path_length(
        graph, source=source, target=target, weight="weight"
    )

    print(f"Optimal path from {source} to {target}: {optimal_path}")
    print(f"Optimal cost: {optimal_cost}")


if __name__ == "__main__":
    main()
