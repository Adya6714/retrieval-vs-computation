#!/usr/bin/env python3
"""Verify WIS_005 via maximum weight independent set computation."""

from __future__ import annotations

import networkx as nx


WEIGHTS = {
    0: 12,
    1: 9,
    2: 14,
    3: 11,
    4: 8,
    5: 13,
    6: 10,
    7: 6,
    8: 15,
    9: 7,
    10: 11,
    11: 9,
    12: 8,
    13: 12,
    14: 10,
    15: 7,
    16: 13,
    17: 6,
}

EDGES = [
    (0, 1),
    (0, 4),
    (0, 8),
    (1, 2),
    (1, 5),
    (1, 9),
    (2, 3),
    (2, 6),
    (2, 10),
    (3, 7),
    (3, 11),
    (4, 5),
    (4, 12),
    (5, 6),
    (5, 13),
    (6, 7),
    (6, 14),
    (7, 15),
    (8, 9),
    (8, 12),
    (8, 16),
    (9, 10),
    (9, 13),
    (10, 11),
    (10, 14),
    (11, 15),
    (11, 17),
    (12, 13),
    (12, 16),
    (13, 14),
    (13, 17),
    (14, 15),
    (16, 17),
    (0, 5),
    (1, 8),
    (2, 12),
    (3, 13),
    (4, 10),
    (6, 16),
    (7, 12),
    (9, 14),
    (11, 16),
]


def main() -> None:
    graph = nx.Graph()
    graph.add_nodes_from(WEIGHTS.keys())
    graph.add_edges_from(EDGES)

    complement_graph = nx.complement(graph)
    for node, weight in WEIGHTS.items():
        complement_graph.nodes[node]["weight"] = weight

    clique_nodes, max_weight = nx.max_weight_clique(complement_graph, weight="weight")
    independent_set = sorted(clique_nodes)

    print(f"Optimal independent set: {independent_set}")
    print(f"Total weight: {max_weight}")


if __name__ == "__main__":
    main()
