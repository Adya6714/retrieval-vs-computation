#!/usr/bin/env python3
"""Update shortest_path W5 difficulty_params to swapped endpoints.

Stage 2 ``generate_w5_sp`` solves the reversed graph from canonical target to
canonical source and writes that path as gold, but copies canonical
``difficulty_params`` unchanged. The verifier then rejects gold with
``path_endpoints_invalid``.

This script:
  * swaps ``source`` and ``target``
  * reverses the stored edge list so it matches the graph used to compute gold
  * logs every row it changes
  * reports rows whose original directed graph has no path in the swapped
    direction, and rows that still have no path after the update
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BANK_PATH = REPO_ROOT / "data/problems/question_bank_algo.csv"


def _graph_from_params(params: dict) -> nx.DiGraph | nx.Graph:
    directed = bool(params.get("directed", True))
    graph = nx.DiGraph() if directed else nx.Graph()
    for edge in params.get("graph", []):
        graph.add_edge(int(edge["u"]), int(edge["v"]), weight=int(edge["w"]))
    source = int(params["source"])
    target = int(params["target"])
    graph.add_nodes_from([source, target])
    return graph


def _has_path(params: dict, source: int, target: int) -> bool:
    graph = _graph_from_params(params)
    if source not in graph or target not in graph:
        return False
    return nx.has_path(graph, source, target)


def _reverse_edges(graph: list[dict]) -> list[dict]:
    reversed_edges: list[dict] = []
    for edge in graph:
        reversed_edges.append(
            {
                **edge,
                "u": int(edge["v"]),
                "v": int(edge["u"]),
                "w": int(edge["w"]),
            }
        )
    return reversed_edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bank", type=Path, default=BANK_PATH)
    args = parser.parse_args()

    with args.bank.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    changed: list[dict[str, object]] = []
    not_traversable_before: list[str] = []
    not_traversable_after: list[str] = []

    for row in rows:
        if str(row.get("problem_subtype", "")).strip() != "shortest_path":
            continue
        if str(row.get("variant_type", "")).strip() != "W5":
            continue
        params = json.loads(row["difficulty_params"])
        old_source = int(params["source"])
        old_target = int(params["target"])
        new_source, new_target = old_target, old_source
        pid = str(row["problem_id"]).strip()

        traversable_endpoint_only = _has_path(params, new_source, new_target)
        if not traversable_endpoint_only:
            not_traversable_before.append(pid)

        new_params = dict(params)
        new_params["source"] = new_source
        new_params["target"] = new_target
        new_params["graph"] = _reverse_edges(list(params.get("graph") or []))

        traversable_after = _has_path(new_params, new_source, new_target)
        if not traversable_after:
            not_traversable_after.append(pid)

        encoded = json.dumps(new_params, ensure_ascii=False)
        changed.append(
            {
                "problem_id": pid,
                "old_source": old_source,
                "old_target": old_target,
                "new_source": new_source,
                "new_target": new_target,
                "n_edges": len(new_params["graph"]),
                "traversable_endpoint_only": traversable_endpoint_only,
                "traversable_after": traversable_after,
            }
        )
        row["difficulty_params"] = encoded

    print(f"{'Would change' if args.dry_run else 'Changed'} {len(changed)} W5 shortest_path rows:")
    for item in changed:
        print(
            f"  {item['problem_id']}: source/target {item['old_source']}->{item['old_target']} "
            f"-> {item['new_source']}->{item['new_target']}; "
            f"reversed {item['n_edges']} edges; "
            f"endpoint-only path={item['traversable_endpoint_only']}; "
            f"after path={item['traversable_after']}"
        )
    print(
        f"Original directed graph not traversable after endpoint-only swap: "
        f"{len(not_traversable_before)}/{len(changed)}"
    )
    if not_traversable_before:
        print("  " + ", ".join(not_traversable_before))
    print(
        f"No valid path after swapping endpoints and reversing edges: "
        f"{len(not_traversable_after)}/{len(changed)}"
    )
    if not_traversable_after:
        print("  " + ", ".join(not_traversable_after))

    if not args.dry_run:
        with args.bank.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.bank}")


if __name__ == "__main__":
    main()
