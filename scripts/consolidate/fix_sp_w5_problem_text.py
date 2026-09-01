#!/usr/bin/env python3
"""Rewrite shortest_path W5 problem_text so printed edges match reversed params.

The previous endpoint migration reversed difficulty_params and gold, but left
the prompt's road list in the original direction. This script flips each
``Node U to Node V: W`` line (case-insensitive) and asserts that the printed
graph has a path whose cost equals gold.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BANK_PATH = REPO_ROOT / "data/problems/question_bank_algo.csv"

_EDGE_LINE = re.compile(
    r"(?P<prefix>^\s*[-*]?\s*)[Nn]ode\s+(?P<u>\d+)\s+to\s+[Nn]ode\s+(?P<v>\d+)\s*:\s*(?P<w>-?\d+)\s*$"
)
_QUERY_ENDPOINTS = re.compile(
    r"from\s+node\s+(\d+)\s+to\s+node\s+(\d+)",
    re.IGNORECASE,
)
_GOLD = re.compile(
    r"Path:\s*(.+?)\s*,\s*Cost:\s*(-?\d+)",
    re.IGNORECASE,
)


def rewrite_edge_lines(text: str) -> str:
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        core, nl = (line[:-1], line[-1]) if line.endswith("\n") else (line, "")
        m = _EDGE_LINE.match(core)
        if not m:
            out.append(line)
            continue
        prefix, u, v, w = m.group("prefix"), m.group("u"), m.group("v"), m.group("w")
        # Preserve the original "Node" capitalization from the line.
        node_word = "Node" if "Node" in core else "node"
        out.append(f"{prefix}{node_word} {v} to {node_word} {u}: {w}{nl}")
    return "".join(out)


def parse_printed_graph(text: str) -> tuple[list[tuple[int, int, int]], int | None, int | None]:
    edges: list[tuple[int, int, int]] = []
    for line in text.splitlines():
        m = _EDGE_LINE.match(line)
        if m:
            edges.append((int(m.group("u")), int(m.group("v")), int(m.group("w"))))
    q = _QUERY_ENDPOINTS.search(text)
    src = int(q.group(1)) if q else None
    tgt = int(q.group(2)) if q else None
    return edges, src, tgt


def gold_cost(answer: str) -> int | None:
    m = _GOLD.search(answer)
    return int(m.group(2)) if m else None


def path_cost(graph: nx.DiGraph, source: int, target: int, use_bf: bool) -> int:
    if use_bf:
        return int(nx.bellman_ford_path_length(graph, source, target, weight="weight"))
    return int(nx.dijkstra_path_length(graph, source, target, weight="weight"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--bank", type=Path, default=BANK_PATH)
    args = parser.parse_args()

    with args.bank.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    changed = 0
    failures: list[str] = []
    for row in rows:
        if str(row.get("problem_subtype", "")).strip() != "shortest_path":
            continue
        if str(row.get("variant_type", "")).strip() != "W5":
            continue
        pid = str(row["problem_id"]).strip()
        params = json.loads(row["difficulty_params"])
        new_text = rewrite_edge_lines(row["problem_text"])
        edges, src, tgt = parse_printed_graph(new_text)
        wanted = {(int(e["u"]), int(e["v"]), int(e["w"])) for e in params.get("graph") or []}
        got = set(edges)
        if wanted != got:
            failures.append(
                f"{pid}: printed edges {sorted(got)} != params {sorted(wanted)}"
            )
            continue
        if src != int(params["source"]) or tgt != int(params["target"]):
            failures.append(
                f"{pid}: query {src}->{tgt} != params {params['source']}->{params['target']}"
            )
            continue
        graph = nx.DiGraph()
        for u, v, w in edges:
            graph.add_edge(u, v, weight=w)
        use_bf = bool(params.get("requires_bellman_ford", False))
        try:
            if not nx.has_path(graph, src, tgt):
                failures.append(f"{pid}: no printed path {src}->{tgt}")
                continue
            cost = path_cost(graph, src, tgt, use_bf)
        except Exception as exc:
            failures.append(f"{pid}: solver {type(exc).__name__}: {exc}")
            continue
        expected = gold_cost(row["correct_answer"])
        if expected is None or cost != expected:
            failures.append(f"{pid}: printed cost {cost} != gold {expected}")
            continue
        if new_text != row["problem_text"]:
            changed += 1
            print(f"  {pid}: reversed {len(edges)} printed edges; path {src}->{tgt} cost {cost}")
            row["problem_text"] = new_text

    print(f"{'Would change' if args.dry_run else 'Changed'} {changed} W5 prompts")
    if failures:
        print(f"ASSERT FAIL {len(failures)}:")
        for item in failures:
            print("  " + item)
        raise SystemExit(1)
    print("All 50 W5 prompts have a printed path whose cost matches gold.")
    if not args.dry_run:
        with args.bank.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {args.bank}")


if __name__ == "__main__":
    main()
