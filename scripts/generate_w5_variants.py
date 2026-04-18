#!/usr/bin/env python3
"""Generate deterministic W5 variants for Probe 1 instances."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import networkx as nx
import numpy as np


INPUT_CSV = Path("data/problems/probe1_instances.csv")
OUTPUT_CSV = Path("data/problems/probe1_variants.csv")
OUTPUT_COLUMNS = [
    "problem_id",
    "variant_type",
    "problem_text",
    "correct_answer",
    "source_problem_id",
]
VARIANT_TYPE = "W5"


@dataclass(frozen=True)
class ProblemRow:
    problem_id: str
    problem_family: str
    problem_text: str
    correct_answer: str
    raw: Dict[str, str]


def _seed_from_problem_id(problem_id: str) -> int:
    digest = hashlib.sha256(problem_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def _read_input_rows(path: Path) -> List[ProblemRow]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: List[ProblemRow] = []
        for raw in reader:
            rows.append(
                ProblemRow(
                    problem_id=str(raw.get("problem_id", "")).strip(),
                    problem_family=str(raw.get("problem_family", "")).strip().lower(),
                    problem_text=str(raw.get("problem_text", "")).strip(),
                    correct_answer=str(raw.get("correct_answer", "")).strip(),
                    raw={k: (v if v is not None else "") for k, v in raw.items()},
                )
            )
    return rows


def _read_existing_source_ids(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row.get("source_problem_id", "")).strip()
            for row in reader
            if str(row.get("source_problem_id", "")).strip()
        }


def _next_problem_id(source_problem_id: str, used_ids: Set[str]) -> str:
    candidate = f"{source_problem_id}_{VARIANT_TYPE}"
    if candidate not in used_ids:
        used_ids.add(candidate)
        return candidate
    idx = 2
    while True:
        candidate = f"{source_problem_id}_{VARIANT_TYPE}_{idx}"
        if candidate not in used_ids:
            used_ids.add(candidate)
            return candidate
        idx += 1


def _parse_num_nodes(row: ProblemRow) -> int:
    for key in ("num_nodes", "n_nodes", "nodes"):
        raw_value = row.raw.get(key, "").strip()
        if raw_value.isdigit():
            return max(2, int(raw_value))
    labels = set(re.findall(r"\b[A-Z]\b", row.problem_text))
    if len(labels) >= 2:
        return len(labels)
    return 6


def _generate_shortest_path(row: ProblemRow, rng: np.random.Generator) -> Tuple[str, str]:
    num_nodes = _parse_num_nodes(row)
    node_labels = [chr(ord("A") + i) for i in range(num_nodes)]

    graph = nx.Graph()
    graph.add_nodes_from(node_labels)

    # Build a connected graph via a random spanning tree.
    for idx in range(1, num_nodes):
        neighbor_idx = int(rng.integers(0, idx))
        weight = int(rng.integers(1, 11))
        graph.add_edge(node_labels[idx], node_labels[neighbor_idx], weight=weight)

    # Add extra random edges.
    possible_edges: List[Tuple[str, str]] = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if not graph.has_edge(node_labels[i], node_labels[j]):
                possible_edges.append((node_labels[i], node_labels[j]))
    rng.shuffle(possible_edges)
    extra_edges = int(rng.integers(max(1, num_nodes // 2), max(2, num_nodes)))
    for u, v in possible_edges[:extra_edges]:
        graph.add_edge(u, v, weight=int(rng.integers(1, 11)))

    start, end = rng.choice(node_labels, size=2, replace=False).tolist()
    path_nodes = nx.shortest_path(graph, source=start, target=end, weight="weight")

    edge_desc = ", ".join(
        f"{u}-{v}:{int(data['weight'])}" for u, v, data in sorted(graph.edges(data=True))
    )
    problem_text = (
        f"Given an undirected weighted graph with edges {edge_desc}, "
        f"find the shortest path from {start} to {end}. "
        "Respond as comma-separated node labels."
    )
    correct_answer = ",".join(path_nodes)
    return problem_text, correct_answer


def _extract_block_names(row: ProblemRow) -> List[str]:
    for key in ("num_blocks", "n_blocks", "blocks"):
        raw_value = row.raw.get(key, "").strip()
        if raw_value.isdigit():
            count = max(2, int(raw_value))
            return [chr(ord("A") + i) for i in range(count)]
    names = sorted(set(re.findall(r"\b[A-Z]\b", row.problem_text)))
    if len(names) >= 2:
        return names
    return ["A", "B", "C", "D"]


def _random_stacks(blocks: Sequence[str], rng: np.random.Generator) -> Tuple[Tuple[str, ...], ...]:
    blocks_list = list(blocks)
    rng.shuffle(blocks_list)
    stack_count = int(rng.integers(1, len(blocks_list) + 1))
    stacks: List[List[str]] = [[] for _ in range(stack_count)]
    for block in blocks_list:
        stacks[int(rng.integers(0, stack_count))].append(block)
    stacks = [s for s in stacks if s]
    rng.shuffle(stacks)
    return tuple(tuple(s) for s in stacks)


def _stacks_to_positions(stacks: Tuple[Tuple[str, ...], ...]) -> Dict[str, str]:
    positions: Dict[str, str] = {}
    for stack in stacks:
        for i, block in enumerate(stack):
            positions[block] = "table" if i == 0 else stack[i - 1]
    return positions


def _positions_to_state(positions: Dict[str, str]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(positions.items()))


def _clear_blocks(positions: Dict[str, str]) -> Set[str]:
    supported: Set[str] = {support for support in positions.values() if support != "table"}
    return set(positions.keys()) - supported


def _neighbors(state: Tuple[Tuple[str, str], ...]) -> Iterable[Tuple[Tuple[Tuple[str, str], ...], str]]:
    positions = dict(state)
    clear = _clear_blocks(positions)
    blocks = sorted(positions.keys())
    for block in sorted(clear):
        current_support = positions[block]
        for target in ["table"] + sorted(clear):
            if target == block or target == current_support:
                continue
            if target != "table" and target not in clear:
                continue
            new_positions = dict(positions)
            new_positions[block] = target
            move = f"move {block} from {current_support} to {target}"
            yield _positions_to_state(new_positions), move


def _find_plan(
    initial: Tuple[Tuple[str, ...], ...], goal: Tuple[Tuple[str, ...], ...], max_depth: int = 24
) -> List[str]:
    start_state = _positions_to_state(_stacks_to_positions(initial))
    goal_state = _positions_to_state(_stacks_to_positions(goal))
    if start_state == goal_state:
        return []

    queue: List[Tuple[Tuple[Tuple[str, str], ...], List[str]]] = [(start_state, [])]
    seen = {start_state}
    head = 0
    while head < len(queue):
        state, path = queue[head]
        head += 1
        if len(path) >= max_depth:
            continue
        for nxt_state, move in _neighbors(state):
            if nxt_state in seen:
                continue
            next_path = path + [move]
            if nxt_state == goal_state:
                return next_path
            seen.add(nxt_state)
            queue.append((nxt_state, next_path))
    return []


def _describe_stacks(stacks: Tuple[Tuple[str, ...], ...]) -> str:
    return "; ".join("[" + ", ".join(stack) + "]" for stack in stacks)


def _generate_blocksworld(row: ProblemRow, rng: np.random.Generator) -> Tuple[str, str]:
    blocks = _extract_block_names(row)
    max_attempts = 30
    for _ in range(max_attempts):
        initial = _random_stacks(blocks, rng)
        goal = _random_stacks(blocks, rng)
        plan = _find_plan(initial, goal)
        if plan:
            problem_text = (
                "Blocks: "
                + ", ".join(blocks)
                + f". Initial state stacks: {_describe_stacks(initial)}. "
                + f"Goal state stacks: {_describe_stacks(goal)}. "
                + "Provide a valid sequence of moves in the format "
                + "'move X from Y to Z', one move per line."
            )
            return problem_text, "\n".join(plan)

    # Fallback to a no-op problem if random search fails.
    initial = _random_stacks(blocks, rng)
    problem_text = (
        "Blocks: "
        + ", ".join(blocks)
        + f". Initial state stacks: {_describe_stacks(initial)}. "
        + f"Goal state stacks: {_describe_stacks(initial)}. "
        + "Provide a valid sequence of moves in the format "
        + "'move X from Y to Z', one move per line."
    )
    return problem_text, ""


_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?")


def _scale_number_string(value: str, factor: float) -> str:
    number = float(value)
    scaled = int(round(number * factor))
    if number != 0 and scaled == 0:
        scaled = 1 if number > 0 else -1
    return str(scaled)


def _generate_gsm(row: ProblemRow, rng: np.random.Generator) -> Tuple[str, str]:
    factor = float(rng.uniform(0.5, 2.0))
    problem_text = _NUMBER_PATTERN.sub(
        lambda match: _scale_number_string(match.group(0), factor), row.problem_text
    )
    if "leaving" in problem_text.lower() or "remaining" in problem_text.lower() or "left" in problem_text.lower():
        print(f"WARNING: {row.problem_id} may have intermediate values in text that "
              f"scaling doesn't update. Manual review required.")
    try:
        answer_value = float(row.correct_answer)
        correct_answer = str(int(round(answer_value * factor)))
    except ValueError:
        # If non-numeric answer appears, scale numbers in-place.
        correct_answer = _NUMBER_PATTERN.sub(
            lambda match: _scale_number_string(match.group(0), factor), row.correct_answer
        )
    return problem_text, correct_answer


def _generate_variant(row: ProblemRow, rng: np.random.Generator) -> Tuple[str, str]:
    if row.problem_family == "shortest_path":
        return _generate_shortest_path(row, rng)
    if row.problem_family == "blocksworld":
        return _generate_blocksworld(row, rng)
    if row.problem_family == "gsm":
        return _generate_gsm(row, rng)
    raise ValueError(f"Unsupported problem_family '{row.problem_family}' for {row.problem_id}")


def _collect_used_problem_ids(output_path: Path) -> Set[str]:
    if not output_path.exists():
        return set()
    with output_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {str(row.get("problem_id", "")).strip() for row in reader if row.get("problem_id")}


def _append_rows(path: Path, rows: Sequence[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = (not path.exists()) or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    input_rows = _read_input_rows(INPUT_CSV)
    existing_source_ids = _read_existing_source_ids(OUTPUT_CSV)
    used_problem_ids = _collect_used_problem_ids(OUTPUT_CSV)

    new_rows: List[Dict[str, str]] = []
    for row in input_rows:
        if not row.problem_id:
            continue
        if row.problem_id in existing_source_ids:
            continue

        rng = np.random.default_rng(_seed_from_problem_id(row.problem_id))
        problem_text, correct_answer = _generate_variant(row, rng)
        new_rows.append(
            {
                "problem_id": _next_problem_id(row.problem_id, used_problem_ids),
                "variant_type": VARIANT_TYPE,
                "problem_text": problem_text,
                "correct_answer": correct_answer,
                "source_problem_id": row.problem_id,
            }
        )

    if new_rows:
        _append_rows(OUTPUT_CSV, new_rows)
    print(f"Wrote {len(new_rows)} new variants to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
