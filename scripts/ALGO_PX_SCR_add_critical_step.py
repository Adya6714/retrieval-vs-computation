#!/usr/bin/env python3
"""Add/propagate critical_step_index into question_bank_algo difficulty_params."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


BANK_PATH = Path("data/problems/question_bank_algo.csv")


def _parse_cc_coins(answer: str) -> list[int]:
    m = re.search(r"(?:Coins|Scoops)\s*:\s*\[([^\]]*)\]", str(answer), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unable to parse coin sequence from answer: {answer!r}")
    return [int(x.strip()) for x in m.group(1).split(",") if x.strip()]


def _cc_greedy_sequence(denoms: list[int], target: int) -> list[int]:
    rem = target
    out: list[int] = []
    for c in sorted(denoms, reverse=True):
        while rem >= c:
            out.append(c)
            rem -= c
    if rem != 0:
        raise ValueError(f"Greedy could not make exact target={target} with denoms={denoms}")
    return out


def _first_divergence(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i  # 0-based
    if len(a) != len(b):
        return n
    return -1


def _parse_wis_selected(answer: str) -> list[int]:
    m = re.search(r"Selected:\s*\{([^}]*)\}", str(answer), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Unable to parse selected set from answer: {answer!r}")
    return [int(x.strip()) for x in m.group(1).split(",") if x.strip()]


def _parse_wis_edges(problem_text: str) -> list[tuple[int, int]]:
    edges = [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*-\s*(\d+)", str(problem_text))]
    if not edges:
        raise ValueError("Unable to parse WIS edges from problem_text.")
    norm = {(min(a, b), max(a, b)) for a, b in edges}
    return sorted(norm)


def _wis_neighbors(nodes: list[int], edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    nbrs = {n: set() for n in nodes}
    for a, b in edges:
        if a in nbrs and b in nbrs:
            nbrs[a].add(b)
            nbrs[b].add(a)
    return nbrs


def _wis_greedy_order(weights: dict[int, int], edges: list[tuple[int, int]]) -> list[int]:
    nodes = sorted(weights)
    nbrs = _wis_neighbors(nodes, edges)
    remaining = set(nodes)
    order: list[int] = []
    while remaining:
        best = max(remaining, key=lambda n: (weights[n], -n))
        order.append(best)
        remaining.discard(best)
        for nb in nbrs.get(best, set()):
            remaining.discard(nb)
    return order


def _wis_optimal_induced_order(
    selected: list[int], weights: dict[int, int], edges: list[tuple[int, int]]
) -> list[int]:
    selected_set = set(selected)
    nbrs = _wis_neighbors(sorted(weights), edges)
    if any((b in nbrs.get(a, set())) for a in selected_set for b in selected_set if a != b):
        raise ValueError("Optimal selected set is not independent.")

    remaining = set(selected_set)
    blocked: set[int] = set()
    order: list[int] = []
    while remaining:
        # Deterministic order within optimal set: highest weight then smallest id.
        candidates = [n for n in remaining if n not in blocked]
        if not candidates:
            # Should not happen for independent set, but keep fail-loud.
            raise ValueError("No valid candidate left while ordering optimal set.")
        pick = max(candidates, key=lambda n: (weights[n], -n))
        order.append(pick)
        remaining.remove(pick)
        blocked.update(nbrs.get(pick, set()))
    return order


def main() -> None:
    if not BANK_PATH.exists():
        raise FileNotFoundError(f"Missing bank file: {BANK_PATH}")

    df = pd.read_csv(BANK_PATH, dtype=str).fillna("")
    canon = df[df["variant_type"] == "canonical"].copy()
    canon = canon[canon["problem_id"].str.match(r"^(CC|SP|WIS)_")]

    if canon.empty:
        raise ValueError("No canonical algorithmic rows found.")

    critical_by_pid: dict[str, int] = {}
    summary_by_subtype: dict[str, Counter] = defaultdict(Counter)

    for _, row in canon.iterrows():
        pid = str(row["problem_id"]).strip()
        subtype = str(row["problem_subtype"]).strip().lower()
        params = json.loads(str(row["difficulty_params"]))
        instance_type = str(params.get("instance_type", "")).strip().lower()

        if subtype == "coin_change":
            if instance_type == "standard":
                critical = -1
            elif instance_type == "adversarial":
                denoms = [int(x) for x in params["denominations"]]
                target = int(params["target"])
                greedy_seq = _cc_greedy_sequence(denoms, target)
                optimal_seq = _parse_cc_coins(row["correct_answer"])
                critical = _first_divergence(greedy_seq, optimal_seq)
                if critical == -1:
                    # adversarial should diverge
                    critical = 1
            else:
                raise ValueError(f"{pid}: invalid instance_type={instance_type!r}")

        elif subtype == "wis":
            if instance_type == "standard":
                critical = -1
            elif instance_type == "adversarial":
                intervals = params.get("intervals")
                if not isinstance(intervals, list):
                    raise ValueError(f"{pid}: intervals missing in difficulty_params.")
                weights: dict[int, int] = {}
                for entry in intervals:
                    if isinstance(entry, dict):
                        weights[int(entry["id"])] = int(entry["weight"])
                    elif isinstance(entry, list) and len(entry) >= 3:
                        # Fallback shape [start,end,weight] with implicit id by index.
                        idx = len(weights)
                        weights[idx] = int(entry[2])
                    else:
                        raise ValueError(f"{pid}: invalid interval entry: {entry!r}")
                edges = _parse_wis_edges(row["problem_text"])
                greedy_order = _wis_greedy_order(weights, edges)
                optimal_selected = _parse_wis_selected(row["correct_answer"])
                optimal_order = _wis_optimal_induced_order(optimal_selected, weights, edges)
                critical = _first_divergence(greedy_order, optimal_order)
                if critical == -1:
                    critical = 1
            else:
                raise ValueError(f"{pid}: invalid instance_type={instance_type!r}")

        elif subtype == "shortest_path":
            if instance_type == "standard":
                critical = -1
            elif instance_type == "adversarial":
                if pid == "SP_004":
                    critical = 0
                elif pid in {"SP_003", "SP_005"}:
                    critical = 1
                else:
                    raise ValueError(
                        f"{pid}: adversarial SP rule missing; only SP_003/004/005 supported."
                    )
            else:
                raise ValueError(f"{pid}: invalid instance_type={instance_type!r}")
        else:
            raise ValueError(f"{pid}: unexpected subtype={subtype!r}")

        critical_by_pid[pid] = critical
        summary_by_subtype[subtype][critical] += 1

    updated_rows = 0
    for idx, row in df.iterrows():
        pid = str(row["problem_id"]).strip()
        if pid not in critical_by_pid:
            continue
        params = json.loads(str(row["difficulty_params"]))
        new_value = int(critical_by_pid[pid])
        if params.get("critical_step_index") != new_value:
            params["critical_step_index"] = new_value
            df.at[idx, "difficulty_params"] = json.dumps(
                params, separators=(",", ":"), ensure_ascii=True
            )
            updated_rows += 1

    df.to_csv(BANK_PATH, index=False)

    print(f"Updated bank: {BANK_PATH}")
    print(f"Rows updated: {updated_rows}")
    print("Summary (canonical values by subtype):")
    for subtype in ("coin_change", "shortest_path", "wis"):
        counter = summary_by_subtype.get(subtype, Counter())
        print(f"- {subtype}: {dict(sorted(counter.items(), key=lambda kv: kv[0]))}")


if __name__ == "__main__":
    main()
