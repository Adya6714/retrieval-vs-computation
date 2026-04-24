#!/usr/bin/env python3
"""Backfill greedy_succeeds + instance_type for ALGO bank difficulty_params."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


BANK_PATH = Path("data/problems/question_bank_algo.csv")


def _parse_cc_optimal_count(ans: str) -> int:
    m = re.search(r"(?:Count|Total)\s*:\s*(\d+)", str(ans), flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse CC optimal count from {ans!r}")
    return int(m.group(1))


def _cc_greedy_count(denoms: list[int], target: int) -> int | None:
    rem = target
    cnt = 0
    for c in sorted(denoms, reverse=True):
        q, rem = divmod(rem, c)
        cnt += q
    if rem != 0:
        return None
    return cnt


def _parse_wis_total(ans: str) -> int:
    m = re.search(r"Total:\s*(\d+)", str(ans))
    if not m:
        raise ValueError(f"Cannot parse WIS total from {ans!r}")
    return int(m.group(1))


def _parse_wis_weights_edges(text: str) -> tuple[dict[int, int], list[tuple[int, int]]]:
    weights: dict[int, int] = {}
    for i, w in re.findall(
        r"\b(?:Plot|District|Server|Tower|Center|House|Node)\s+(\d+)\s*:\s*(\d+)",
        text,
        flags=re.IGNORECASE,
    ):
        weights[int(i)] = int(w)
    for i, w in re.findall(r"(\d+)\s*->\s*(\d+)", text):
        weights.setdefault(int(i), int(w))
    edges = [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*-\s*(\d+)", text)]
    if not weights or not edges:
        raise ValueError("Unable to parse WIS weights/edges from canonical text.")
    return weights, edges


def _wis_greedy_weight(weights: dict[int, int], edges: list[tuple[int, int]]) -> int:
    neighbors = {n: set() for n in weights}
    for a, b in edges:
        if a in neighbors and b in neighbors:
            neighbors[a].add(b)
            neighbors[b].add(a)
    remaining = set(weights)
    total = 0
    while remaining:
        best = max(remaining, key=lambda n: (weights[n], -n))
        total += weights[best]
        remaining.discard(best)
        for nb in neighbors.get(best, set()):
            remaining.discard(nb)
    return total


def main() -> None:
    if not BANK_PATH.exists():
        raise FileNotFoundError(f"Missing bank file: {BANK_PATH}")
    df = pd.read_csv(BANK_PATH, dtype=str).fillna("")

    canon = df[df["variant_type"] == "canonical"].copy()
    canon = canon[canon["problem_id"].str.match(r"^(CC|SP|WIS)_")]

    per_pid: dict[str, bool] = {}
    per_subtype = defaultdict(int)

    for _, row in canon.iterrows():
        pid = str(row["problem_id"])
        params = json.loads(str(row["difficulty_params"]))
        subtype = str(row["problem_subtype"]).strip().lower()

        if pid.startswith("CC_"):
            denoms = [int(x) for x in params["denominations"]]
            target = int(params["target"])
            optimal = _parse_cc_optimal_count(row["correct_answer"])
            g = _cc_greedy_count(denoms, target)
            greedy_succeeds = g is not None and g == optimal
            per_pid[pid] = bool(greedy_succeeds)
            per_subtype["coin_change"] += 1
        elif pid.startswith("WIS_"):
            weights, edges = _parse_wis_weights_edges(str(row["problem_text"]))
            greedy_w = _wis_greedy_weight(weights, edges)
            optimal_w = _parse_wis_total(row["correct_answer"])
            per_pid[pid] = bool(greedy_w == optimal_w)
            per_subtype["wis"] += 1
        elif pid.startswith("SP_"):
            if pid in {"SP_001", "SP_002"}:
                per_pid[pid] = True
            elif pid in {"SP_003", "SP_004", "SP_005"}:
                per_pid[pid] = False
            else:
                raise ValueError(f"Unexpected SP id for rule-based assignment: {pid}")
            per_subtype["shortest_path"] += 1
        else:
            raise ValueError(f"Unexpected canonical algorithmic id: {pid}")

    updates = defaultdict(int)
    for idx, row in df.iterrows():
        pid = str(row["problem_id"])
        if pid not in per_pid:
            continue
        params = json.loads(str(row["difficulty_params"]))
        greedy = bool(per_pid[pid])
        instance_type = "standard" if greedy else "adversarial"
        changed = False
        if params.get("greedy_succeeds") != greedy:
            params["greedy_succeeds"] = greedy
            changed = True
        if params.get("instance_type") != instance_type:
            params["instance_type"] = instance_type
            changed = True
        if changed:
            df.at[idx, "difficulty_params"] = json.dumps(
                params, separators=(",", ":"), ensure_ascii=True
            )
            updates[str(row["problem_subtype"]).strip().lower()] += 1

    df.to_csv(BANK_PATH, index=False)
    print(f"Updated bank: {BANK_PATH}")
    print("Rows updated per subtype:")
    for subtype in ("coin_change", "shortest_path", "wis"):
        print(f"- {subtype}: {updates[subtype]}")


if __name__ == "__main__":
    main()
