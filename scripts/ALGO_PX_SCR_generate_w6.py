#!/usr/bin/env python3
"""Generate W6 procedural variants for algorithmic suite instances."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import pandas as pd


SEED = 42
TARGET_BANK = Path("data/problems/question_bank.csv")
SOURCE_FALLBACK_BANK = Path("data/problems/question_bank_algo.csv")


def parse_kv_blob(blob: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(blob).split("|"):
        token = part.strip()
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def parse_first_int(text: str, pattern: str) -> int:
    match = re.search(pattern, text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Could not parse integer using pattern: {pattern}")
    return int(match.group(1))


def parse_cc_instance(problem_text: str) -> tuple[list[int], int]:
    denom_match = re.search(r"denominations:\s*\[([^\]]+)\]", problem_text, flags=re.IGNORECASE)
    if not denom_match:
        raise ValueError("Could not parse CC denominations from problem_text.")
    denoms = [int(x.strip()) for x in denom_match.group(1).split(",") if x.strip()]
    target = parse_first_int(problem_text, r"exact change for\s*(\d+)")
    return denoms, target


def dp_coin_change(denoms: Sequence[int], target: int) -> tuple[int, list[int]] | None:
    inf = 10**9
    dp = [inf] * (target + 1)
    prev = [-1] * (target + 1)
    dp[0] = 0
    for amount in range(1, target + 1):
        for c in denoms:
            if c <= amount and dp[amount - c] + 1 < dp[amount]:
                dp[amount] = dp[amount - c] + 1
                prev[amount] = c
    if dp[target] >= inf:
        return None
    coins: list[int] = []
    cur = target
    while cur > 0:
        c = prev[cur]
        if c == -1:
            return None
        coins.append(c)
        cur -= c
    coins.sort(reverse=True)
    return dp[target], coins


def greedy_coin_change(denoms: Sequence[int], target: int) -> tuple[int, list[int]] | None:
    rem = target
    coins: list[int] = []
    for c in sorted(denoms, reverse=True):
        while rem >= c:
            rem -= c
            coins.append(c)
    if rem != 0:
        return None
    return len(coins), coins


def render_cc_text_from_canonical(canonical_text: str, denoms: Sequence[int], target: int) -> str:
    text = canonical_text
    text = re.sub(
        r"denominations:\s*\[[^\]]+\]",
        f"denominations: [{', '.join(str(d) for d in denoms)}]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"exact change for\s*\d+",
        f"exact change for {target}",
        text,
        flags=re.IGNORECASE,
    )
    return text


def format_cc_answer(count: int, coins: Sequence[int]) -> str:
    return f"Count: {count}\nCoins: [{', '.join(str(c) for c in coins)}]"


def parse_src_tgt_from_answer(answer: str) -> tuple[int, int]:
    nums = [int(x) for x in re.findall(r"\d+", answer)]
    if len(nums) < 2:
        raise ValueError(f"Could not parse src/tgt from SP answer: {answer!r}")
    return nums[0], nums[-2] if "Cost" in answer and len(nums) >= 3 else nums[-1]


def parse_src_tgt_from_sp_text(text: str, fallback_answer: str) -> tuple[int, int]:
    m = re.search(r"from [^\n]*?(\d+)\s+to [^\n]*?(\d+)", text, flags=re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    nums = [int(x) for x in re.findall(r"\d+", fallback_answer)]
    if len(nums) < 2:
        raise ValueError("Unable to parse SP source/target.")
    return nums[0], nums[-2] if len(nums) >= 3 else nums[-1]


def shortest_path_unique(
    graph: nx.Graph | nx.DiGraph,
    src: int,
    tgt: int,
    use_bellman_ford: bool,
) -> tuple[list[int], int] | None:
    try:
        if use_bellman_ford:
            path = nx.bellman_ford_path(graph, src, tgt, weight="weight")
            cost = int(nx.bellman_ford_path_length(graph, src, tgt, weight="weight"))
        else:
            path = nx.dijkstra_path(graph, src, tgt, weight="weight")
            cost = int(nx.dijkstra_path_length(graph, src, tgt, weight="weight"))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    all_shortest = list(nx.all_shortest_paths(graph, src, tgt, weight="weight"))
    if len(all_shortest) != 1:
        return None
    return path, cost


def generate_sp_graph(
    rng: random.Random,
    num_vertices: int,
    num_edges: int,
    directed: bool,
    src: int,
    tgt: int,
    use_bellman_ford: bool,
) -> tuple[nx.Graph | nx.DiGraph, list[int], int]:
    if directed:
        all_possible = [(u, v) for u in range(num_vertices) for v in range(num_vertices) if u != v]
    else:
        all_possible = [(u, v) for u in range(num_vertices) for v in range(u + 1, num_vertices)]
    if num_edges > len(all_possible):
        raise ValueError("Requested edges exceed possible graph edges.")

    for _ in range(4000):
        g = nx.DiGraph() if directed else nx.Graph()
        g.add_nodes_from(range(num_vertices))

        intermediates = [x for x in range(num_vertices) if x not in {src, tgt}]
        rng.shuffle(intermediates)
        k = min(len(intermediates), rng.randint(1, 4))
        path_nodes = [src] + intermediates[:k] + [tgt]
        forced_edges = list(zip(path_nodes[:-1], path_nodes[1:]))

        edge_set = set(forced_edges)
        candidates = [e for e in all_possible if e not in edge_set]
        rng.shuffle(candidates)
        edge_set.update(candidates[: max(0, num_edges - len(edge_set))])
        if len(edge_set) != num_edges:
            continue

        for u, v in edge_set:
            if (u, v) in forced_edges:
                w = rng.randint(1, 4)
            else:
                w = rng.randint(5, 18)
            g.add_edge(u, v, weight=w)

        solved = shortest_path_unique(g, src, tgt, use_bellman_ford=use_bellman_ford)
        if solved is None:
            continue
        path, cost = solved
        return g, path, cost

    raise RuntimeError("Failed to generate SP graph with unique shortest path.")


def render_sp_text_from_canonical(
    canonical_text: str, edges: list[tuple[int, int, int]], directed: bool
) -> str:
    suffix_match = re.search(
        r"\n\nWhat is the fastest route[\s\S]*$",
        canonical_text,
        flags=re.IGNORECASE,
    )
    suffix = suffix_match.group(0) if suffix_match else (
        "\n\nReply with only the path as a sequence of nodes and the total cost.\n"
        "Format: Path: X -> X -> X, Cost: X"
    )
    if "are:" in canonical_text:
        prefix = canonical_text.split("are:", 1)[0] + "are:"
    else:
        prefix = (
            "A route-planning task over weighted "
            + ("one-way" if directed else "two-way")
            + " roads. Edge list and travel times are:"
        )

    lines = [f"- {u} {'->' if directed else '--'} {v}: {w}" for u, v, w in edges]
    return prefix + "\n\n" + "\n".join(lines) + suffix


def extract_bool_param(difficulty_params: str, key: str) -> bool:
    kv = parse_kv_blob(difficulty_params)
    value = kv.get(key, "").strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"Missing or invalid boolean {key} in difficulty_params.")
    return value == "true"


def greedy_wis(graph: nx.Graph, weights: dict[int, int]) -> tuple[list[int], int]:
    remaining = set(graph.nodes())
    chosen: list[int] = []
    while remaining:
        best = max(remaining, key=lambda n: (weights[n], -n))
        chosen.append(best)
        remaining -= {best}
        remaining -= set(graph.neighbors(best))
    chosen.sort()
    total = sum(weights[n] for n in chosen)
    return chosen, total


def wis_path_dp(weights: dict[int, int]) -> tuple[list[int], int]:
    nodes = sorted(weights)
    n = len(nodes)
    dp = [0] * (n + 1)
    take = [False] * n
    for i in range(1, n + 1):
        w = weights[nodes[i - 1]]
        include = w + (dp[i - 2] if i >= 2 else 0)
        exclude = dp[i - 1]
        if include >= exclude:
            dp[i] = include
            take[i - 1] = True
        else:
            dp[i] = exclude
    chosen: list[int] = []
    i = n
    while i > 0:
        if take[i - 1]:
            chosen.append(nodes[i - 1])
            i -= 2
        else:
            i -= 1
    chosen.sort()
    return chosen, dp[n]


def wis_tree_dp(graph: nx.Graph, weights: dict[int, int]) -> tuple[list[int], int]:
    root = min(graph.nodes())
    parent: dict[int, int | None] = {root: None}
    order = [root]
    for u in order:
        for v in graph.neighbors(u):
            if v == parent[u]:
                continue
            parent[v] = u
            order.append(v)
    post = list(reversed(order))
    dp_take: dict[int, int] = {}
    dp_skip: dict[int, int] = {}
    for u in post:
        take_u = weights[u]
        skip_u = 0
        for v in graph.neighbors(u):
            if parent.get(v) != u:
                continue
            take_u += dp_skip[v]
            skip_u += max(dp_take[v], dp_skip[v])
        dp_take[u] = take_u
        dp_skip[u] = skip_u

    chosen: set[int] = set()

    def recover(u: int, parent_taken: bool) -> None:
        take_u = False
        if not parent_taken and dp_take[u] >= dp_skip[u]:
            take_u = True
            chosen.add(u)
        for v in graph.neighbors(u):
            if parent.get(v) == u:
                recover(v, take_u)

    recover(root, False)
    selected = sorted(chosen)
    return selected, sum(weights[n] for n in selected)


def wis_general_bruteforce(graph: nx.Graph, weights: dict[int, int]) -> tuple[list[int], int]:
    nodes = sorted(graph.nodes())
    n = len(nodes)
    best_weight = -1
    best_set: list[int] = []
    for mask in range(1 << n):
        selected = [nodes[i] for i in range(n) if (mask >> i) & 1]
        ok = True
        for i, u in enumerate(selected):
            for v in selected[i + 1 :]:
                if graph.has_edge(u, v):
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        w = sum(weights[x] for x in selected)
        if w > best_weight:
            best_weight = w
            best_set = selected
    return best_set, best_weight


def generate_wis_graph(
    rng: random.Random,
    n: int,
    graph_type: str,
    greedy_succeeds: bool,
    num_edges_hint: int | None,
) -> tuple[nx.Graph, dict[int, int], list[int], int]:
    for _ in range(6000):
        weights = {i: rng.randint(4, 18) for i in range(n)}
        g = nx.Graph()
        g.add_nodes_from(range(n))

        if graph_type == "path":
            g.add_edges_from((i, i + 1) for i in range(n - 1))
            optimal_set, optimal_weight = wis_path_dp(weights)
        elif graph_type == "tree":
            tree = nx.random_labeled_tree(n, seed=rng.randint(0, 10**9))
            g.add_edges_from(tree.edges())
            optimal_set, optimal_weight = wis_tree_dp(g, weights)
        else:
            edge_target = num_edges_hint if num_edges_hint is not None else max(n, int(1.8 * n))
            all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
            rng.shuffle(all_edges)
            g.add_edges_from(all_edges[: min(edge_target, len(all_edges))])
            optimal_set, optimal_weight = wis_general_bruteforce(g, weights)

        _, greedy_weight = greedy_wis(g, weights)
        if (greedy_weight == optimal_weight) != greedy_succeeds:
            continue
        return g, weights, optimal_set, optimal_weight

    raise RuntimeError("Failed to generate WIS graph matching greedy_succeeds.")


def render_wis_text_from_canonical(
    canonical_text: str, weights: dict[int, int], edges: list[tuple[int, int]]
) -> str:
    weight_block = "\n".join(f"Node {i}: {weights[i]}" for i in sorted(weights))
    edge_block = ", ".join(f"{u}-{v}" for u, v in edges)

    direct_idx = canonical_text.find("Direct connections")
    which_idx = canonical_text.find("\n\nWhich ")
    if direct_idx != -1 and which_idx != -1 and which_idx > direct_idx:
        head = canonical_text[:direct_idx]
        tail = canonical_text[which_idx:]
        if ":\n" in head:
            head = re.sub(r":\n[\s\S]*$", ":\n" + weight_block, head, count=1)
        else:
            head = head.rstrip() + "\n\nNode weights:\n" + weight_block
        middle = "Direct connections (cannot both be selected):\n" + edge_block
        return head + "\n\n" + middle + tail

    return (
        f"{canonical_text.strip()}\n\n"
        f"Node weights:\n{weight_block}\n\n"
        f"Direct connections (cannot both be selected):\n{edge_block}\n"
    )


def format_sp_answer(path: Sequence[int], cost: int) -> str:
    return f"Path: {' -> '.join(str(x) for x in path)}, Cost: {cost}"


def format_wis_answer(selected: Sequence[int], total: int) -> str:
    return f"Selected: {{{', '.join(str(x) for x in selected)}}}, Total: {total}"


def main() -> None:
    rng = random.Random(SEED)

    if not TARGET_BANK.exists():
        raise FileNotFoundError(f"Missing target CSV: {TARGET_BANK}")

    target_df = pd.read_csv(TARGET_BANK)
    source_df = target_df
    source_canon = source_df[
        (source_df["variant_type"] == "canonical")
        & source_df["problem_id"].astype(str).str.match(r"^(CC|SP|WIS)_")
    ]
    if source_canon.empty:
        if not SOURCE_FALLBACK_BANK.exists():
            raise ValueError(
                "No algorithmic canonical rows in question_bank.csv and fallback missing."
            )
        source_df = pd.read_csv(SOURCE_FALLBACK_BANK)
        source_canon = source_df[
            (source_df["variant_type"] == "canonical")
            & source_df["problem_id"].astype(str).str.match(r"^(CC|SP|WIS)_")
        ]
        if source_canon.empty:
            raise ValueError("Fallback file has no algorithmic canonical rows either.")

    existing_w6_ids = set(
        target_df.loc[target_df["variant_type"] == "W6", "problem_id"].astype(str).tolist()
    )

    generated_rows: list[dict] = []
    skipped_existing = 0

    for _, row in source_canon.sort_values("problem_id").iterrows():
        pid = str(row["problem_id"])
        if pid in existing_w6_ids:
            skipped_existing += 1
            continue

        text = str(row["problem_text"])
        answer = str(row["correct_answer"])
        verifier = str(row["verifier_function"])
        dparams = str(row.get("difficulty_params", ""))

        if pid.startswith("CC_"):
            denoms, target = parse_cc_instance(text)
            original_opt = parse_first_int(answer, r"Count:\s*(\d+)")
            has_one_original = 1 in denoms
            k = len(denoms)
            lo = 2 if not has_one_original else 1
            hi = max(30, max(denoms) + 30)

            solved = None
            for _ in range(8000):
                if has_one_original:
                    pool = [1] + rng.sample(range(2, hi + 1), k - 1)
                else:
                    pool = rng.sample(range(lo, hi + 1), k)
                new_denoms = sorted(set(pool))
                if len(new_denoms) != k:
                    continue
                if not has_one_original and 1 in new_denoms:
                    continue
                new_target = max(2, target + rng.randint(-max(2, target // 4), max(2, target // 4)))
                dp = dp_coin_change(new_denoms, new_target)
                if dp is None:
                    continue
                dp_count, dp_coins = dp
                greedy = greedy_coin_change(new_denoms, new_target)
                greedy_count = greedy[0] if greedy is not None else 10**9
                if greedy_count == dp_count:
                    continue
                if abs(dp_count - original_opt) > 1:
                    continue
                solved = (new_denoms, new_target, dp_count, dp_coins)
                break
            if solved is None:
                raise RuntimeError(f"Could not generate valid CC W6 for {pid}")
            new_denoms, new_target, dp_count, dp_coins = solved
            new_text = render_cc_text_from_canonical(text, new_denoms, new_target)
            new_answer = format_cc_answer(dp_count, dp_coins)

        elif pid.startswith("SP_"):
            kv = parse_kv_blob(dparams)
            n = int(kv.get("num_vertices", "0"))
            m = int(kv.get("num_edges", "0"))
            directed = kv.get("directed", "").lower() == "true"
            if n <= 0 or m <= 0:
                raise ValueError(f"Invalid SP difficulty_params for {pid}: {dparams}")
            src, tgt = parse_src_tgt_from_sp_text(text, answer)
            use_bf = "bellman" in verifier.lower()
            graph, path, cost = generate_sp_graph(
                rng, n, m, directed, src, tgt, use_bellman_ford=use_bf
            )
            edge_triplets = [
                (u, v, int(data["weight"])) for u, v, data in sorted(graph.edges(data=True))
            ]
            new_text = render_sp_text_from_canonical(text, edge_triplets, directed)
            new_answer = format_sp_answer(path, cost)

        elif pid.startswith("WIS_"):
            kv = parse_kv_blob(dparams)
            n = int(kv.get("num_nodes", kv.get("num_items", "0")))
            if n <= 0:
                raise ValueError(f"Invalid WIS difficulty_params for {pid}: {dparams}")
            graph_type = kv.get("graph_type", "general").strip().lower()
            greedy_succeeds = extract_bool_param(dparams, "greedy_succeeds")
            num_edges_hint = int(kv["num_edges"]) if "num_edges" in kv else None
            graph, weights, optimal_set, optimal_weight = generate_wis_graph(
                rng,
                n,
                graph_type=graph_type,
                greedy_succeeds=greedy_succeeds,
                num_edges_hint=num_edges_hint,
            )
            edges = sorted((min(u, v), max(u, v)) for u, v in graph.edges())
            new_text = render_wis_text_from_canonical(text, weights, edges)
            new_answer = format_wis_answer(optimal_set, optimal_weight)

        else:
            continue

        new_row = row.to_dict()
        new_row["variant_type"] = "W6"
        new_row["problem_text"] = new_text
        new_row["correct_answer"] = new_answer
        new_row["contamination_pole"] = "Low"
        new_row["source"] = "procedural"
        new_row["verifier_function"] = row["verifier_function"]
        new_row["problem_family"] = row["problem_family"]
        new_row["difficulty"] = row["difficulty"]
        generated_rows.append(new_row)

    if not generated_rows:
        print("No new W6 rows generated.")
        print(f"Skipped existing W6 rows: {skipped_existing}")
        return

    append_df = pd.DataFrame(generated_rows)
    append_df = append_df.reindex(columns=target_df.columns, fill_value="")
    final_df = pd.concat([target_df, append_df], ignore_index=True)
    final_df.to_csv(TARGET_BANK, index=False)

    print(f"Seed: {SEED}")
    print(f"Source canonical rows: {len(source_canon)}")
    print(f"Generated W6 rows: {len(generated_rows)}")
    print(f"Skipped existing W6 rows: {skipped_existing}")
    print(f"Appended to: {TARGET_BANK}")
    by_prefix: dict[str, int] = {"CC": 0, "SP": 0, "WIS": 0}
    for r in generated_rows:
        prefix = str(r["problem_id"]).split("_", 1)[0]
        if prefix in by_prefix:
            by_prefix[prefix] += 1
    print(f"Breakdown: CC={by_prefix['CC']}, SP={by_prefix['SP']}, WIS={by_prefix['WIS']}")


if __name__ == "__main__":
    main()
