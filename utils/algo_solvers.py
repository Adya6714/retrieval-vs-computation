from __future__ import annotations

import heapq
from bisect import bisect_right
from typing import Iterable


def coin_change_greedy(denominations: Iterable[int], target: int) -> tuple[int, list[int]] | None:
    denoms = sorted({int(x) for x in denominations if int(x) > 0}, reverse=True)
    remaining = int(target)
    picked: list[int] = []
    for coin in denoms:
        while remaining >= coin:
            picked.append(coin)
            remaining -= coin
    if remaining != 0:
        return None
    return len(picked), picked


def coin_change_dp(denominations: Iterable[int], target: int) -> tuple[int, list[int]] | None:
    denoms = sorted({int(x) for x in denominations if int(x) > 0})
    t = int(target)
    inf = 10**9
    dp = [inf] * (t + 1)
    parent = [-1] * (t + 1)
    dp[0] = 0
    for amt in range(1, t + 1):
        for coin in denoms:
            if coin > amt:
                break
            cand = dp[amt - coin] + 1
            if cand < dp[amt]:
                dp[amt] = cand
                parent[amt] = coin
    if dp[t] >= inf:
        return None
    coins: list[int] = []
    cur = t
    while cur > 0:
        coin = parent[cur]
        if coin < 0:
            return None
        coins.append(coin)
        cur -= coin
    coins.sort(reverse=True)
    return dp[t], coins


def dijkstra_shortest_path(
    num_nodes: int, edges: list[tuple[int, int, int]], source: int, target: int
) -> tuple[list[int], int] | None:
    n = int(num_nodes)
    src = int(source)
    tgt = int(target)
    graph: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[int(u)].append((int(v), int(w)))
    dist = [10**15] * n
    parent = [-1] * n
    dist[src] = 0
    heap: list[tuple[int, int]] = [(0, src)]
    while heap:
        d, node = heapq.heappop(heap)
        if d != dist[node]:
            continue
        if node == tgt:
            break
        for nxt, w in graph[node]:
            nd = d + w
            if nd < dist[nxt]:
                dist[nxt] = nd
                parent[nxt] = node
                heapq.heappush(heap, (nd, nxt))
    if dist[tgt] >= 10**14:
        return None
    path: list[int] = []
    cur = tgt
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path, int(dist[tgt])


def nearest_neighbor_greedy(
    num_nodes: int, edges: list[tuple[int, int, int]], source: int, target: int
) -> tuple[list[int], int] | None:
    n = int(num_nodes)
    src = int(source)
    tgt = int(target)
    graph: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[int(u)].append((int(v), int(w)))
    for node in range(n):
        graph[node].sort(key=lambda x: (x[1], x[0]))
    path = [src]
    visited = {src}
    total = 0
    cur = src
    while cur != tgt:
        candidates = [(nbr, wt) for nbr, wt in graph[cur] if nbr not in visited]
        if not candidates:
            return None
        nxt, wt = candidates[0]
        path.append(nxt)
        visited.add(nxt)
        total += wt
        cur = nxt
    return path, total


def wis_edf_greedy(intervals: list[dict[str, int]]) -> tuple[list[int], int]:
    ordered = sorted(
        intervals,
        key=lambda it: (int(it["end"]), -int(it["weight"]), int(it["start"]), int(it["id"])),
    )
    selected: list[dict[str, int]] = []
    current_end = -10**9
    for item in ordered:
        if int(item["start"]) >= current_end:
            selected.append(item)
            current_end = int(item["end"])
    ids = [int(it["id"]) for it in selected]
    total = sum(int(it["weight"]) for it in selected)
    return ids, total


def wis_interval_dp(intervals: list[dict[str, int]]) -> tuple[list[int], int]:
    ordered = sorted(intervals, key=lambda it: (int(it["end"]), int(it["start"]), int(it["id"])))
    ends = [int(it["end"]) for it in ordered]
    p: list[int] = []
    for i, cur in enumerate(ordered):
        j = bisect_right(ends, int(cur["start"])) - 1
        p.append(j if j < i else -1)

    n = len(ordered)
    dp = [0] * (n + 1)
    take = [False] * n
    for i in range(1, n + 1):
        w = int(ordered[i - 1]["weight"])
        include = w + dp[p[i - 1] + 1]
        exclude = dp[i - 1]
        if include >= exclude:
            dp[i] = include
            take[i - 1] = True
        else:
            dp[i] = exclude
    selected: list[int] = []
    i = n
    while i > 0:
        if take[i - 1]:
            selected.append(int(ordered[i - 1]["id"]))
            i = p[i - 1] + 1
        else:
            i -= 1
    selected.sort()
    return selected, int(dp[n])
