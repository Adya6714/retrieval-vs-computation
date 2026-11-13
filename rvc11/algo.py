"""Shortest path, max-weight independent/compatible set, coin change: specs, solvers, verifiers.

Every verifier checks the structure of the answer against the formal spec (never against a gold
string), so alternative optimal answers are accepted and the rule is identical for every variant.
"""

from __future__ import annotations

import math
import re
from functools import lru_cache

NUM = r"-?\d+(?:\.\d+)?"


# ---------------------------------------------------------------- shortest path
def sp_optimum(spec: dict) -> float:
    """Bellman-Ford. Handles negative weights; raises on negative cycles or no path."""
    nodes = {spec["source"], spec["target"]}
    edges = []
    for u, v, w in spec["edges"]:
        nodes |= {u, v}
        edges.append((u, v, w))
        if not spec.get("directed", True):
            edges.append((v, u, w))
    dist = {n: math.inf for n in nodes}
    dist[spec["source"]] = 0
    for _ in range(len(nodes) - 1):
        changed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True
        if not changed:
            break
    else:
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                raise ValueError("negative cycle")
    if math.isinf(dist[spec["target"]]):
        raise ValueError("no path")
    return dist[spec["target"]]


def _alias_table(labels: dict) -> dict:
    """Map every unambiguous way of naming a node/item to its id."""
    cand: dict[str, set] = {}
    for key, value in labels.items():
        idx = int(key)
        for label in value if isinstance(value, list) else [value]:
            lab = re.sub(r"\s+", " ", str(label)).strip().lower()
            forms = {lab}
            toks = lab.split()
            if len(toks) > 1:
                forms.add(toks[-1])
            for f in forms:
                cand.setdefault(f, set()).add(idx)
    return {k: next(iter(v)) for k, v in cand.items() if len(v) == 1}


def _resolve(token: str, aliases: dict):
    t = re.sub(r"\s+", " ", str(token)).strip().strip("`*'\"[]{}()").strip().lower()
    t = re.sub(r"^(?:node|vertex)\s+", "", t)
    if t in aliases:
        return aliases[t]
    last = t.split()[-1] if t.split() else t
    return aliases.get(last)


def verify_sp(block: str, spec: dict) -> tuple[bool, str, str]:
    text = str(block).replace("→", "->").replace("⟶", "->").replace("=>", "->")
    m_path = re.findall(r"path\s*[:=]\s*(.+)", text, flags=re.IGNORECASE)
    m_cost = re.findall(rf"(?:total\s+)?cost\s*[:=]\s*\$?({NUM})", text, flags=re.IGNORECASE)
    if not m_path or not m_cost:
        return False, "format_fail", "need both 'Path:' and 'Cost:'"
    raw = re.split(r",\s*(?:total\s+)?cost\b", m_path[-1], flags=re.IGNORECASE)[0]
    toks = [t for t in re.split(r"\s*->\s*", raw.strip().rstrip(".")) if t.strip()]
    if len(toks) < 2:
        toks = [t for t in re.split(r"\s*,\s*", raw.strip().rstrip(".")) if t.strip()]
    aliases = _alias_table(spec["labels"])
    nodes = [_resolve(t, aliases) for t in toks]
    if len(nodes) < 2 or any(n is None for n in nodes):
        return False, "format_fail", f"unrecognised node name in path: {raw.strip()[:80]!r}"
    w = {}
    for u, v, c in spec["edges"]:
        w[(u, v)] = min(c, w.get((u, v), math.inf))
        if not spec.get("directed", True):
            w[(v, u)] = min(c, w.get((v, u), math.inf))
    if nodes[0] != spec["source"] or nodes[-1] != spec["target"]:
        return False, "parsed", "path does not start at source and end at target"
    total = 0
    for a, b in zip(nodes, nodes[1:]):
        if (a, b) not in w:
            return False, "parsed", f"edge {a}->{b} does not exist"
        total += w[(a, b)]
    best = sp_optimum(spec)
    if abs(total - best) > 1e-9:
        return False, "parsed", f"path cost {total:g} is not optimal ({best:g})"
    if abs(float(m_cost[-1]) - best) > 1e-9:
        return False, "parsed", f"stated cost {m_cost[-1]} differs from path cost {best:g}"
    return True, "parsed", "ok"


# ---------------------------------------------------------------- weighted selection (WIS)
def wis_optimum(spec: dict) -> float:
    ids = sorted(int(k) for k in spec["weights"])
    wt = {int(k): v for k, v in spec["weights"].items()}
    nb = {i: set() for i in ids}
    for a, b in spec["conflicts"]:
        nb[a].add(b)
        nb[b].add(a)

    @lru_cache(maxsize=None)
    def best(rem: frozenset) -> float:
        if not rem:
            return 0
        i = min(rem)
        rest = rem - {i}
        return max(best(rest), wt[i] + best(rest - nb[i])) if wt[i] > 0 else best(rest)

    return best(frozenset(ids))


def verify_wis(block: str, spec: dict) -> tuple[bool, str, str]:
    text = str(block)
    m_sel = re.findall(r"selected\s*[:=]\s*(.+)", text, flags=re.IGNORECASE)
    m_tot = re.findall(rf"total\s*[:=]\s*\$?({NUM})", text, flags=re.IGNORECASE)
    if not m_sel or not m_tot:
        return False, "format_fail", "need both 'Selected:' and 'Total:'"
    raw = re.split(r",?\s*total\s*[:=]", m_sel[-1], flags=re.IGNORECASE)[0].strip().rstrip(".")
    raw = raw.strip("{}[]() ")
    toks = [t for t in re.split(r"\s*[,;]\s*", raw) if t.strip()]
    aliases = _alias_table(spec["labels"])
    chosen = [_resolve(t, aliases) for t in toks]
    if not chosen or any(c is None for c in chosen):
        return False, "format_fail", f"unrecognised item in selection: {raw[:80]!r}"
    if len(set(chosen)) != len(chosen):
        return False, "parsed", "item selected twice"
    sel = set(chosen)
    for a, b in spec["conflicts"]:
        if a in sel and b in sel:
            return False, "parsed", f"items {a} and {b} conflict"
    wt = {int(k): v for k, v in spec["weights"].items()}
    total = sum(wt[i] for i in sel)
    best = wis_optimum(spec)
    if abs(total - best) > 1e-9:
        return False, "parsed", f"selection weight {total:g} is not optimal ({best:g})"
    if abs(float(m_tot[-1]) - best) > 1e-9:
        return False, "parsed", f"stated total {m_tot[-1]} differs from selection weight {best:g}"
    return True, "parsed", "ok"


# ---------------------------------------------------------------- coin change
def cc_optimum(spec: dict) -> int:
    target, denoms = int(spec["target"]), [int(d) for d in spec["denominations"]]
    dp = [0] + [math.inf] * target
    for t in range(1, target + 1):
        for d in denoms:
            if d <= t and dp[t - d] + 1 < dp[t]:
                dp[t] = dp[t - d] + 1
    if math.isinf(dp[target]):
        raise ValueError("target unreachable")
    return int(dp[target])


def verify_cc(block: str, spec: dict) -> tuple[bool, str, str]:
    text = str(block)
    m_cnt = re.findall(r"count\s*[:=]\s*(\d+)", text, flags=re.IGNORECASE)
    m_val = re.findall(r"values\s*[:=]\s*(.+)", text, flags=re.IGNORECASE)
    if not m_cnt or not m_val:
        return False, "format_fail", "need both 'Count:' and 'Values:'"
    vals = [int(x) for x in re.findall(r"\d+", m_val[-1])]
    if not vals:
        return False, "format_fail", "no values listed"
    denoms = {int(d) for d in spec["denominations"]}
    if any(v not in denoms for v in vals):
        return False, "parsed", "value not among the allowed denominations"
    if sum(vals) != int(spec["target"]):
        return False, "parsed", f"values sum to {sum(vals)}, not {spec['target']}"
    best = cc_optimum(spec)
    if len(vals) != best:
        return False, "parsed", f"uses {len(vals)} items; optimum is {best}"
    if int(m_cnt[-1]) != len(vals):
        return False, "parsed", "stated count differs from number of values listed"
    return True, "parsed", "ok"


def render_gold(spec: dict) -> str:
    """A correct final-answer block built from the spec by the solvers (for gold-in-gold-out)."""
    kind = spec["kind"]
    lab = {int(k): (v[0] if isinstance(v, list) else v) for k, v in spec.get("labels", {}).items()}
    if kind == "sp":
        best = sp_optimum(spec)
        # recover one optimal path by DFS over tight edges
        w = {}
        for u, v, c in spec["edges"]:
            w.setdefault(u, []).append((v, c))
            if not spec.get("directed", True):
                w.setdefault(v, []).append((u, c))
        path = _sp_witness(spec, w, best)
        return "Path: " + " -> ".join(str(lab[n]) for n in path) + f"\nCost: {best:g}"
    if kind == "wis":
        ids = sorted(int(k) for k in spec["weights"])
        wt = {int(k): v for k, v in spec["weights"].items()}
        best = wis_optimum(spec)
        conf = {tuple(sorted(c)) for c in spec["conflicts"]}
        for mask in range(1 << len(ids)):
            sel = [ids[i] for i in range(len(ids)) if mask >> i & 1]
            if abs(sum(wt[i] for i in sel) - best) < 1e-9 and not any(
                tuple(sorted((a, b))) in conf for a in sel for b in sel if a < b
            ):
                return "Selected: " + ", ".join(str(lab[i]) for i in sel) + f"\nTotal: {best:g}"
        raise ValueError("no witness")
    if kind == "cc":
        target, denoms = int(spec["target"]), sorted(int(d) for d in spec["denominations"])
        best = cc_optimum(spec)
        dp = {0: []}
        frontier = [0]
        while frontier:
            nxt = []
            for t in frontier:
                for d in denoms:
                    if t + d <= target and t + d not in dp:
                        dp[t + d] = dp[t] + [d]
                        nxt.append(t + d)
            frontier = nxt
        return f"Count: {best}\nValues: " + ", ".join(str(v) for v in dp[target])
    raise ValueError(kind)


def _sp_witness(spec, adj, best):
    target = spec["target"]
    n_nodes = len({spec["source"], target} | {x for e in spec["edges"] for x in e[:2]})
    stack = [(spec["source"], 0, [spec["source"]])]
    while stack:
        node, cost, path = stack.pop()
        if node == target and abs(cost - best) < 1e-9:
            return path
        if len(path) > n_nodes:
            continue
        for v, c in adj.get(node, []):
            if v not in path:
                stack.append((v, cost + c, path + [v]))
    raise ValueError("no witness path")
