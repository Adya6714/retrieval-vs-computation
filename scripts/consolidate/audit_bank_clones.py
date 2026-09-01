#!/usr/bin/env python3
"""H4: Clone-detect canonical problems in all three banks.

A clone edge is token Jaccard >= NEAR_DUP_JACCARD or SequenceMatcher ratio
>= NEAR_DUP_RATIO (same floors as the W6 transform audit) **and** identical
gold (WIS Selected-set / Cost token, else normalized answer). WIS_017–020
share Selected: {4, 5} and still cluster.

Does not write results/raw/.
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DERIVED = REPO_ROOT / "results/derived"
OUT = DERIVED / "bank_clone_audit.csv"
SUMMARY = DERIVED / "bank_clone_audit_summary.csv"

BANKS = {
    "ALGO": REPO_ROOT / "data/problems/question_bank_algo.csv",
    "BW": REPO_ROOT / "data/problems/question_bank_bw.csv",
    "GSM": REPO_ROOT / "data/problems/question_bank_gsm.csv",
}

NEAR_DUP_JACCARD = 0.85
NEAR_DUP_RATIO = 0.90


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip())


def _tokens(text: str) -> list[str]:
    return _norm_ws(text).split()


def token_jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokens(a)), set(_tokens(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def char_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm_ws(a), _norm_ws(b)).ratio()


def _gold_key(pid: str, gold: str) -> str:
    s = _norm_ws(gold).lower()
    pid_u = str(pid).upper()
    if pid_u.startswith("WIS"):
        m = re.search(r"selected\s*:\s*\{([^}]*)\}", s, flags=re.I)
        if m:
            return "selected:{" + ",".join(sorted(x.strip() for x in m.group(1).split(",") if x.strip())) + "}"
        m2 = re.search(r"total\s*:\s*(-?\d+)", s, flags=re.I)
        if m2:
            return f"total:{m2.group(1)}"
    if pid_u.startswith(("SP", "CC")):
        m = re.search(r"(?:cost|count|total)\s*:\s*(-?\d+)", s, flags=re.I)
        if m:
            return m.group(1)
    return s


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {x: x for x in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict] = []
    summary: list[dict] = []

    for fam, path in BANKS.items():
        df = pd.read_csv(path, dtype=str).fillna("")
        can = df[df["variant_type"].astype(str).str.strip().str.lower() == "canonical"].copy()
        can["problem_id"] = can["problem_id"].astype(str).str.strip()
        pids = can["problem_id"].tolist()
        texts = dict(zip(can["problem_id"], can["problem_text"].astype(str)))
        golds = dict(zip(can["problem_id"], can["correct_answer"].astype(str)))
        uf = UnionFind(pids)
        edges = 0
        for i, a in enumerate(pids):
            for b in pids[i + 1 :]:
                jac = token_jaccard(texts[a], texts[b])
                rat = char_ratio(texts[a], texts[b])
                if jac >= NEAR_DUP_JACCARD or rat >= NEAR_DUP_RATIO:
                    same_gold = _gold_key(a, golds[a]) == _gold_key(b, golds[b])
                    if same_gold:
                        uf.union(a, b)
                        edges += 1
        families: dict[str, list[str]] = defaultdict(list)
        for pid in pids:
            families[uf.find(pid)].append(pid)
        clone_fams = {k: sorted(v) for k, v in families.items() if len(v) > 1}
        n_in_clones = sum(len(v) for v in clone_fams.values())
        n_singleton = len(pids) - n_in_clones
        effective_n = n_singleton + len(clone_fams)
        fam_id_map = {}
        for i, (root, members) in enumerate(sorted(clone_fams.items()), start=1):
            cid = f"{fam}_CLONE_{i:03d}"
            for pid in members:
                fam_id_map[pid] = cid
        for pid in pids:
            cid = fam_id_map.get(pid, "")
            members = clone_fams.get(uf.find(pid), [pid])
            peer_golds = [_gold_key(p, golds[p]) for p in members]
            rows_out.append(
                {
                    "family": fam,
                    "problem_id": pid,
                    "clone_family_id": cid,
                    "clone_family_size": len(members) if cid else 1,
                    "clone_members": ",".join(members) if cid else "",
                    "gold_key": _gold_key(pid, golds[pid]),
                    "identical_gold_key_within_family": str(
                        bool(cid) and len(set(peer_golds)) == 1
                    ),
                    "near_dup_jaccard_floor": NEAR_DUP_JACCARD,
                    "near_dup_ratio_floor": NEAR_DUP_RATIO,
                }
            )
        summary.append(
            {
                "family": fam,
                "n_canonical": len(pids),
                "n_clone_families": len(clone_fams),
                "n_problems_in_clones": n_in_clones,
                "effective_n_collapsing_clones": effective_n,
                "n_pairwise_edges": edges,
                "materially_below_110": str(fam == "ALGO" and effective_n < 100),
            }
        )

    pd.DataFrame(rows_out).to_csv(OUT, index=False)
    pd.DataFrame(summary).to_csv(SUMMARY, index=False)
    print(f"Wrote {OUT} ({len(rows_out)} rows)")
    print(f"Wrote {SUMMARY}")
    print(pd.DataFrame(summary).to_string(index=False))
    clones = [r for r in rows_out if r["clone_family_id"]]
    if clones:
        print("clone families:")
        seen = set()
        for r in clones:
            if r["clone_family_id"] in seen:
                continue
            seen.add(r["clone_family_id"])
            print(f"  {r['clone_family_id']} n={r['clone_family_size']} {r['clone_members']}")


if __name__ == "__main__":
    main()
