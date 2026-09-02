"""ALGO clone-family IDs for cluster bootstrap."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CLONE_AUDIT = REPO_ROOT / "results" / "derived" / "bank_clone_audit.csv"


@lru_cache(maxsize=1)
def algo_cluster_map() -> dict[str, str]:
    """problem_id → clone_family_id; singletons get SINGLETON_{pid}."""
    if not CLONE_AUDIT.exists():
        return {}
    df = pd.read_csv(CLONE_AUDIT, dtype=str).fillna("")
    df = df[df["family"].astype(str).str.strip().str.upper() == "ALGO"]
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        pid = str(r["problem_id"]).strip()
        cid = str(r.get("clone_family_id") or "").strip()
        out[pid] = cid if cid else f"SINGLETON_{pid}"
    return out


def cluster_ids_for(problem_ids: list[str]) -> list[str]:
    cmap = algo_cluster_map()
    return [cmap.get(str(p), f"SINGLETON_{p}") for p in problem_ids]
