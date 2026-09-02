#!/usr/bin/env python3
"""K2: Phi (canonical vs W3) with bootstrap CIs and construct correlations."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.clones import cluster_ids_for  # noqa: E402
from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
OUT = DER / "P1_phi_canonical_w3.csv"
CORR_OUT = DER / "P1_phi_construct_correlations.csv"

PAPER_MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "google/gemini-2.5-flash": "Gemini",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
N_BOOT = 5000
SEED = 42
CAN_ACC_FLOOR = 0.30


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _phi(a: int, b: int, c: int, d: int) -> float:
    denom = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denom == 0:
        return float("nan")
    return (a * d - b * c) / denom


def _cells_from_pairs(can_fail: list[bool], w3_fail: list[bool]) -> tuple[int, int, int, int]:
    a = b = c = d = 0
    for cf, wf in zip(can_fail, w3_fail):
        if not cf and not wf:
            a += 1
        elif not cf and wf:
            b += 1
        elif cf and not wf:
            c += 1
        else:
            d += 1
    return a, b, c, d


def _load_included() -> pd.DataFrame:
    parts = []
    for path in sorted(DER.glob("*_P1_*rescored.csv")):
        if "review" in path.name.lower():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "included" not in df.columns:
            continue
        df = df[_is_true(df["included"])].copy()
        if path.name.startswith("ALGO_"):
            fam = "ALGO"
        elif path.name.startswith("GSM_"):
            fam = "GSM"
        elif path.name.startswith("BW_"):
            fam = "BW"
        else:
            continue
        df = filter_excluded(df, family=fam)
        df["family"] = fam
        df["model_short"] = df["model"].map(PAPER_MODELS)
        df = df[df["model_short"].notna()].copy()
        df["variant"] = df["variant_type"].map(normalize_variant)
        ok = df["rescored_correct"] if "rescored_correct" in df.columns else df.get("verified", "")
        df["ok"] = _is_true(ok)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["family", "problem_id", "variant", "model_short"], keep="last")


def _bootstrap_phi(
    can_fail: list[bool],
    w3_fail: list[bool],
    cluster_ids: list[str] | None,
) -> tuple[float, float]:
    n = len(can_fail)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(SEED)
    phis = np.empty(N_BOOT, dtype=float)
    if cluster_ids is None:
        idx = np.arange(n)
        for i in range(N_BOOT):
            draw = rng.choice(idx, size=n, replace=True)
            cf = [can_fail[j] for j in draw]
            wf = [w3_fail[j] for j in draw]
            phis[i] = _phi(*_cells_from_pairs(cf, wf))
    else:
        grouped: dict[str, list[int]] = defaultdict(list)
        for j, c in enumerate(cluster_ids):
            grouped[str(c)].append(j)
        fams = list(grouped.keys())
        n_f = len(fams)
        for i in range(N_BOOT):
            draw = rng.integers(0, n_f, size=n_f)
            idx: list[int] = []
            for j in draw:
                idx.extend(grouped[fams[j]])
            cf = [can_fail[j] for j in idx]
            wf = [w3_fail[j] for j in idx]
            phis[i] = _phi(*_cells_from_pairs(cf, wf))
    return (float(np.percentile(phis, 2.5)), float(np.percentile(phis, 97.5)))


def main() -> None:
    p1 = _load_included()
    can = p1[p1["variant"] == "canonical"][["family", "problem_id", "model_short", "ok"]].rename(
        columns={"ok": "ok_can"}
    )
    w3 = p1[p1["variant"] == "W3"][["family", "problem_id", "model_short", "ok"]].rename(
        columns={"ok": "ok_w3"}
    )
    merged = can.merge(w3, on=["family", "problem_id", "model_short"], how="inner")
    rows: list[dict] = []
    construct: list[dict] = []

    for (fam, model), g in merged.groupby(["family", "model_short"]):
        can_fail = (~g["ok_can"]).tolist()
        w3_fail = (~g["ok_w3"]).tolist()
        pids = g["problem_id"].astype(str).tolist()
        a, b, c, d = _cells_from_pairs(can_fail, w3_fail)
        phi = _phi(a, b, c, d)
        clusters = cluster_ids_for(pids) if fam == "ALGO" else None
        lo, hi = _bootstrap_phi(can_fail, w3_fail, clusters)
        n_clusters = len(set(clusters)) if clusters else len(pids)
        acc_can = float(g["ok_can"].mean())
        acc_w3 = float(g["ok_w3"].mean())
        retention = acc_w3 / acc_can if acc_can > 0 else float("nan")
        rows.append(
            {
                "family": fam,
                "model": model,
                "phi": round(phi, 4) if phi == phi else "",
                "phi_ci_low": round(lo, 4) if lo == lo else "",
                "phi_ci_high": round(hi, 4) if hi == hi else "",
                "n": len(g),
                "n_clusters": n_clusters,
                "cell_ok_ok": a,
                "cell_ok_fail": b,
                "cell_fail_ok": c,
                "cell_fail_fail": d,
                "acc_canonical": round(acc_can, 4),
                "acc_w3": round(acc_w3, 4),
                "retention_w3": round(retention, 4) if retention == retention else "",
            }
        )
        if acc_can >= CAN_ACC_FLOOR:
            construct.append(
                {
                    "family": fam,
                    "model": model,
                    "phi": phi,
                    "retention_w3": retention,
                    "acc_canonical": acc_can,
                    "acc_w3": acc_w3,
                }
            )

    out_df = pd.DataFrame(rows).sort_values(["family", "model"])
    out_df.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(out_df)} rows)")

    cd = pd.DataFrame(construct)
    corr_rows = []
    if len(cd) >= 3:
        for x, y, label in [
            ("phi", "retention_w3", "phi_vs_retention"),
            ("phi", "acc_canonical", "phi_vs_accuracy"),
            ("retention_w3", "acc_canonical", "retention_vs_accuracy"),
        ]:
            sub = cd[[x, y]].dropna()
            if len(sub) < 3:
                rho, p = float("nan"), float("nan")
            else:
                rho, p = stats.spearmanr(sub[x], sub[y])
            corr_rows.append(
                {
                    "pair": label,
                    "spearman_rho": round(float(rho), 3) if rho == rho else "",
                    "p_value": round(float(p), 3) if p == p else "",
                    "n_cells": len(sub),
                    "can_acc_floor": CAN_ACC_FLOOR,
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(CORR_OUT, index=False)
    print(f"Wrote {CORR_OUT}")
    print(corr_df.to_string(index=False))


if __name__ == "__main__":
    main()
