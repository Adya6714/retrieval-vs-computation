#!/usr/bin/env python3
"""L2: discriminant construct validity for P1 phi (not convergent MTMM).

Retention and phi are computed from the same canonical/W3 contingency table;
their correlation is algebraic, not empirical evidence of convergent validity.
This script reports discriminant validity only: phi vs canonical accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.behavioral.retention import MIN_CANONICAL_FOR_RETENTION  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
PHI_IN = DER / "P1_phi_canonical_w3.csv"
OUT = DER / "P1_construct_validity.csv"

N_BOOT = 5000
SEED = 42


def _cluster_bootstrap_spearman(df: pd.DataFrame, x: str, y: str) -> tuple[float, float, float]:
    sub = df[[x, y, "family"]].dropna()
    if len(sub) < 3:
        return float("nan"), float("nan"), float("nan")
    rho, _ = stats.spearmanr(sub[x], sub[y])
    fams = sorted(sub["family"].unique())
    grouped = {f: sub[sub["family"] == f] for f in fams}
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        draw = rng.choice(fams, size=len(fams), replace=True)
        chunk = pd.concat([grouped[f] for f in draw], ignore_index=True)
        if chunk[x].nunique() < 2 or chunk[y].nunique() < 2:
            boots[i] = float("nan")
        else:
            boots[i], _ = stats.spearmanr(chunk[x], chunk[y])
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return float(rho), float("nan"), float("nan")
    return float(rho), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    if not PHI_IN.exists():
        raise FileNotFoundError(f"Run p1_phi_canonical_w3.py first: {PHI_IN}")
    df = pd.read_csv(PHI_IN, dtype=str).fillna("")
    for col in ["phi", "retention_w3", "acc_canonical"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["acc_canonical"] >= MIN_CANONICAL_FOR_RETENTION].copy()

    rows: list[dict] = []
    for row_name, col_name, label in [
        ("phi", "acc_canonical", "discriminant_phi_vs_canonical_accuracy"),
        ("retention_w3", "acc_canonical", "discriminant_retention_vs_canonical_accuracy"),
    ]:
        sub = df[[row_name, col_name, "family"]].dropna()
        n = len(sub)
        if n < 3:
            rho, lo, hi, p = float("nan"), float("nan"), float("nan"), float("nan")
        else:
            rho, lo, hi = _cluster_bootstrap_spearman(df, row_name, col_name)
            _, p = stats.spearmanr(sub[row_name], sub[col_name])
        rows.append(
            {
                "analysis": label,
                "row_construct": row_name.replace("_w3", ""),
                "col_construct": "canonical_accuracy",
                "spearman_rho": round(rho, 3) if rho == rho else "",
                "ci_low": round(lo, 3) if lo == lo else "",
                "ci_high": round(hi, 3) if hi == hi else "",
                "p_value": round(float(p), 3) if p == p else "",
                "n_cells": n,
                "can_acc_floor": MIN_CANONICAL_FOR_RETENTION,
                "bootstrap": "cluster_by_family",
                "n_boot": N_BOOT,
                "seed": SEED,
                "note": (
                    "retention-vs-phi omitted — same contingency table, not independent evidence"
                    if row_name == "phi"
                    else ""
                ),
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(out)} rows)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
