#!/usr/bin/env python3
"""L2: discriminant construct validity for P1 phi (not convergent MTMM).

Retention and phi are computed from the same canonical/W3 contingency table;
their correlation is algebraic, not empirical evidence of convergent validity.
This script reports discriminant validity only: phi vs canonical accuracy.

CI and p from the same family-cluster bootstrap (H0: rho=0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.behavioral.retention import MIN_CANONICAL_FOR_RETENTION  # noqa: E402
from probes.common.cluster_inference import cluster_bootstrap_assoc  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
PHI_IN = DER / "P1_phi_canonical_w3.csv"
OUT = DER / "P1_construct_validity.csv"

N_BOOT = 5000
SEED = 42


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
            res = {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                   "p_clustered": float("nan"), "n": n, "n_clusters": 0}
        else:
            res = cluster_bootstrap_assoc(
                sub[row_name],
                sub[col_name],
                sub["family"].astype(str),
                kind="spearman",
                n_boot=N_BOOT,
                seed=SEED,
            )
        rows.append(
            {
                "analysis": label,
                "row_construct": row_name.replace("_w3", ""),
                "col_construct": "canonical_accuracy",
                "spearman_rho": round(res["estimate"], 3) if res["estimate"] == res["estimate"] else "",
                "ci_low": round(res["ci_low"], 3) if res["ci_low"] == res["ci_low"] else "",
                "ci_high": round(res["ci_high"], 3) if res["ci_high"] == res["ci_high"] else "",
                "p_value": round(res["p_clustered"], 3) if res["p_clustered"] == res["p_clustered"] else "",
                "p_value_method": "cluster_bootstrap_two_sided",
                "n_cells": res["n"],
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
