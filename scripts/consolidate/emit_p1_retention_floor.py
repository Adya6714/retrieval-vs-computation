#!/usr/bin/env python3
"""Emit the W3 retention floor audit table.

R_W3 = Acc_W3 / Acc_canonical is undefined when Acc_canonical is below
MIN_CANONICAL_FOR_RETENTION (a single pre-specified constant). This script
lists every (family, subtype, model) cell the floor suppresses, with its
canonical accuracy, so the rule is auditable.

Does not tune the constant. Does not write results/raw/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.behavioral.retention import (  # noqa: E402
    MIN_CANONICAL_FOR_RETENTION,
    REASON_CANONICAL_BELOW_FLOOR,
    retention_ratio,
)

SUMMARY = REPO_ROOT / "results/derived/P1_rescore_summary.csv"
OUT = REPO_ROOT / "results/derived/P1_retention_floor_report.csv"

FAMILY_FROM_SOURCE = {
    "ALGO": "ALGO",
    "BW": "BW",
    "GSM": "GSM",
}


def _probe_family(source_file: str) -> str:
    name = str(source_file)
    for prefix, fam in FAMILY_FROM_SOURCE.items():
        if name.startswith(prefix):
            return fam
    return "unknown"


def main() -> None:
    df = pd.read_csv(SUMMARY)
    df["variant"] = df["variant"].astype(str).str.strip()
    df["probe_family"] = df["source_file"].map(_probe_family)

    suppressed: list[dict] = []
    defined = 0
    missing = 0
    n_pairs = 0

    keys = ["source_file", "model", "family"]
    for (source, model, subtype), g in df.groupby(keys, dropna=False):
        can = g[g["variant"] == "canonical"]
        w3 = g[g["variant"] == "W3"]
        if can.empty and w3.empty:
            continue
        n_pairs += 1
        a_can = float(can["new_accuracy"].iloc[0]) if not can.empty else None
        a_w3 = float(w3["new_accuracy"].iloc[0]) if not w3.empty else None
        n_can = int(can["n"].iloc[0]) if not can.empty else 0
        n_w3 = int(w3["n"].iloc[0]) if not w3.empty else 0
        ret, reason = retention_ratio(a_w3, a_can)
        if reason == REASON_CANONICAL_BELOW_FLOOR:
            suppressed.append(
                {
                    "source_file": source,
                    "probe_family": _probe_family(str(source)),
                    "family": subtype,
                    "model": model,
                    "n_canonical": n_can,
                    "n_w3": n_w3,
                    "acc_canonical": a_can,
                    "acc_w3": a_w3,
                    "would_be_R_W3": (None if a_can in (None, 0) else a_w3 / a_can)
                    if a_w3 is not None and a_can not in (None, 0)
                    else None,
                    "R_W3": "",
                    "exclusion_reason": reason,
                    "floor": MIN_CANONICAL_FOR_RETENTION,
                }
            )
        elif ret is None:
            missing += 1
        else:
            defined += 1

    out = pd.DataFrame(suppressed)
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "source_file",
                "probe_family",
                "family",
                "model",
                "n_canonical",
                "n_w3",
                "acc_canonical",
                "acc_w3",
                "would_be_R_W3",
                "R_W3",
                "exclusion_reason",
                "floor",
            ]
        )
    out.to_csv(OUT, index=False)
    print(
        f"Wrote {OUT} ({len(out)} suppressed cells; "
        f"{defined} defined; {missing} missing; {n_pairs} W3/canonical pairs; "
        f"floor={MIN_CANONICAL_FOR_RETENTION})"
    )


if __name__ == "__main__":
    main()
