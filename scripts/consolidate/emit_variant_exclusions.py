#!/usr/bin/env python3
"""Emit variant_exclusions.csv from transform audit + fixed instrument rules.

Does not write results/raw/. Does not regenerate W6.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DERIVED = REPO_ROOT / "results/derived"
AUDIT = DERIVED / "variant_transform_audit.csv"
OUT = DERIVED / "variant_exclusions.csv"
MBW_W5 = [f"MBW_{i}" for i in range(496, 501)]
GSM_OFFBANK_W6 = [f"GSM_{i:03d}" for i in range(1, 21)]


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    if not AUDIT.exists():
        raise FileNotFoundError(f"Run audit_variant_transforms.py first: {AUDIT}")

    audit = pd.read_csv(AUDIT, dtype=str).fillna("")
    rows: list[dict] = []

    w6_bad = audit[
        (audit["variant"] == "W6")
        & (audit["transform_status"] == "identical_to_canonical")
    ]
    for _, r in w6_bad.iterrows():
        rows.append(
            {
                "family": str(r["bank"]).strip().upper(),
                "problem_id": str(r["problem_id"]).strip(),
                "variant": "W6",
                "reason": "variant_not_transformed",
            }
        )

    for pid in MBW_W5:
        rows.append(
            {
                "family": "BW",
                "problem_id": pid,
                "variant": "W5",
                "reason": "variant_not_transformed",
            }
        )

    for pid in GSM_OFFBANK_W6:
        rows.append(
            {
                "family": "GSM",
                "problem_id": pid,
                "variant": "W6",
                "reason": "missing_bank_row",
            }
        )

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["family", "problem_id", "variant", "reason"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT} ({len(rows)} rows)")
    print(pd.DataFrame(rows).groupby(["family", "variant", "reason"]).size().to_string())


if __name__ == "__main__":
    main()
