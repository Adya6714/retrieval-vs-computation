#!/usr/bin/env python3
"""Emit variant_exclusions.csv for H2 (ALGO/BW W6) and H5 (MBW W5 clones).

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
OUT = DERIVED / "variant_exclusions.csv"
BANKS = {
    "ALGO": REPO_ROOT / "data/problems/question_bank_algo.csv",
    "BW": REPO_ROOT / "data/problems/question_bank_bw.csv",
}
MBW_W5 = [f"MBW_{i}" for i in range(496, 501)]


def _norm_vt(v: str) -> str:
    s = str(v).strip()
    if s.lower() == "canonical":
        return "canonical"
    if len(s) == 2 and s[0].lower() == "w" and s[1].isdigit():
        return f"W{s[1]}"
    return s.upper() if s else s


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for fam, path in BANKS.items():
        df = pd.read_csv(path, dtype=str).fillna("")
        df["variant"] = df["variant_type"].map(_norm_vt)
        w6 = df[df["variant"] == "W6"]
        for pid in w6["problem_id"].astype(str).str.strip().unique():
            rows.append(
                {
                    "family": fam,
                    "problem_id": pid,
                    "variant": "W6",
                    "reason": "variant_not_transformed",
                }
            )
    bw = pd.read_csv(BANKS["BW"], dtype=str).fillna("")
    bw["variant"] = bw["variant_type"].map(_norm_vt)
    for pid in MBW_W5:
        hit = bw[(bw["problem_id"] == pid) & (bw["variant"] == "W5")]
        if hit.empty:
            continue
        rows.append(
            {
                "family": "BW",
                "problem_id": pid,
                "variant": "W5",
                "reason": "variant_not_transformed",
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
