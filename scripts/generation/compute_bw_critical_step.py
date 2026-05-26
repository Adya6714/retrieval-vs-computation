#!/usr/bin/env python3
"""Set critical_step_index in difficulty_params for new BW canonical rows."""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

BANK_PATH = Path("data/problems/question_bank_bw.csv")
ID_PATTERN = re.compile(r"BW_4|BW_5|BW_E_|MBW_4|MBW_5")


def _needs_update(raw: str) -> bool:
    text = str(raw).strip()
    if text in ("", "{}", "null", "nan"):
        return True
    try:
        params = json.loads(text)
        return not isinstance(params, dict) or "critical_step_index" not in params
    except json.JSONDecodeError:
        return True


def main() -> None:
    df = pd.read_csv(BANK_PATH, dtype=str).fillna("")
    updated = 0
    sample_line = ""

    for idx, row in df.iterrows():
        if str(row.get("variant_type", "")).strip() != "canonical":
            continue
        pid = str(row.get("problem_id", "")).strip()
        if not ID_PATTERN.search(pid):
            continue
        if not _needs_update(str(row.get("difficulty_params", ""))):
            continue

        answer = str(row.get("correct_answer", ""))
        plan_lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
        plan_length = len(plan_lines)
        critical_step_index = math.floor(plan_length / 2)

        params = {
            "critical_step_index": critical_step_index,
            "plan_length": plan_length,
            "computed_from": "correct_answer_midpoint",
        }
        df.at[idx, "difficulty_params"] = json.dumps(params)
        updated += 1
        if pid == "BW_496":
            sample_line = (
                f"Sample: {pid} plan_length={plan_length} "
                f"critical_step_index={critical_step_index}"
            )

    df.to_csv(BANK_PATH, index=False)
    print(f"Updated {updated} rows")
    if sample_line:
        print(sample_line)
    elif updated:
        first = df[
            (df["variant_type"] == "canonical")
            & df["problem_id"].str.contains(ID_PATTERN, na=False)
        ].iloc[0]
        p = json.loads(str(first["difficulty_params"]))
        print(
            f"Sample: {first['problem_id']} plan_length={p.get('plan_length')} "
            f"critical_step_index={p.get('critical_step_index')}"
        )


if __name__ == "__main__":
    main()
