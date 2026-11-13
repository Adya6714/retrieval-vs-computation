"""Derive scores from raw calls. Deterministic; safe to re-run at any time."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from . import verify
from .runner import read_calls

FIELDS = ["call_id", "experiment", "model", "thinking", "instruction", "arm", "item_id", "problem_id", "family", "subtype",
          "variant", "tier", "sample", "status", "usable", "exclude_reason", "correct", "correct_strict", "marker",
          "parse_status", "reason", "finish_reason", "completion_tokens", "reasoning_tokens", "provider", "served_model"]


def score_run(d: Path, bank_by_id: dict) -> list[dict]:
    rows = []
    for cid, c in read_calls(d / "calls.jsonl").items():
        item = bank_by_id.get(c["item_id"])
        row = {k: c.get(k, "") for k in FIELDS}
        row.update(problem_id=c["item_id"].split("::")[0], variant=c["item_id"].split("::")[1])
        usage = c.get("usage") or {}
        row.update(completion_tokens=usage.get("completion_tokens"), reasoning_tokens=usage.get("reasoning_tokens"))
        if item is None:
            row.update(usable=False, exclude_reason="item_not_in_current_bank")
        elif c.get("status") != "ok":
            row.update(family=item["family"], subtype=item["subtype"], tier=item["tier"], usable=False, exclude_reason="api_error")
        else:
            v = verify.score(c.get("response", ""), item)
            row.update(family=item["family"], subtype=item["subtype"], tier=item["tier"], **v.as_dict())
            # A truncated response is not evidence about the model's answer. It is excluded and counted.
            if c.get("finish_reason") == "length" and not v.correct:
                row.update(usable=False, exclude_reason="truncated")
            else:
                row.update(usable=True, exclude_reason="")
        rows.append(row)
    rows.sort(key=lambda r: (r["experiment"], r["model"], r["thinking"], r["instruction"], r["item_id"], r["arm"], r["sample"]))
    with (d / "scored.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows
