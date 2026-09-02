#!/usr/bin/env python3
"""K3: Instance-level Claude BW canonical vs W6 comparison (counts only)."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
BANK = REPO_ROOT / "data/problems/question_bank_bw.csv"
OUT = DER / "K3_claude_bw_canonical_w6_instances.csv"
SUM = DER / "K3_claude_bw_canonical_w6_summary.csv"
CLAUDE = "anthropic/claude-sonnet-4"


def _parse_dp(raw: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(raw or "").split("|"):
        part = part.strip()
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _plan_length(answer: str) -> int:
    return len([ln for ln in str(answer or "").splitlines() if ln.strip()])


def _goal_depth(text: str) -> int:
    """Count blocks named in ON(...) goal clauses (approximate goal depth)."""
    goals = re.findall(r"\(\s*on\s+(\w+)\s+(\w+)\s*\)", str(text or ""), flags=re.IGNORECASE)
    if not goals:
        return 0
    stacks: dict[str, set[str]] = {}
    for block, surface in goals:
        stacks.setdefault(surface.lower(), set()).add(block.lower())
    return max(len(s) for s in stacks.values()) if stacks else 0


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    bank = pd.read_csv(BANK, dtype=str).fillna("")
    bank["variant"] = bank["variant_type"].map(normalize_variant)
    can = bank[bank["variant"] == "canonical"].set_index("problem_id")
    w6 = bank[bank["variant"] == "W6"].set_index("problem_id")
    common = sorted(set(can.index) & set(w6.index))

    rescored = pd.read_csv(DER / "BW_P1_behavioral_rescored.csv", dtype=str).fillna("")
    rescored = rescored[rescored["included"].str.strip().str.lower().eq("true")].copy()
    bank_ids = set(
        pd.read_csv(BANK, dtype=str)
        .loc[lambda d: d["variant_type"].str.strip().str.lower() == "canonical", "problem_id"]
        .astype(str)
    )
    rescored = rescored[rescored["model"] == CLAUDE].copy()
    rescored = rescored[rescored["problem_id"].isin(bank_ids)].copy()
    rescored["variant"] = rescored["variant_type"].map(normalize_variant)
    rescored = filter_excluded(rescored, family="BW")
    rescored["ok"] = rescored["rescored_correct"].str.strip().str.lower().eq("true")
    score = rescored.set_index(["problem_id", "variant"])["ok"]

    rows: list[dict] = []
    for pid in common:
        c, w = can.loc[pid], w6.loc[pid]
        dc, dw = _parse_dp(c.get("difficulty_params", "")), _parse_dp(w.get("difficulty_params", ""))
        rows.append(
            {
                "problem_id": pid,
                "problem_subtype": c.get("problem_subtype", ""),
                "canonical_correct": bool(score.get((pid, "canonical"), False)),
                "w6_correct": bool(score.get((pid, "W6"), False)),
                "byte_identical_text": c["problem_text"] == w["problem_text"],
                "identical_answer": c["correct_answer"] == w["correct_answer"],
                "canonical_plan_length": _plan_length(c["correct_answer"]),
                "w6_plan_length": _plan_length(w["correct_answer"]),
                "plan_length_delta": _plan_length(w["correct_answer"]) - _plan_length(c["correct_answer"]),
                "canonical_num_blocks": dc.get("num_blocks", ""),
                "w6_num_blocks": dw.get("num_blocks", ""),
                "canonical_max_initial_stack_depth": dc.get("max_initial_stack_depth", ""),
                "w6_max_initial_stack_depth": dw.get("max_initial_stack_depth", ""),
                "canonical_goal_depth": _goal_depth(c["problem_text"]),
                "w6_goal_depth": _goal_depth(w["problem_text"]),
                "canonical_text_len": len(c["problem_text"]),
                "w6_text_len": len(w["problem_text"]),
                "text_len_delta": len(w["problem_text"]) - len(c["problem_text"]),
                "canonical_source": c.get("source", ""),
                "w6_source": w.get("source", ""),
                "canonical_notes": c.get("notes", "")[:120],
                "w6_notes": w.get("notes", "")[:120],
            }
        )

    inst = pd.DataFrame(rows)
    inst.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(inst)} rows)")

    valid = inst[~inst["byte_identical_text"]].copy()
    summary_rows = [
        {"metric": "n_instance_pairs", "value": len(inst)},
        {"metric": "n_excluded_true_duplicate", "value": int(inst["byte_identical_text"].sum())},
        {"metric": "n_valid_w6_pairs", "value": len(valid)},
        {"metric": "claude_acc_canonical", "value": round(valid["canonical_correct"].mean(), 3)},
        {"metric": "claude_acc_w6", "value": round(valid["w6_correct"].mean(), 3)},
        {"metric": "canonical_plan_length_mean", "value": round(valid["canonical_plan_length"].mean(), 3)},
        {"metric": "w6_plan_length_mean", "value": round(valid["w6_plan_length"].mean(), 3)},
        {"metric": "plan_length_delta_mean", "value": round(valid["plan_length_delta"].mean(), 3)},
        {"metric": "canonical_goal_depth_mean", "value": round(valid["canonical_goal_depth"].mean(), 3)},
        {"metric": "w6_goal_depth_mean", "value": round(valid["w6_goal_depth"].mean(), 3)},
        {"metric": "canonical_text_len_mean", "value": round(valid["canonical_text_len"].mean(), 1)},
        {"metric": "w6_text_len_mean", "value": round(valid["w6_text_len"].mean(), 1)},
    ]
    nb = valid[valid["canonical_num_blocks"].astype(str).str.len() > 0]
    if len(nb):
        summary_rows.extend(
            [
                {
                    "metric": "canonical_num_blocks_mean",
                    "value": round(pd.to_numeric(nb["canonical_num_blocks"], errors="coerce").mean(), 3),
                },
                {
                    "metric": "w6_num_blocks_mean",
                    "value": round(pd.to_numeric(nb["w6_num_blocks"], errors="coerce").mean(), 3),
                },
            ]
        )
    sum_df = pd.DataFrame(summary_rows)
    sum_df.to_csv(SUM, index=False)
    print(f"Wrote {SUM}")
    print(sum_df.to_string(index=False))


if __name__ == "__main__":
    main()
