#!/usr/bin/env python3
"""Offline NL-remap rescore of existing BW/MBW Probe 2 traces.

Reads results/raw/BW_P2_{cci,tep,plans}.csv and MBW_P2_cci_nl.csv.
Does not call any model API. Does not write results/raw/.

Strict CCI/TEP still nulls scores on aborted sessions (E3 separates abort
from TEP scoring). Precondition checking stays strict; only the action
string is rewritten by remap_to_canonical / normalize_action.
"""

from __future__ import annotations

import copy
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.behavioral.bw_action_parser_nl import normalize_action  # noqa: E402
from probes.behavioral.bw_cci_pipeline import (  # noqa: E402
    execute_action,
    parse_state_from_text,
    profile_actions,
    seeded_inject_error,
)
from probes.behavioral.cci import compute_cci  # noqa: E402
from probes.contamination.verify import _extract_mystery_actions_line_based  # noqa: E402

RAW = REPO_ROOT / "results/raw"
DERIVED = REPO_ROOT / "results/derived"
BANK = REPO_ROOT / "data/problems/question_bank_bw.csv"

PRECOND_CATS = {
    "hand_not_empty",
    "block_not_clear",
    "block_not_on_table",
    "wrong_stack_source",
    "target_not_clear",
    "other_illegal",
}


def _json_list(raw) -> list:
    try:
        val = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
    except json.JSONDecodeError:
        return []
    return val if isinstance(val, list) else []


def _num(s):
    v = pd.to_numeric(s, errors="coerce")
    if pd.isna(v):
        return None
    return float(v)


def _aborted(status: str) -> bool:
    return str(status or "").strip().startswith("aborted:")


def _bank_lookup() -> dict[str, dict]:
    bank = pd.read_csv(BANK, dtype=str).fillna("")
    out = {}
    for _, row in bank.iterrows():
        if str(row.get("variant_type", "")).strip() != "canonical":
            continue
        out[str(row["problem_id"]).strip()] = row.to_dict()
    return out


def classify_action(action, true_state, displayed_state) -> str:
    try:
        execute_action(copy.deepcopy(true_state), action)
        valid_true = True
    except ValueError:
        valid_true = False
    try:
        execute_action(copy.deepcopy(displayed_state), action)
        valid_disp = True
    except ValueError:
        valid_disp = False
    if valid_disp and valid_true:
        return "ambiguous"
    if valid_disp and not valid_true:
        return "adapted"
    if valid_true and not valid_disp:
        return "resistant"
    return "illegal_both"


def compute_tep(cascade_sequence) -> float | None:
    adapted = sum(1 for s in cascade_sequence if s.get("classification") == "adapted")
    resistant = sum(1 for s in cascade_sequence if s.get("classification") == "resistant")
    denom = adapted + resistant
    return round(adapted / denom, 4) if denom > 0 else None


def remap_steps(steps: list) -> list[str]:
    out = []
    for act in steps:
        s = str(act or "").strip()
        if not s or s == "STEP_SKIP":
            out.append(s)
            continue
        out.append(normalize_action(s))
    return out


def rescore_cci(bank: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RAW / "BW_P2_cci.csv", dtype=str).fillna("")
    plans = pd.read_csv(RAW / "BW_P2_plans.csv", dtype=str).fillna("")
    plan_lookup = {}
    for _, row in plans.iterrows():
        plan_lookup[(str(row["problem_id"]), str(row["model"]))] = _json_list(
            row.get("parsed_plan_json")
        )

    before_fmt = int(pd.to_numeric(df["violation_format_error"], errors="coerce").fillna(0).sum())
    before_pre = 0
    for col in (
        "violation_hand_not_empty",
        "violation_block_not_clear",
        "violation_block_not_on_table",
        "violation_wrong_stack_source",
        "violation_target_not_clear",
        "violation_other",
    ):
        before_pre += int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
    before_cci = int(pd.to_numeric(df["cci"], errors="coerce").notna().sum())

    rows = []
    cat_counter: Counter[str] = Counter()
    parse_fail = 0
    for _, row in df.iterrows():
        pid = str(row["problem_id"]).strip()
        model = str(row["model"]).strip()
        executed = remap_steps(_json_list(row.get("executed_steps_json")))
        plan = remap_steps(plan_lookup.get((pid, model), []))
        bank_row = bank.get(pid)
        profile = []
        parse_ok = True
        if bank_row is None:
            parse_ok = False
            parse_fail += 1
        else:
            try:
                _, init_state, _ = parse_state_from_text(bank_row["problem_text"])
                profile = profile_actions(executed, init_state)
            except Exception:
                parse_ok = False
                parse_fail += 1
        vcounts = Counter(p["category"] for p in profile)
        cat_counter.update(vcounts)
        filtered = [s for s in executed if s != "STEP_SKIP"]
        if _aborted(row.get("session_status")) or not plan or not filtered:
            cci_result = {"cci": None, "matched_steps": 0, "total_steps_compared": 0}
        else:
            cci_result = compute_cci(pid, plan, filtered)
        out = row.to_dict()
        out.update(
            {
                "cci": "" if cci_result["cci"] is None else cci_result["cci"],
                "matched_steps": cci_result["matched_steps"],
                "total_steps_compared": cci_result["total_steps_compared"],
                "generated_plan_length": len(plan),
                "executed_length": len(executed),
                "executed_steps_json": json.dumps(executed),
                "violation_hand_not_empty": vcounts.get("hand_not_empty", 0),
                "violation_block_not_clear": vcounts.get("block_not_clear", 0),
                "violation_block_not_on_table": vcounts.get("block_not_on_table", 0),
                "violation_wrong_stack_source": vcounts.get("wrong_stack_source", 0),
                "violation_target_not_clear": vcounts.get("target_not_clear", 0),
                "violation_format_error": vcounts.get("format_error", 0),
                "violation_other": vcounts.get("other_illegal", 0),
                "violation_profile_json": json.dumps(profile),
                "nl_remap": "True",
                "state_parse_ok": str(parse_ok),
            }
        )
        rows.append(out)

    out_df = pd.DataFrame(rows)
    path = DERIVED / "BW_P2_cci_nl_rescored.csv"
    out_df.to_csv(path, index=False)
    after_cci = int(pd.to_numeric(out_df["cci"], errors="coerce").notna().sum())
    after_fmt = int(cat_counter.get("format_error", 0))
    after_pre = sum(cat_counter.get(c, 0) for c in PRECOND_CATS)
    stats = {
        "cci_usable_before": before_cci,
        "cci_usable_after": after_cci,
        "format_error_before": before_fmt,
        "format_error_after": after_fmt,
        "precondition_before": before_pre,
        "precondition_after": after_pre,
        "cci_state_parse_failures": parse_fail,
        "cci_rows": len(out_df),
        "cci_out": str(path.relative_to(REPO_ROOT)),
    }
    print(f"Wrote {path} ({len(out_df)} rows; usable CCI {before_cci} -> {after_cci})")
    print(
        f"  format_error {before_fmt} -> {after_fmt}; "
        f"precondition {before_pre} -> {after_pre}"
    )
    return out_df, stats


def rescore_tep(bank: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RAW / "BW_P2_tep.csv", dtype=str).fillna("")
    plans = pd.read_csv(RAW / "BW_P2_plans.csv", dtype=str).fillna("")
    plan_lookup = {}
    for _, row in plans.iterrows():
        plan_lookup[(str(row["problem_id"]), str(row["model"]))] = remap_steps(
            _json_list(row.get("parsed_plan_json"))
        )

    before_tep = int(pd.to_numeric(df["tep"], errors="coerce").notna().sum())
    rows = []
    recon_fail = 0
    for _, row in df.iterrows():
        pid = str(row["problem_id"]).strip()
        model = str(row["model"]).strip()
        cascade = _json_list(row.get("cascade_sequence_json"))
        try:
            inject_at = int(float(row.get("inject_at_step") or 0))
        except (TypeError, ValueError):
            inject_at = 0
        bank_row = bank.get(pid)
        new_cascade = []
        parse_ok = True
        if bank_row is None:
            parse_ok = False
            recon_fail += 1
            new_cascade = cascade
        else:
            try:
                objects, init_state, _ = parse_state_from_text(bank_row["problem_text"])
                true_state = copy.deepcopy(init_state)
                displayed = copy.deepcopy(init_state)
                plan = plan_lookup.get((pid, model), [])
                for i, act in enumerate(plan):
                    if i > inject_at:
                        break
                    if i == inject_at:
                        displayed, _ = seeded_inject_error(
                            displayed, objects, seed_str=f"{pid}_{inject_at}"
                        )
                    if not act or act == "STEP_SKIP":
                        continue
                    try:
                        true_state = execute_action(copy.deepcopy(true_state), act)
                    except ValueError:
                        pass
                    try:
                        displayed = execute_action(copy.deepcopy(displayed), act)
                    except ValueError:
                        pass
                if inject_at >= len(plan):
                    displayed, _ = seeded_inject_error(
                        displayed, objects, seed_str=f"{pid}_{inject_at}"
                    )
                for entry in cascade:
                    if not isinstance(entry, dict):
                        continue
                    raw_act = str(entry.get("action", "")).strip()
                    if raw_act == "STEP_SKIP" or not raw_act:
                        new_cascade.append(
                            {
                                "step": entry.get("step"),
                                "action": raw_act,
                                "classification": "illegal_both",
                            }
                        )
                        continue
                    remapped = normalize_action(raw_act)
                    cls = classify_action(remapped, true_state, displayed)
                    new_cascade.append(
                        {
                            "step": entry.get("step"),
                            "action": remapped,
                            "action_original": raw_act,
                            "classification": cls,
                        }
                    )
                    try:
                        true_state = execute_action(copy.deepcopy(true_state), remapped)
                    except ValueError:
                        pass
                    try:
                        displayed = execute_action(copy.deepcopy(displayed), remapped)
                    except ValueError:
                        pass
            except Exception:
                parse_ok = False
                recon_fail += 1
                new_cascade = cascade

        adapted = sum(1 for s in new_cascade if s.get("classification") == "adapted")
        resistant = sum(1 for s in new_cascade if s.get("classification") == "resistant")
        ambiguous = sum(1 for s in new_cascade if s.get("classification") == "ambiguous")
        illegal_both = sum(1 for s in new_cascade if s.get("classification") == "illegal_both")
        status = str(row.get("session_status", "")).strip()
        tep = None if _aborted(status) else compute_tep(new_cascade)
        out = row.to_dict()
        out.update(
            {
                "tep": "" if tep is None else f"{tep:.4f}",
                "adapted_count": adapted,
                "resistant_count": resistant,
                "ambiguous_count": ambiguous,
                "illegal_both_count": illegal_both,
                "steps_after_injection": len(new_cascade),
                "first_response_class": (
                    new_cascade[0]["classification"] if new_cascade else ""
                ),
                "cascade_sequence_json": json.dumps(new_cascade),
                "nl_remap": "True",
                "state_parse_ok": str(parse_ok),
            }
        )
        rows.append(out)

    out_df = pd.DataFrame(rows)
    path = DERIVED / "BW_P2_tep_nl_rescored.csv"
    out_df.to_csv(path, index=False)
    after_tep = int(pd.to_numeric(out_df["tep"], errors="coerce").notna().sum())
    stats = {
        "tep_usable_before": before_tep,
        "tep_usable_after": after_tep,
        "tep_recon_failures": recon_fail,
        "tep_rows": len(out_df),
        "tep_out": str(path.relative_to(REPO_ROOT)),
    }
    print(f"Wrote {path} ({len(out_df)} rows; usable TEP {before_tep} -> {after_tep})")
    return out_df, stats


def rescore_mbw(bank: dict[str, dict]) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(RAW / "MBW_P2_cci_nl.csv", dtype=str).fillna("")
    before = 0
    if "cci" in df.columns:
        before = int(pd.to_numeric(df["cci"], errors="coerce").notna().sum())
    rows = []
    scored = 0
    complete = 0
    for _, row in df.iterrows():
        pid = str(row["problem_id"]).strip()
        status = str(row.get("session_status", "")).strip()
        if status == "complete":
            complete += 1
        executed = [s for s in _json_list(row.get("executed_steps_json")) if s != "STEP_SKIP"]
        gold = _extract_mystery_actions_line_based(
            str(bank.get(pid, {}).get("correct_answer", ""))
        )
        if _aborted(status) or not gold or not executed:
            cci_result = {"cci": None, "matched_steps": 0, "total_steps_compared": 0}
        else:
            cci_result = compute_cci(pid, gold, executed)
        if cci_result["cci"] is not None:
            scored += 1
        out = row.to_dict()
        out.update(
            {
                "cci": "" if cci_result["cci"] is None else cci_result["cci"],
                "matched_steps": cci_result["matched_steps"],
                "total_steps_compared": cci_result["total_steps_compared"],
                "generated_plan_length": len(gold),
            }
        )
        rows.append(out)
    out_df = pd.DataFrame(rows)
    path = DERIVED / "MBW_P2_cci_nl_rescored.csv"
    out_df.to_csv(path, index=False)
    stats = {
        "mbw_cci_usable_before": before,
        "mbw_cci_usable_after": scored,
        "mbw_complete": complete,
        "mbw_rows": len(out_df),
        "mbw_out": str(path.relative_to(REPO_ROOT)),
    }
    print(
        f"Wrote {path} ({len(out_df)} rows; complete={complete}; "
        f"usable CCI {before} -> {scored})"
    )
    return out_df, stats


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    bank = _bank_lookup()
    _, cci_stats = rescore_cci(bank)
    _, tep_stats = rescore_tep(bank)
    _, mbw_stats = rescore_mbw(bank)
    report = {
        **cci_stats,
        **tep_stats,
        **mbw_stats,
    }
    report_path = DERIVED / "P2_bw_nl_rescore_report.csv"
    pd.DataFrame([report]).to_csv(report_path, index=False)
    print(f"Wrote {report_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
