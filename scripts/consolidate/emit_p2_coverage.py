#!/usr/bin/env python3
"""Probe 2 coverage table and BW CCI/TEP null diagnosis.

Does not call any model API. Does not write results/raw/.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.behavioral.bw_action_parser_nl import remap_to_canonical  # noqa: E402
from probes.common.results_paths import algo_p2_phase1_files  # noqa: E402

RAW = REPO_ROOT / "results/raw"
DERIVED = REPO_ROOT / "results/derived"


def _num(s) -> float | None:
    v = pd.to_numeric(s, errors="coerce")
    if pd.isna(v):
        return None
    return float(v)


def _status(s) -> str:
    t = str(s or "").strip()
    return t if t else "blank"


def _json_list(raw) -> list:
    try:
        val = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
    except json.JSONDecodeError:
        return []
    return val if isinstance(val, list) else []


def _nl_remap_counts(actions: list) -> tuple[int, int, int]:
    """Return (n_actions, n_already_canonical_or_empty, n_nl_remappable)."""
    n = n_canonish = n_remap = 0
    for act in actions:
        s = str(act or "").strip()
        if not s or s == "STEP_SKIP":
            continue
        n += 1
        remapped = remap_to_canonical(s)
        parts = s.lower().split()
        verb = parts[0] if parts else ""
        if verb in {"pick-up", "put-down", "stack", "unstack"} and len(parts) in {2, 3}:
            n_canonish += 1
        elif remapped:
            n_remap += 1
    return n, n_canonish, n_remap


def _int0(s) -> int:
    v = pd.to_numeric(s, errors="coerce")
    if pd.isna(v):
        return 0
    return int(v)


def diagnose_bw_cci() -> pd.DataFrame:
    df = pd.read_csv(RAW / "BW_P2_cci.csv", dtype=str).fillna("")
    rows = []
    cat_counter: Counter[str] = Counter()
    for _, r in df.iterrows():
        cci = _num(r.get("cci"))
        status = _status(r.get("session_status"))
        fmt = _int0(r.get("violation_format_error"))
        precond = 0
        for col in (
            "violation_hand_not_empty",
            "violation_block_not_clear",
            "violation_block_not_on_table",
            "violation_wrong_stack_source",
            "violation_target_not_clear",
            "violation_other",
        ):
            precond += _int0(r.get(col))
        actions = _json_list(r.get("executed_steps_json", "[]"))
        n_act, n_canon, n_remap = _nl_remap_counts(actions)
        skip = _int0(r.get("skip_count"))
        if cci is not None:
            cause = "has_score"
        elif status.startswith("aborted: excessive illegal"):
            if fmt > precond and n_remap > n_canon:
                cause = "parser_then_abort"
            elif precond > fmt:
                cause = "precondition_then_abort"
            else:
                cause = "abort_excessive_illegal"
        elif status.startswith("aborted"):
            cause = f"abort:{status}"
        elif status == "blank":
            cause = "blank_status_null_cci"
        else:
            cause = f"null_cci_status:{status}"
        cat_counter[cause] += 1
        rows.append(
            {
                "problem_id": r.get("problem_id"),
                "model": r.get("model"),
                "cci": cci,
                "session_status": status,
                "skip_count": skip,
                "violation_format_error": fmt,
                "violation_precondition": precond,
                "n_executed_actions": n_act,
                "n_canonical_shape": n_canon,
                "n_nl_remappable": n_remap,
                "null_cause": cause,
            }
        )
    out = pd.DataFrame(rows)
    print("=== B1 BW_P2_cci.csv ===")
    print(f"rows={len(df)} null_cci={(out['cci'].isna()).sum()} nonnull={(out['cci'].notna()).sum()}")
    print("session_status:", df["session_status"].fillna("").map(_status).value_counts().to_dict())
    print("null_cause:", dict(cat_counter))
    print(
        "format_error total",
        int(pd.to_numeric(df["violation_format_error"], errors="coerce").fillna(0).sum()),
        "precondition total",
        int(out["violation_precondition"].sum()),
        "NL-remappable actions (all rows)",
        int(out["n_nl_remappable"].sum()),
        "canonical-shape actions",
        int(out["n_canonical_shape"].sum()),
    )
    return out


def diagnose_bw_tep() -> pd.DataFrame:
    df = pd.read_csv(RAW / "BW_P2_tep.csv", dtype=str).fillna("")
    rows = []
    for _, r in df.iterrows():
        tep = _num(r.get("tep"))
        status = _status(r.get("session_status"))
        cascade = _json_list(r.get("cascade_sequence_json", "[]"))
        classes = Counter(
            str(s.get("classification", "")).strip().lower()
            for s in cascade
            if isinstance(s, dict)
        )
        adapted = classes.get("adapted", 0)
        resistant = classes.get("resistant", 0)
        illegal = classes.get("illegal_both", 0)
        denom = adapted + resistant
        actions = [str(s.get("action", "")) for s in cascade if isinstance(s, dict)]
        _, n_canon, n_remap = _nl_remap_counts(actions)
        if tep is not None:
            cause = "has_score"
        elif status.startswith("aborted"):
            cause = "abort_forces_null"
        elif denom == 0:
            cause = "tep_denom_zero_all_illegal_or_ambiguous"
        else:
            cause = "tep_null_other"
        rows.append(
            {
                "problem_id": r.get("problem_id"),
                "model": r.get("model"),
                "tep": tep,
                "session_status": status,
                "adapted": adapted,
                "resistant": resistant,
                "illegal_both": illegal,
                "n_nl_remappable": n_remap,
                "n_canonical_shape": n_canon,
                "null_cause": cause,
            }
        )
    out = pd.DataFrame(rows)
    print("=== B1 BW_P2_tep.csv ===")
    print(f"rows={len(df)} null_tep={(out['tep'].isna()).sum()} nonnull={(out['tep'].notna()).sum()}")
    print("session_status:", df["session_status"].fillna("").map(_status).value_counts().to_dict())
    print("null_cause:", out["null_cause"].value_counts().to_dict())
    print(
        "illegal_both sum",
        int(out["illegal_both"].sum()),
        "NL-remappable cascade actions",
        int(out["n_nl_remappable"].sum()),
    )
    return out


def diagnose_nl() -> None:
    bw = pd.read_csv(RAW / "BW_P2_cci_nl.csv", dtype=str).fillna("")
    mbw = pd.read_csv(RAW / "MBW_P2_cci_nl.csv", dtype=str).fillna("")
    print("=== B1 BW_P2_cci_nl.csv ===")
    cci = pd.to_numeric(bw.get("cci"), errors="coerce")
    print(
        f"rows={len(bw)} nonnull_cci={int(cci.notna().sum())} null={int(cci.isna().sum())}"
    )
    print("session_status:", bw["session_status"].fillna("").map(_status).value_counts().to_dict())
    print("=== B1 MBW_P2_cci_nl.csv ===")
    print(f"rows={len(mbw)} columns={list(mbw.columns)}")
    print("has cci column:", "cci" in mbw.columns)
    print("session_status:", mbw["session_status"].fillna("").map(_status).value_counts().to_dict())


def _coverage_row(family: str, model: str, attempted: int, usable: int, reasons: Counter) -> dict:
    top = reasons.most_common(3)
    return {
        "family": family,
        "model": model,
        "rows_attempted": attempted,
        "rows_with_usable_score": usable,
        "null_rows": attempted - usable,
        "null_reason_1": top[0][0] if top else "",
        "null_reason_1_n": top[0][1] if top else 0,
        "null_reason_2": top[1][0] if len(top) > 1 else "",
        "null_reason_2_n": top[1][1] if len(top) > 1 else 0,
        "null_reason_3": top[2][0] if len(top) > 2 else "",
        "null_reason_3_n": top[2][1] if len(top) > 2 else 0,
    }


def emit_coverage() -> pd.DataFrame:
    rows = []

    def add_score_file(family: str, path: Path, score_col: str, extra_reason=None) -> None:
        if not path.exists():
            return
        df = pd.read_csv(path, dtype=str).fillna("")
        if df.empty:
            return
        for model, sub in df.groupby(df["model"].astype(str)):
            scores = pd.to_numeric(sub[score_col], errors="coerce") if score_col in sub.columns else pd.Series([None] * len(sub))
            usable = int(scores.notna().sum())
            reasons: Counter[str] = Counter()
            for _, r in sub.iterrows():
                if pd.notna(pd.to_numeric(r.get(score_col), errors="coerce")):
                    continue
                if extra_reason is not None:
                    reasons[extra_reason(r)] += 1
                else:
                    st = _status(r.get("session_status"))
                    if st == "blank":
                        reasons["blank_session_status"] += 1
                    elif st.startswith("aborted"):
                        reasons[st] += 1
                    elif score_col not in sub.columns:
                        reasons["score_column_absent"] += 1
                    else:
                        reasons["score_null"] += 1
            rows.append(_coverage_row(family, str(model), len(sub), usable, reasons))

    add_score_file("BW_cci_strict", RAW / "BW_P2_cci.csv", "cci")
    add_score_file("BW_cci_nl", RAW / "BW_P2_cci_nl.csv", "cci")
    add_score_file("BW_tep", RAW / "BW_P2_tep.csv", "tep")

    mbw_path = RAW / "MBW_P2_cci_nl.csv"
    mbw = pd.read_csv(mbw_path, dtype=str).fillna("")
    for model, sub in mbw.groupby(mbw["model"].astype(str)):
        reasons: Counter[str] = Counter()
        if "cci" not in sub.columns:
            reasons["cci_column_never_emitted"] = len(sub)
            usable = 0
        else:
            usable = int(pd.to_numeric(sub["cci"], errors="coerce").notna().sum())
            for _, r in sub.iterrows():
                if pd.isna(pd.to_numeric(r.get("cci"), errors="coerce")):
                    reasons[_status(r.get("session_status"))] += 1
        rows.append(_coverage_row("MBW_cci_nl", str(model), len(sub), usable, reasons))

    add_score_file("GSM_cci", RAW / "GSM_P2_cci.csv", "cci_score")
    add_score_file("GSM_tep", RAW / "GSM_P2_cci.csv", "tep_score")

    for path in algo_p2_phase1_files():
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        for model, sub in df.groupby(df["model"].astype(str)):
            parseable = sub["phase1_parseable"].astype(str).str.strip().str.lower().eq("true")
            usable = int(parseable.sum())
            reasons: Counter[str] = Counter()
            reasons["phase1_unparseable"] = int((~parseable).sum())
            rows.append(
                _coverage_row(f"ALGO_phase1:{path.name}", str(model), len(sub), usable, reasons)
            )

    for name, fname, col in (
        ("ALGO_phase2_normal", "ALGO_P2_phase2_normal.csv", "step_correct"),
        ("ALGO_phase2_injected", "ALGO_P2_phase2_injected.csv", "step_correct"),
    ):
        path = RAW / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        score_col = col if col in df.columns else None
        if score_col is None:
            for cand in ("correct", "cci", "tep"):
                if cand in df.columns:
                    score_col = cand
                    break
        if score_col is None:
            continue
        add_score_file(name, path, score_col)

    out = pd.DataFrame(rows)
    return out


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    cci_d = diagnose_bw_cci()
    tep_d = diagnose_bw_tep()
    diagnose_nl()
    cci_d.to_csv(DERIVED / "P2_bw_cci_null_diagnosis.csv", index=False)
    tep_d.to_csv(DERIVED / "P2_bw_tep_null_diagnosis.csv", index=False)
    cov = emit_coverage()
    out = DERIVED / "P2_coverage.csv"
    cov.to_csv(out, index=False)
    print("=== B3 P2_coverage.csv ===")
    print(cov.to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
