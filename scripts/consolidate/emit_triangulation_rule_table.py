#!/usr/bin/env python3
"""Compare canonical appendix triangulation vs the legacy 5-field sensitivity variant.

Writes results/derived/P3_triangulation_rule_comparison_detail.csv.
The two-row published-vs-variant table is emitted by
scripts/consolidate/run_appendix_triangulation_sweep.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rebuild.triangulation_rule import (  # noqa: E402
    count_labels,
    label_appendix_three_signal,
    label_default,
    label_legacy_five_field,
)

DERIVED = REPO_ROOT / "results/derived"
PANEL_CANDIDATES = [
    DERIVED / "ALGO_P3_triangulation_v3.csv",
    DERIVED / "ALGO_P3_triangulation.csv",
]
CURRENT_REF = {"ambiguous": 205, "mixed": 115, "retrieval": 6, "computation": 4}


def _load_panel() -> tuple[pd.DataFrame, Path]:
    for path in PANEL_CANDIDATES:
        if path.exists():
            df = pd.read_csv(path)
            return df, path
    raise FileNotFoundError("No ALGO triangulation panel found")


def _boolish(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    mapped = s.map(
        lambda x: True
        if x is True or str(x).strip().lower() in {"true", "1", "1.0"}
        else False
        if x is False or str(x).strip().lower() in {"false", "0", "0.0", ""}
        else pd.NA
    )
    return mapped


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("missing_core", "parse_failure_or_missing", "missing_phase2", "greedy_succeeds"):
        if col in out.columns:
            out[col] = _boolish(out[col])
    if "instance_rank_pct" not in out.columns:
        out["instance_rank_pct"] = out.groupby("problem_subtype")["instance_contamination_score"].rank(
            method="average", pct=True
        )
    if "ACI" not in out.columns and "cci_composite" in out.columns:
        out["ACI"] = pd.to_numeric(out["cci_composite"], errors="coerce")
    return out


def _rescored_var() -> pd.DataFrame:
    parts = []
    for path in sorted(DERIVED.glob("ALGO_P1_behavioral_*_rescored.csv")):
        df = pd.read_csv(path, dtype=str).fillna("")
        parts.append(df)
    if not parts:
        raise FileNotFoundError("No ALGO P1 rescored files")
    raw = pd.concat(parts, ignore_index=True)
    raw = raw[raw["model"].astype(str).str.strip().str.lower() != "mock"].copy()
    raw["variant_type"] = raw.get("variant_type_normalized", raw["variant_type"]).astype(str).str.strip()
    included = raw["included"].astype(str).str.strip().str.lower().eq("true")
    raw = raw[included].copy()
    raw["ok"] = raw["rescored_correct"].astype(str).str.strip().str.lower().map(
        {"true": 1.0, "false": 0.0}
    )
    can = raw[raw["variant_type"] == "canonical"][["problem_id", "model", "ok"]].rename(
        columns={"ok": "VAR_canonical_rescored"}
    )
    w3 = raw[raw["variant_type"] == "W3"][["problem_id", "model", "ok"]].rename(
        columns={"ok": "VAR_W3_rescored"}
    )
    return can.merge(w3, on=["problem_id", "model"], how="outer")


def _counts_row(rule: str, description: str, counts: dict[str, int], panel: str) -> dict:
    return {
        "rule": rule,
        "description": description,
        "panel": panel,
        "retrieval": counts.get("retrieval", 0),
        "computation": counts.get("computation", 0),
        "mixed": counts.get("mixed", 0),
        "ambiguous": counts.get("ambiguous", 0),
        "n": counts.get("n", 0),
    }


def main() -> None:
    panel, src = _load_panel()
    panel = _prepare(panel)
    print(f"Panel {src.name}: {len(panel)} rows, models={sorted(panel['model'].astype(str).unique())}")

    existing = panel["convergence_label"].astype(str).str.replace("_signal", "", regex=False)
    print("existing convergence_label:", count_labels(existing))

    executed = label_default(panel)
    appendix = label_appendix_three_signal(panel)
    legacy = label_legacy_five_field(panel)
    rows = [
        _counts_row(
            "appendix_canonical",
            "published rule: symmetric W3, CCI bands 0.10/0.67, contamination floor vs p75; mixed=conflict; ambiguous=remainder. No greedy_succeeds.",
            count_labels(appendix),
            src.name,
        ),
        _counts_row(
            "legacy_five_field_sensitivity",
            "named sensitivity variant only: canonical>0.5 AND W3<0.2 AND high contam AND greedy_succeeds; W3>0.5 AND ACI>0.5 AND low contam. Asymmetric W3 0.2/0.5. Not published.",
            count_labels(legacy),
            src.name,
        ),
        _counts_row(
            "file_convergence_label",
            "labels already stored on the panel CSV (legacy snapshot; v3 is not overwritten)",
            count_labels(existing),
            src.name,
        ),
    ]

    overlay = _rescored_var()
    corrected = panel.merge(overlay, on=["problem_id", "model"], how="left")
    n_can_changed = int(
        (
            pd.to_numeric(corrected["VAR_canonical"], errors="coerce")
            != pd.to_numeric(corrected["VAR_canonical_rescored"], errors="coerce")
        ).sum()
    )
    n_w3_changed = int(
        (
            pd.to_numeric(corrected["VAR_W3"], errors="coerce")
            != pd.to_numeric(corrected["VAR_W3_rescored"], errors="coerce")
        ).sum()
    )
    corrected["VAR_canonical"] = pd.to_numeric(corrected["VAR_canonical_rescored"], errors="coerce")
    corrected["VAR_W3"] = pd.to_numeric(corrected["VAR_W3_rescored"], errors="coerce")
    required = ["VAR_canonical", "VAR_W3", "instance_contamination_score", "greedy_succeeds"]
    corrected["missing_core"] = corrected[required].isna().any(axis=1)
    corrected["instance_rank_pct"] = corrected.groupby("problem_subtype")[
        "instance_contamination_score"
    ].rank(method="average", pct=True)

    exec_c = label_default(corrected)
    app_c = label_appendix_three_signal(corrected)
    legacy_c = label_legacy_five_field(corrected)
    rows.append(
        _counts_row(
            "appendix_canonical_on_rescored_p1",
            "canonical appendix rule after overlaying included=True rescored VAR_canonical / VAR_W3",
            count_labels(app_c),
            src.name + "+rescored_p1",
        )
    )
    rows.append(
        _counts_row(
            "legacy_five_field_on_rescored_p1",
            "sensitivity variant after the same P1 overlay",
            count_labels(legacy_c),
            src.name + "+rescored_p1",
        )
    )
    rows.append(
        _counts_row(
            "reference_current_paper_working_set",
            "user-stated current labels to beat {ambiguous 205, mixed 115, retrieval 6, computation 4}",
            {**CURRENT_REF, "n": sum(CURRENT_REF.values())},
            "working_set",
        )
    )

    three_models = {
        "anthropic/claude-sonnet-4",
        "openai/gpt-4o",
        "meta-llama/llama-3.1-8b-instruct",
    }
    three = corrected[corrected["model"].astype(str).isin(three_models)].copy()
    if len(three):
        three["instance_rank_pct"] = three.groupby("problem_subtype")[
            "instance_contamination_score"
        ].rank(method="average", pct=True)
        rows.append(
            _counts_row(
                "executed_on_rescored_p1_3model",
                "executed rule on Claude/GPT-4o/Llama after P1 overlay (n=330)",
                count_labels(label_default(three)),
                src.name + "+rescored_p1_3model",
            )
        )
        three_orig = panel[panel["model"].astype(str).isin(three_models)].copy()
        rows.append(
            _counts_row(
                "file_convergence_label_3model",
                "stored labels on Claude/GPT-4o/Llama subset of the same panel",
                count_labels(
                    three_orig["convergence_label"].astype(str).str.replace("_signal", "", regex=False)
                ),
                src.name + "_3model",
            )
        )

    out = pd.DataFrame(rows)
    path = DERIVED / "P3_triangulation_rule_comparison_detail.csv"
    out.to_csv(path, index=False)
    print(out.to_string(index=False))
    print(f"VAR_canonical cells changed vs panel: {n_can_changed}; VAR_W3 changed: {n_w3_changed}")
    print(f"Wrote {path}")

    labeled = corrected.copy()
    labeled["label_executed_rescored"] = exec_c
    labeled["label_appendix_rescored"] = app_c
    labeled.to_csv(DERIVED / "ALGO_P3_triangulation_rescored_p1.csv", index=False)


if __name__ == "__main__":
    main()
