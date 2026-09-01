#!/usr/bin/env python3
"""Score TEP on steps completed before abort instead of nulling the session.

Reads the E2 remapped TEP traces and the original raw TEP file. Does not
call any API. Does not write results/raw/.

skip_count > 5 still marks the session aborted (aborted_at_step records
where), but TEP is computed from the cascade collected before the abort.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RAW = REPO_ROOT / "results/raw"
DERIVED = REPO_ROOT / "results/derived"


def _json_list(raw) -> list:
    try:
        val = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
    except json.JSONDecodeError:
        return []
    return val if isinstance(val, list) else []


def compute_tep(cascade) -> float | None:
    adapted = sum(1 for s in cascade if isinstance(s, dict) and s.get("classification") == "adapted")
    resistant = sum(
        1 for s in cascade if isinstance(s, dict) and s.get("classification") == "resistant"
    )
    denom = adapted + resistant
    return round(adapted / denom, 4) if denom > 0 else None


def aborted_at(cascade, status: str):
    if not str(status).startswith("aborted:"):
        return ""
    steps = [s.get("step") for s in cascade if isinstance(s, dict) and s.get("step") is not None]
    return steps[-1] if steps else ""


def apply_e3(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        cascade = _json_list(row.get("cascade_sequence_json"))
        tep = compute_tep(cascade)
        out = row.to_dict()
        out["tep"] = "" if tep is None else f"{tep:.4f}"
        out["aborted_at_step"] = aborted_at(cascade, str(row.get("session_status", "")))
        out["tep_scored_despite_abort"] = str(
            str(row.get("session_status", "")).startswith("aborted:") and tep is not None
        )
        rows.append(out)
    return pd.DataFrame(rows)


def n_usable(s: pd.Series) -> int:
    return int(pd.to_numeric(s, errors="coerce").notna().sum())


def main() -> None:
    raw = pd.read_csv(RAW / "BW_P2_tep.csv", dtype=str).fillna("")
    remapped = pd.read_csv(DERIVED / "BW_P2_tep_nl_rescored.csv", dtype=str).fillna("")
    if len(raw) != len(remapped):
        raise ValueError(f"row count mismatch raw={len(raw)} remapped={len(remapped)}")

    raw_usable = n_usable(raw["tep"])
    raw_null = len(raw) - raw_usable
    remap_usable = n_usable(remapped["tep"])

    e3_raw = apply_e3(raw)
    e3_remap = apply_e3(remapped)

    e3_raw_usable = n_usable(e3_raw["tep"])
    e3_remap_usable = n_usable(e3_remap["tep"])

    # Align on original null mask
    orig_null = pd.to_numeric(raw["tep"], errors="coerce").isna()
    of_534_e3_raw = int(pd.to_numeric(e3_raw.loc[orig_null, "tep"], errors="coerce").notna().sum())
    of_534_e3_remap = int(
        pd.to_numeric(e3_remap.loc[orig_null, "tep"], errors="coerce").notna().sum()
    )
    of_534_remap_only = int(
        pd.to_numeric(remapped.loc[orig_null, "tep"], errors="coerce").notna().sum()
    )

    out_path = DERIVED / "BW_P2_tep_abort_separated.csv"
    e3_remap.to_csv(out_path, index=False)

    report = pd.DataFrame(
        [
            {
                "raw_rows": len(raw),
                "raw_usable_tep": raw_usable,
                "raw_null_tep": raw_null,
                "e2_remap_usable_still_abort_null": remap_usable,
                "e3_on_original_traces_usable": e3_raw_usable,
                "e3_on_remapped_traces_usable": e3_remap_usable,
                "of_534_nulls_scoreable_e3_original": of_534_e3_raw,
                "of_534_nulls_scoreable_e2_remap_abort_still_nulls": of_534_remap_only,
                "of_534_nulls_scoreable_e2_remap_plus_e3": of_534_e3_remap,
                "aborted_sessions": int(
                    remapped["session_status"].astype(str).str.startswith("aborted:").sum()
                ),
                "out": str(out_path.relative_to(REPO_ROOT)),
            }
        ]
    )
    report_path = DERIVED / "P2_bw_tep_abort_separation_report.csv"
    report.to_csv(report_path, index=False)
    print(f"Wrote {out_path}")
    print(f"Wrote {report_path}")
    print(report.T.to_string(header=False))


if __name__ == "__main__":
    main()
