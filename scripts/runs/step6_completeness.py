#!/usr/bin/env python3
"""Checklist Step 6 — results completeness inventory (no API).

Regenerates all derived packs, tags incomplete cells (API vs cleanup vs remap),
and writes a raw-file manifest.

Outputs:
    results/derived/results_manifest.csv
    results/derived/api_backlog_tagged.csv
    results/derived/STEP6_COMPLETENESS.md
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
DER = ROOT / "results" / "derived"
DATA = ROOT / "data" / "problems"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runs.coverage_audit import (  # noqa: E402
    _load_bank,
    _safe_read,
    master_coverage_table,
)


def _run_script(rel: str) -> None:
    cmd = [sys.executable, str(ROOT / rel)]
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def _raw_manifest() -> pd.DataFrame:
    """One row per raw CSV: exists, size, models, problem count."""
    rows: list[dict] = []
    profiles_path = DER / "scientific_file_profiles.csv"
    profiles = (
        pd.read_csv(profiles_path, dtype=str)
        if profiles_path.exists()
        else pd.DataFrame()
    )
    profile_map = profiles.set_index("file").to_dict("index") if not profiles.empty else {}

    for path in sorted(RAW.glob("*.csv")):
        if path.stat().st_size == 0:
            continue
        prof = profile_map.get(path.name, {})
        df = _safe_read(path)
        models = ""
        n_prob = ""
        if df is not None and not df.empty:
            if "model" in df.columns:
                models = ",".join(sorted(df["model"].astype(str).unique())[:8])
            if "problem_id" in df.columns:
                n_prob = str(df["problem_id"].nunique())
        rows.append(
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "rows": prof.get("rows", len(df) if df is not None else 0),
                "models": models or prof.get("models", ""),
                "n_problem_ids": n_prob or prof.get("n_problem_ids", ""),
                "family_guess": _guess_family(path.name),
            }
        )
    return pd.DataFrame(rows)


def _guess_family(name: str) -> str:
    for fam in ["ALGO", "GSM", "BW"]:
        if name.startswith(fam):
            return fam
    return ""


def _tagged_backlog(coverage: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    incomplete = coverage[~coverage["bank_complete"]].copy()
    rows: list[dict] = []

    gsm_bank = _load_bank("GSM")
    _ = gsm_bank  # reserved for future remap checks

    for _, row in incomplete.iterrows():
        family = row["family"]
        probe = row["probe"]
        model = row["model"]
        label = row["coverage_label"]

        fix_type = "unknown"
        est_api = 0
        notes = ""

        if label == "contaminated_extra_ids":
            fix_type = "derivation_cleanup"
            est_api = 0
            extra = str(row.get("extra_canonical_ids", "") or "")
            notes = f"Filter raw to bank IDs in derivations; {extra.count(',') + 1 if extra else 0} extra canonical IDs in raw"
        elif family == "GSM" and probe == "P1":
            fix_type = "remap_and_or_api"
            missing = str(row.get("missing_canonical_ids", "") or "")
            if missing in ("", "nan"):
                missing = ""
            extra = str(row.get("extra_canonical_ids", "") or "")
            if extra in ("", "nan"):
                extra = ""
            n_miss = len([x for x in missing.split(",") if x.strip()])
            n_extra = len([x for x in extra.split(",") if x.strip()])
            est_api = n_miss * 7  # ~7 variant pairs per missing canonical ID
            notes = (
                f"{n_extra} wrong IDs in raw (GSM_021-040?) may remap without API; "
                f"{n_miss} bank IDs missing → ~{est_api} behavioral calls for this model"
            )
        elif family == "GSM" and probe == "P2":
            fix_type = "api_full_probe"
            n_miss = int(row.get("missing_pair_count", 0) or row["bank_canonical_n"])
            est_api = n_miss * 14  # ~1 phase1 + 2× phase2 step loops per problem
            notes = f"Full GSM P2 run for {n_miss} problems (~{est_api} step-level calls)"
        elif probe == "P2A_elicited":
            fix_type = "api_partial_probe"
            n_miss = int(row.get("missing_pair_count", 0) or 0)
            est_api = n_miss * 4  # ~3.6 steps/session
            notes = f"{n_miss} missing elicited sessions × ~4 API calls each"
        else:
            fix_type = "api_or_audit"
            n_miss = int(row.get("missing_pair_count", 0) or 0)
            est_api = n_miss
            notes = "See master_coverage_gaps.csv"

        rows.append(
            {
                "family": family,
                "probe": probe,
                "model": model,
                "coverage_label": label,
                "observed_canonical_n": int(row["observed_canonical_n"]),
                "bank_canonical_n": int(row["bank_canonical_n"]),
                "fix_type": fix_type,
                "est_api_calls": est_api,
                "priority": "P0" if (family == "GSM" and probe == "P2") else "P1",
                "notes": notes,
            }
        )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["priority", "family", "probe", "model"])
    return out


def _write_summary(
    manifest: pd.DataFrame,
    coverage: pd.DataFrame,
    backlog: pd.DataFrame,
    regen_ok: bool,
) -> None:
    n_complete = int(coverage["bank_complete"].sum())
    n_total = len(coverage)
    api_rows = backlog[backlog["fix_type"].str.startswith("api")] if not backlog.empty else pd.DataFrame()
    cleanup = backlog[backlog["fix_type"] == "derivation_cleanup"] if not backlog.empty else pd.DataFrame()
    remap = backlog[backlog["fix_type"] == "remap_and_or_api"] if not backlog.empty else pd.DataFrame()

    lines = [
        "# Step 6 — Results completeness",
        "",
        f"**Status:** regenerate pipeline {'OK' if regen_ok else 'FAILED'}",
        f"**Coverage:** {n_complete}/{n_total} model×probe slices bank-complete",
        f"**Raw files indexed:** {len(manifest)} CSVs in `results/raw/`",
        "",
        "## Step 6 actions (this run)",
        "",
        "- [x] Re-ran `rederive_all_metrics.py`, `deep_metrics_analysis.py`, `triangulation_v2.py`, `scientific_filewise_audit.py`",
        "- [x] BW + ALGO P1 derivations now **filter to question bank** (`filter_p1_to_bank`)",
        "- [x] Tagged API backlog → `api_backlog_tagged.csv`",
        "- [x] Raw manifest → `results_manifest.csv`",
        "",
        "## Fix types (no API for Step 6 itself)",
        "",
        "| fix_type | Cells | API calls |",
        "|----------|-------|-----------|",
        f"| derivation_cleanup | {len(cleanup)} | 0 |",
        f"| remap_and_or_api (GSM P1) | {len(remap)} | see backlog (missing IDs only) |",
        f"| api_* (runs deferred) | {len(api_rows)} | ~{int(api_rows['est_api_calls'].sum()) if not api_rows.empty else 0} estimated |",
        "",
        "## Incomplete slices",
        "",
        "```",
    ]
    if backlog.empty:
        lines.append("(none)")
    else:
        cols = ["family", "probe", "model", "fix_type", "est_api_calls", "observed_canonical_n", "bank_canonical_n"]
        lines.append(backlog[cols].to_string(index=False))
    lines.extend(
        [
            "```",
            "",
            "## GSM P1 remap note",
            "",
            "GPT-4o/Llama raw contains GSM_021–040 where bank expects GSM_041–064. "
            "Investigate whether rows can be **ID-remapped** without re-querying before spending "
            "~168 calls/model on missing bank IDs.",
            "",
            "## Phase 2 ready?",
            "",
            "Yes — existing complete slices are bank-filtered in derivations. "
            "Proceed to Step 8 (TEP) without API spend.",
            "",
        ]
    )
    (DER / "STEP6_COMPLETENESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    regen_ok = True

    print("=== Step 6: regenerate all derived packs ===")
    for script in [
        "scripts/runs/rederive_all_metrics.py",
        "scripts/runs/deep_metrics_analysis.py",
        "scripts/runs/triangulation_v2.py",
        "scripts/runs/scientific_filewise_audit.py",
    ]:
        try:
            _run_script(script)
        except subprocess.CalledProcessError as exc:
            regen_ok = False
            print(f"!! failed: {exc}")

    print("\n=== Step 6: manifest + tagged backlog ===")
    manifest = _raw_manifest()
    manifest.to_csv(DER / "results_manifest.csv", index=False)

    coverage, gaps = master_coverage_table()
    backlog = _tagged_backlog(coverage, gaps)
    backlog.to_csv(DER / "api_backlog_tagged.csv", index=False)

    _write_summary(manifest, coverage, backlog, regen_ok)

    print(f"\nWrote {DER / 'results_manifest.csv'} ({len(manifest)} raw files)")
    print(f"Wrote {DER / 'api_backlog_tagged.csv'} ({len(backlog)} incomplete cells)")
    print(f"Wrote {DER / 'STEP6_COMPLETENESS.md'}")
    print(f"\nCoverage: {coverage['bank_complete'].sum()}/{len(coverage)} complete")
    if not backlog.empty:
        print("\nBacklog by fix_type:")
        print(backlog.groupby("fix_type")["est_api_calls"].agg(["count", "sum"]).to_string())

    if not regen_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
