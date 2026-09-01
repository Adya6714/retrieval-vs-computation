#!/usr/bin/env python3
"""Emit results/derived/COVERAGE_MASTER.csv — appendix coverage table.

One row per (probe, family, model, variant): cells attempted / included /
excluded, exclusion-reason breakdown, sampling settings known vs unknown.

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

from probes.common.variants import normalize_variant  # noqa: E402

RAW = REPO_ROOT / "results/raw"
DERIVED = REPO_ROOT / "results/derived"
OUT = DERIVED / "COVERAGE_MASTER.csv"


def _boolish(v) -> bool:
    return str(v).strip().lower() in {"true", "1", "1.0", "yes"}


def _sampling(df: pd.DataFrame) -> str:
    cols = {c.lower(): c for c in df.columns}
    has_t = "temperature" in cols
    has_s = "seed" in cols
    if has_t and has_s:
        t = df[cols["temperature"]].astype(str).str.strip()
        s = df[cols["seed"]].astype(str).str.strip()
        filled = ((t != "") & (t.lower() != "nan") & (s != "") & (s.lower() != "nan")).mean()
        return "known" if filled >= 0.5 else "partial"
    if has_t or has_s:
        return "partial"
    return "unknown"


def _breakdown(counter: Counter) -> str:
    return json.dumps(dict(counter.most_common()), sort_keys=False)


def _row(
    probe: str,
    family: str,
    model: str,
    variant: str,
    attempted: int,
    included: int,
    reasons: Counter,
    sampling: str,
    source: str,
) -> dict:
    excluded = attempted - included
    top = reasons.most_common(4)
    return {
        "probe": probe,
        "family": family,
        "model": model,
        "variant": variant,
        "cells_attempted": attempted,
        "cells_included": included,
        "cells_excluded": excluded,
        "exclusion_reason_breakdown": _breakdown(reasons),
        "exclusion_reason_1": top[0][0] if top else "",
        "exclusion_reason_1_n": top[0][1] if top else 0,
        "exclusion_reason_2": top[1][0] if len(top) > 1 else "",
        "exclusion_reason_2_n": top[1][1] if len(top) > 1 else 0,
        "exclusion_reason_3": top[2][0] if len(top) > 2 else "",
        "exclusion_reason_3_n": top[2][1] if len(top) > 2 else 0,
        "sampling_settings": sampling,
        "source_file": source,
    }


def p1_family_from_name(name: str) -> str:
    if name.startswith("ALGO_"):
        return "ALGO"
    if name.startswith("BW_"):
        return "BW"
    if name.startswith("GSM_"):
        return "GSM"
    return "unknown"


def emit_p1(rows: list[dict]) -> None:
    files = sorted(DERIVED.glob("*_P1_*rescored.csv"))
    files = [p for p in files if "review" not in p.name]
    for path in files:
        df = pd.read_csv(path, dtype=str).fillna("")
        if df.empty:
            continue
        family = p1_family_from_name(path.name)
        raw_sib = RAW / path.name.replace("_rescored", "")
        sampling = _sampling(pd.read_csv(raw_sib, nrows=0)) if raw_sib.exists() else _sampling(df)
        if "variant_type_normalized" in df.columns:
            df["variant"] = df["variant_type_normalized"].map(normalize_variant)
        else:
            df["variant"] = df.get("variant_type", pd.Series("", index=df.index)).map(
                normalize_variant
            )
        for (model, variant), sub in df.groupby(["model", "variant"], dropna=False):
            reasons: Counter[str] = Counter()
            included = 0
            for _, r in sub.iterrows():
                if _boolish(r.get("included", False)):
                    included += 1
                else:
                    reasons[str(r.get("exclusion_reason") or "unspecified").strip() or "unspecified"] += 1
            rows.append(
                _row(
                    "P1",
                    family,
                    str(model),
                    str(variant),
                    len(sub),
                    included,
                    reasons,
                    sampling,
                    path.name,
                )
            )


def emit_p2(rows: list[dict]) -> None:
    cov_path = DERIVED / "P2_coverage.csv"
    if not cov_path.exists():
        return
    cov = pd.read_csv(cov_path, dtype=str).fillna("")
    sampling = "unknown"
    for _, r in cov.iterrows():
        label = str(r.get("family", "")).strip()
        if ":" in label:
            family_key, source = label.split(":", 1)
        else:
            family_key, source = label, ""
        if family_key.startswith("BW_"):
            family, variant = "BW", family_key[len("BW_") :]
        elif family_key.startswith("MBW_"):
            family, variant = "MBW", family_key[len("MBW_") :]
        elif family_key.startswith("GSM_"):
            family, variant = "GSM", family_key[len("GSM_") :]
        elif family_key.startswith("ALGO_"):
            family = "ALGO"
            variant = family_key[len("ALGO_") :]
        else:
            family, variant = family_key, "--"
        source_file = source.strip() or cov_path.name
        attempted = int(float(r["rows_attempted"])) if str(r.get("rows_attempted", "")).strip() else 0
        included = int(float(r["rows_with_usable_score"])) if str(r.get("rows_with_usable_score", "")).strip() else 0
        reasons: Counter[str] = Counter()
        for i in (1, 2, 3):
            reason = str(r.get(f"null_reason_{i}", "")).strip()
            n = str(r.get(f"null_reason_{i}_n", "")).strip()
            if reason and n and n not in {"0", "0.0"}:
                reasons[reason] += int(float(n))
        rows.append(
            _row(
                "P2",
                family,
                str(r.get("model", "")),
                variant or "--",
                attempted,
                included,
                reasons,
                sampling,
                source_file,
            )
        )


def emit_p3(rows: list[dict]) -> None:
    for path, family, variant in (
        (RAW / "ALGO_P3_contamination.csv", "ALGO", "contamination"),
        (RAW / "BW_P3_contamination.csv", "BW", "contamination"),
        (RAW / "GSM_P3_contamination.csv", "GSM", "contamination"),
    ):
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        rows.append(
            _row(
                "P3",
                family,
                "--",
                variant,
                len(df),
                len(df),
                Counter(),
                _sampling(df),
                path.name,
            )
        )
    for path, family in (
        (RAW / "ALGO_P3_mechanistic.csv", "ALGO"),
        (RAW / "BW_P3_mechanistic.csv", "BW"),
        (RAW / "GSM_P3_mechanistic.csv", "GSM"),
    ):
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        sampling = _sampling(df)
        variant_col = "variant_type" if "variant_type" in df.columns else None
        if "model" not in df.columns:
            rows.append(
                _row("P3", family, "--", "mechanistic", len(df), len(df), Counter(), sampling, path.name)
            )
            continue
        group_cols = ["model"] + ([variant_col] if variant_col else [])
        for key, sub in df.groupby(group_cols, dropna=False):
            if variant_col:
                model, variant = key
            else:
                model, variant = key, "mechanistic"
            rows.append(
                _row(
                    "P3",
                    family,
                    str(model),
                    str(variant or "mechanistic"),
                    len(sub),
                    len(sub),
                    Counter(),
                    sampling,
                    path.name,
                )
            )


def main() -> None:
    rows: list[dict] = []
    emit_p1(rows)
    emit_p2(rows)
    emit_p3(rows)
    out = pd.DataFrame(rows)
    out = out.sort_values(["probe", "family", "model", "variant"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)
    n_unknown = int((out["sampling_settings"] == "unknown").sum())
    n_known = int((out["sampling_settings"] == "known").sum())
    print(
        f"Wrote {OUT} ({len(out)} rows; "
        f"sampling known={n_known} unknown={n_unknown}; "
        f"P1={int((out.probe=='P1').sum())} P2={int((out.probe=='P2').sum())} "
        f"P3={int((out.probe=='P3').sum())})"
    )


if __name__ == "__main__":
    main()
