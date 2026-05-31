#!/usr/bin/env python3
"""Master pre-API data audit — missing, unusable, and API budget.

Consolidates coverage gaps, analysis-exclusion flags, and estimated API
calls so you can evaluate everything before spending on runs.

Outputs:
    results/derived/PRE_API_MASTER_AUDIT.md
    results/derived/pre_api_slice_inventory.csv
    results/derived/pre_api_missing_ids.csv
    results/derived/pre_api_unusable_flags.csv
    results/derived/pre_api_api_budget.csv

Regenerate:
    python scripts/runs/pre_api_master_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
DATA = ROOT / "data" / "problems"
DER = ROOT / "results" / "derived"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runs.coverage_audit import (  # noqa: E402
    MODELS,
    SHORT,
    P1_TAGS,
    _bank_canonical_ids,
    _load_bank,
    _safe_read,
    filter_p1_to_bank,
    load_gsm_p2_merged,
    master_coverage_table,
)

# Calls per unit (conservative estimates from step6 / run scripts)
CALLS_PER_GSM_P1_PROBLEM = 7  # canonical + W1–W6 variants
CALLS_PER_GSM_P2_PROBLEM = 14  # phase1 + phase2 steps (~616/44)
CALLS_PER_ALGO_P2A_SESSION = 4  # phase1 elicitation + phase2 normal steps


def _gsm_p1_wrong_ids_are_duplicates(tag: str) -> tuple[int, int]:
    """Return (n_wrong_ids, n_matching_GSM_001_020_answers)."""
    path = RAW / f"GSM_P1_behavioral_{tag}.csv"
    df = _safe_read(path)
    if df is None or df.empty:
        return 0, 0
    bank = _load_bank("GSM")
    bank_ids = set(bank["problem_id"])
    canon = df[df["variant_type"].astype(str).str.lower() == "canonical"].drop_duplicates("problem_id")
    wrong = sorted(set(canon["problem_id"]) - bank_ids)
    matches = 0
    for wid in wrong:
        if not wid.startswith("GSM_"):
            continue
        try:
            num = int(wid.split("_")[1])
        except ValueError:
            continue
        if num < 21 or num > 40:
            continue
        ref = f"GSM_{num - 20:03d}"
        w = canon[canon["problem_id"] == wid]
        r = canon[canon["problem_id"] == ref]
        if w.empty or r.empty:
            continue
        if str(w.iloc[0].get("model_answer", "")) == str(r.iloc[0].get("model_answer", "")):
            matches += 1
    return len(wrong), matches


def _gsm_wrong_ids_in_raw(model_tag: str) -> list[str]:
    path = RAW / f"GSM_P1_behavioral_{model_tag}.csv"
    df = _safe_read(path)
    if df is None or df.empty:
        return []
    bank = _load_bank("GSM")
    bank_ids = set(bank["problem_id"])
    canon = df[df["variant_type"].astype(str).str.lower() == "canonical"]
    return sorted(set(canon["problem_id"]) - bank_ids)


def _recovery_inventory() -> pd.DataFrame:
    """Check logs / alternate raw files before scheduling API runs."""
    rows: list[dict] = []

    def add(**kw: object) -> None:
        rows.append(kw)  # type: ignore[arg-type]

    o4 = load_gsm_p2_merged()
    o4_sub = o4[o4["model"] == "openai/o4-mini"] if not o4.empty and "model" in o4.columns else pd.DataFrame()
    bank_n = len(_bank_canonical_ids(_load_bank("GSM")))
    o4_n = int(o4_sub["problem_id"].nunique()) if not o4_sub.empty else 0
    add(
        gap="GSM P2 o4-mini",
        status="recovered_no_api",
        source_file="results/raw/GSM_P2_phase1_o1mini.csv",
        log_ref="results/raw/new_model_sweep_logs/finish_o4mini.log (44/44 complete 2026-05-24)",
        observed=f"{o4_n}/{bank_n}",
        est_api_saved=616 if o4_n >= bank_n else 0,
        action="Merge into GSM P2 derivations; coverage now counts o4-mini from this file",
    )

    for model, tag in [("GPT-4o", "gpt4o"), ("Llama", "llama")]:
        n_wrong, n_dup = _gsm_p1_wrong_ids_are_duplicates(tag)
        bank = _load_bank("GSM")
        df = _safe_read(RAW / f"GSM_P1_behavioral_{tag}.csv")
        canon = df[df["variant_type"].astype(str).str.lower() == "canonical"] if df is not None else pd.DataFrame()
        bank_ids = set(_bank_canonical_ids(bank))
        valid_n = len(set(canon["problem_id"]) & bank_ids) if not canon.empty else 0
        missing = len(bank_ids - set(canon["problem_id"])) if not canon.empty else bank_n
        add(
            gap=f"GSM P1 {model}",
            status="remap_invalid" if n_dup == n_wrong and n_wrong > 0 else "needs_api",
            source_file=f"results/raw/GSM_P1_behavioral_{tag}.csv",
            log_ref="No archived logs for GSM_041–064",
            observed=f"{valid_n}/{bank_n} bank-valid",
            est_api_saved=0,
            action=(
                f"Exclude {n_wrong} duplicate IDs (GSM_021–040 = GSM_001–020); "
                f"API still needed for {missing} missing IDs (GSM_041–064)"
            ),
        )

    elic = _safe_read(RAW / "ALGO_P2_phase2_normal_elicited.csv")
    ref_n = 110
    if elic is not None and "model" in elic.columns:
        ref_n = int(elic[elic["model"] == "openai/gpt-4o"]["problem_id"].nunique())
    for m, short in [
        ("anthropic/claude-sonnet-4", "Claude"),
        ("google/gemini-2.5-flash", "Gemini"),
        ("meta-llama/llama-3.1-8b-instruct", "Llama"),
    ]:
        obs = int(elic[elic["model"] == m]["problem_id"].nunique()) if elic is not None else 0
        miss = max(ref_n - obs, 0)
        add(
            gap=f"ALGO P2A elicited {short}",
            status="needs_api" if obs < ref_n else "recovered_no_api",
            source_file="results/raw/ALGO_P2_phase2_normal_elicited.csv",
            log_ref="No separate elicited logs; 61/110 pilot (CC_031+ missing)",
            observed=f"{obs}/{ref_n} problems",
            est_api_saved=0,
            action=f"~{miss} sessions × {CALLS_PER_ALGO_P2A_SESSION} steps ≈ {miss * CALLS_PER_ALGO_P2A_SESSION} API calls",
        )

    for short in ["Gemini", "o4-mini"]:
        add(
            gap=f"BW P2 {short}",
            status="needs_api",
            source_file="results/raw/BW_P2_tep.csv (pilot)",
            log_ref="No BW P2 sweep logs for these models",
            observed="0/65",
            est_api_saved=0,
            action="~1000 calls/model for 50-problem pilot (Step 20 optional)",
        )

    for fam, fname in [("ALGO", "ALGO_P1_behavioral_o1mini.csv"), ("BW", "BW_P1_behavioral_o1mini.csv")]:
        n_err = _count_errors(RAW / fname)
        if n_err:
            add(
                gap=f"{fam} P1 o4-mini ERROR rows",
                status="resume_not_full_rerun",
                source_file=f"results/raw/{fname}",
                log_ref="finish_o4mini.log — use --resume before full rerun",
                observed=f"{n_err} ERROR rows",
                est_api_saved=0,
                action="Retry ERROR rows only on existing output path",
            )

    return pd.DataFrame(rows)


def _count_errors(path: Path) -> int:
    df = _safe_read(path)
    if df is None or df.empty:
        return 0
    for col in ("raw_response", "model_answer"):
        if col in df.columns:
            return int(df[col].astype(str).str.startswith("ERROR:").sum())
    return 0


def _bw_p2_inventory() -> list[dict]:
    """BW P2 is a partial pilot — not in master_coverage_table."""
    bank = _load_bank("BW")
    expected = sorted(_bank_canonical_ids(bank))
    rows: list[dict] = []
    for fname, probe in [
        ("BW_P2_tep.csv", "P2_tep"),
        ("BW_P2_cci.csv", "P2_cci"),
        ("BW_P2_plans.csv", "P2_plans"),
    ]:
        df = _safe_read(RAW / fname)
        if df is None:
            continue
        for model in MODELS:
            sub = df[df["model"] == model] if "model" in df.columns else pd.DataFrame()
            observed = set(sub["problem_id"]) if not sub.empty else set()
            missing = set(expected) - observed
            rows.append(
                {
                    "family": "BW",
                    "probe": probe,
                    "model": SHORT[model],
                    "bank_canonical_n": len(expected),
                    "observed_canonical_n": len(observed & set(expected)),
                    "missing_canonical_n": len(missing),
                    "bank_complete": len(missing) == 0 and bool(observed),
                    "coverage_label": (
                        "full_bank"
                        if len(missing) == 0 and bool(observed)
                        else ("missing_model" if not observed else "partial")
                    ),
                    "missing_canonical_ids": ",".join(sorted(missing)),
                    "in_master_coverage_table": False,
                }
            )
    return rows


def _tep_unusable_summary() -> pd.DataFrame:
    path = DER / "tep_dissociation_sessions.csv"
    if not path.exists():
        return pd.DataFrame()
    s = pd.read_csv(path)
    if "tep_valid" not in s.columns:
        s["tep_valid"] = s["tep"].notna()
    rows = []
    for (fam, model), g in s.groupby(["family", "model_short"]):
        rows.append(
            {
                "family": fam,
                "probe": "P2_tep",
                "model": model,
                "n_sessions": len(g),
                "tep_valid_n": int(g["tep_valid"].sum()),
                "tep_missing_n": int((~g["tep_valid"]).sum()),
                "final_ok_rate": round(float(g["final_ok"].mean()), 3),
                "issue": (
                    "tep_mostly_missing"
                    if g["tep_valid"].mean() < 0.5
                    else ("final_ok_constant" if g["final_ok"].nunique() < 2 else "")
                ),
                "analysis_use": (
                    "exclude_or_scope"
                    if g["tep_valid"].mean() < 0.5 or g["final_ok"].nunique() < 2
                    else "ok"
                ),
            }
        )
    return pd.DataFrame(rows)


def _unusable_flags(coverage: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def add(**kw: object) -> None:
        rows.append(kw)  # type: ignore[arg-type]

    # GSM P1 duplicate wrong IDs (not valid remap to GSM_041–060)
    for tag, model in [("gpt4o", "GPT-4o"), ("llama", "Llama")]:
        wrong = _gsm_wrong_ids_in_raw(tag)
        n_wrong, n_dup = _gsm_p1_wrong_ids_are_duplicates(tag)
        if wrong:
            add(
                family="GSM",
                probe="P1",
                model=model,
                flag="duplicate_wrong_ids",
                severity="high",
                n_affected=len(wrong),
                detail=(
                    f"{len(wrong)} canonical IDs GSM_021–040 not in bank; "
                    f"{n_dup}/{n_wrong} duplicate GSM_001–020 answers — exclude from bank stats, do not remap to 041–060"
                ),
                fix_type="exclude_from_bank",
                blocks_analysis="P1 accuracy/VRI for 24 bank IDs GSM_041–064 still missing",
            )

    # BW P1 contamination in combined file
    combined = _safe_read(RAW / "BW_P1_behavioral.csv")
    if combined is not None and "problem_id" in combined.columns:
        bw_bank = _bank_canonical_ids(_load_bank("BW"))
        gsm_ids = {p for p in combined["problem_id"].unique() if str(p).startswith("GSM_")}
        if gsm_ids:
            add(
                family="BW",
                probe="P1",
                model="Claude,GPT-4o,Llama",
                flag="cross_family_contamination",
                severity="medium",
                n_affected=len(gsm_ids),
                detail=f"BW_P1_behavioral.csv contains {len(gsm_ids)} GSM problem IDs — filter in derivations (fixed) but raw file unclean",
                fix_type="derivation_filter",
                blocks_analysis="None if bank-filtered; inflates raw row counts",
            )

    # ERROR responses
    for family in ("GSM", "ALGO", "BW"):
        for model in MODELS:
            tag = P1_TAGS[model]
            path = RAW / f"{family}_P1_behavioral_{tag}.csv"
            if family == "BW" and not path.exists():
                path = RAW / "BW_P1_behavioral.csv"
            if not path.exists():
                continue
            n_err = _count_errors(path)
            if n_err:
                add(
                    family=family,
                    probe="P1",
                    model=SHORT[model],
                    flag="api_error_rows",
                    severity="low",
                    n_affected=n_err,
                    detail=f"{n_err} ERROR: rows in raw — excluded from coverage counts",
                    fix_type="re_run_errors",
                    blocks_analysis=f"{n_err} variant rows unusable",
                )

    # Incomplete slices from coverage
    inc = coverage[~coverage["bank_complete"].astype(str).str.lower().eq("true")]
    for _, row in inc.iterrows():
        miss_ids = str(row.get("missing_canonical_ids", "") or "")
        miss_canon = len([x for x in miss_ids.split(",") if x.strip()])
        miss_pairs = int(row.get("missing_pair_count") or 0)
        add(
            family=row["family"],
            probe=row["probe"],
            model=row["model"],
            flag="incomplete_coverage",
            severity="high" if row.get("observed_canonical_n", 0) == 0 else "medium",
            n_affected=miss_canon or miss_pairs,
            detail=(
                f"{row.get('observed_canonical_n')}/{row.get('bank_canonical_n')} bank-valid; "
                f"label={row.get('coverage_label')}"
            ),
            fix_type="api_run",
            blocks_analysis="Any claim needing full bank denominator",
        )

    # BW P2 pilot scope
    bw_p2 = _bw_p2_inventory()
    for r in bw_p2:
        if r["coverage_label"] != "full_bank":
            add(
                family="BW",
                probe=r["probe"],
                model=r["model"],
                flag="pilot_partial" if r["coverage_label"] == "partial" else "missing_model",
                severity="high",
                n_affected=r["missing_canonical_n"],
                detail=(
                    f"{r['observed_canonical_n']}/{r['bank_canonical_n']} problems; "
                    f"BW P2 only 3/5 models — not in master_coverage_table"
                ),
                fix_type="api_run" if r["coverage_label"] == "missing_model" else "pilot_scope",
                blocks_analysis="Five-model BW P2 claims; full bank comparisons",
            )

    # TEP analysis exclusions
    tep = _tep_unusable_summary()
    for _, r in tep.iterrows():
        if r["analysis_use"] != "ok":
            add(
                family=r["family"],
                probe="P2",
                model=r["model"],
                flag="tep_or_final_unusable",
                severity="medium",
                n_affected=int(r["tep_missing_n"]),
                detail=(
                    f"TEP valid {r['tep_valid_n']}/{r['n_sessions']}; "
                    f"final_ok_rate={r['final_ok_rate']}; {r['issue']}"
                ),
                fix_type="protocol_fix_or_scope",
                blocks_analysis="TEP dissociation / BW P2 process claims for this slice",
            )

    # ALGO P2B scoped to n=61 adversarial (by design)
    add(
        family="ALGO",
        probe="P2B_plausible",
        model="all",
        flag="scoped_denominator",
        severity="info",
        n_affected=49,
        detail="Injection runs cover 61/110 problems (adversarial subset) — not missing data",
        fix_type="none",
        blocks_analysis="Do not compare P2B counts to 110-problem P2A without label",
    )

    # P3 mechanistic single architecture
    for fam, path in [("GSM", "GSM_P3_mechanistic.csv"), ("BW", "BW_P3_mechanistic.csv"), ("ALGO", "ALGO_P3_mechanistic.csv")]:
        df = _safe_read(RAW / path)
        if df is None:
            continue
        n_prob = df["problem_id"].nunique() if "problem_id" in df.columns else len(df)
        models = df["model"].unique().tolist() if "model" in df.columns else []
        add(
            family=fam,
            probe="P3_mechanistic",
            model=",".join(str(m) for m in models[:2]),
            flag="single_arch_pilot",
            severity="info",
            n_affected=n_prob,
            detail=f"n={n_prob} problems on {models[0] if models else 'unknown'} — exploratory only",
            fix_type="local_compute",
            blocks_analysis="Cross-model mechanistic claims",
        )

    return pd.DataFrame(rows)


def _api_budget(coverage: pd.DataFrame, missing_long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def add_row(
        family: str,
        probe: str,
        model: str,
        task: str,
        units: int,
        calls_per: int,
        fix_type: str,
        priority: str,
        notes: str,
    ) -> None:
        rows.append(
            {
                "family": family,
                "probe": probe,
                "model": model,
                "task": task,
                "units": units,
                "est_api_calls": units * calls_per,
                "fix_type": fix_type,
                "priority": priority,
                "notes": notes,
            }
        )

    # From incomplete coverage table
    inc = coverage[~coverage["bank_complete"].astype(str).str.lower().eq("true")]
    for _, row in inc.iterrows():
        fam, probe, model = row["family"], row["probe"], row["model"]
        miss_ids = str(row.get("missing_canonical_ids", "") or "")
        miss_canon = len([x for x in miss_ids.split(",") if x.strip()])
        miss_pairs = int(row.get("missing_pair_count") or 0)
        miss_n = miss_canon if miss_canon else miss_pairs

        if fam == "GSM" and probe == "P1" and miss_n > 0:
            add_row(
                fam,
                probe,
                model,
                "P1 behavioral missing bank IDs (GSM_041–064)",
                miss_n,
                CALLS_PER_GSM_P1_PROBLEM,
                "api_partial_p1",
                "P1",
                "GSM_021–040 are duplicates of 001–020 — exclude; no zero-API remap",
            )

        elif fam == "GSM" and probe == "P2" and miss_n > 0:
            add_row(
                fam,
                probe,
                model,
                "Full GSM P2 probe",
                miss_n,
                CALLS_PER_GSM_P2_PROBLEM,
                "api_full_probe",
                "P0",
                f"Complete GSM P2 for {model}: {miss_n} problems",
            )

        elif fam == "ALGO" and probe == "P2A_elicited" and miss_n > 0:
            add_row(
                fam,
                probe,
                model,
                "ALGO P2A elicited sessions",
                miss_n,
                CALLS_PER_ALGO_P2A_SESSION,
                "api_partial_probe",
                "P1",
                f"{miss_n} missing elicited sessions (61/110 have data)",
            )

    # BW P2 gaps (not in coverage table)
    bw_p2 = _bw_p2_inventory()
    for r in bw_p2:
        if r["probe"] != "P2_tep":
            continue
        if r["coverage_label"] == "missing_model":
            add_row(
                "BW", "P2", r["model"],
                "BW P2 full model run (50-problem pilot)",
                50,
                20,
                "api_full_probe",
                "P1",
                "Gemini/o4-mini missing entirely from BW P2 pilot",
            )
        elif r["missing_canonical_n"] > 0 and r["model"] == "Claude":
            add_row(
                "BW", "P2", "all_models",
                "BW P2 extend to full bank (65 problems)",
                15,
                20,
                "api_partial_probe",
                "P2",
                "Pilot uses 50/65 BW problems — 15 IDs missing per model",
            )

    # No zero-API remap rows — GSM_021–040 verified as duplicates of GSM_001–020

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.drop_duplicates(
        subset=["family", "probe", "model", "task"], keep="first"
    ).sort_values(["priority", "family", "probe", "model"])


def _missing_long(gaps: pd.DataFrame) -> pd.DataFrame:
    if gaps.empty:
        return gaps
    out = gaps.copy()
    out["api_needed"] = out.apply(
        lambda r: (
            "exclude_duplicate"
            if r["family"] == "GSM"
            and r["probe"] == "P1"
            and str(r["gap_id"]).startswith("GSM_02")
            else "yes"
            if r["gap_kind"].startswith("missing")
            and not (r["family"] == "GSM" and r["probe"] == "P1" and r["gap_kind"] == "missing_w6")
            else "no"
        ),
        axis=1,
    )
    return out


def write_markdown(
    coverage: pd.DataFrame,
    bw_p2: list[dict],
    unusable: pd.DataFrame,
    budget: pd.DataFrame,
    missing: pd.DataFrame,
    tep: pd.DataFrame,
) -> None:
    n_slices = len(coverage)
    n_complete = int(coverage["bank_complete"].astype(str).str.lower().eq("true").sum())
    total_api = int(budget["est_api_calls"].sum()) if not budget.empty else 0
    api_only = budget[budget["fix_type"].str.startswith("api", na=False)] if not budget.empty else pd.DataFrame()
    core_api = budget[
        budget["family"].isin(["GSM", "ALGO"])
        & budget["fix_type"].str.startswith("api", na=False)
    ] if not budget.empty else pd.DataFrame()
    bw_api = budget[
        (budget["family"] == "BW") & budget["fix_type"].str.startswith("api", na=False)
    ] if not budget.empty else pd.DataFrame()
    api_calls = int(api_only["est_api_calls"].sum()) if not api_only.empty else 0
    core_calls = int(core_api["est_api_calls"].sum()) if not core_api.empty else 0
    bw_calls = int(bw_api["est_api_calls"].sum()) if not bw_api.empty else 0

    lines = [
        "# Pre-API master audit",
        "",
        "Single inventory of **missing data**, **unusable/excluded data**, and **estimated API calls**.",
        "Regenerate: `python scripts/runs/pre_api_master_audit.py`",
        "",
        "## Executive summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Coverage slices (master table) | **{n_complete}/{n_slices}** bank-complete |",
        f"| Missing ID rows (long form) | **{len(missing)}** |",
        f"| Unusable / scoped flags | **{len(unusable)}** |",
        f"| Est. API calls — **core** (GSM+ALGO, Steps 16–18) | **~{core_calls}** |",
        f"| Est. API calls — **BW P2 pilot** (Step 20, optional) | **~{bw_calls}** |",
        f"| Est. API calls — all api_* rows | **~{api_calls}** |",
        f"| **Recovered without API** | GSM P2 o4-mini **44/44** in `GSM_P2_phase1_o1mini.csv` → saves **~616** calls |",
        f"| GSM P1 remap | **Invalid** — GSM_021–040 duplicate GSM_001–020; still need API for GSM_041–064 |",
        "",
        "See **`PRE_API_RECOVERY_AUDIT.md`** for log/file revival checklist.",
        "",
        "## 1. Missing data — needs runs or remap",
        "",
        "```",
        coverage[~coverage["bank_complete"].astype(str).str.lower().eq("true")][
            ["family", "probe", "model", "observed_canonical_n", "bank_canonical_n", "coverage_label"]
        ].to_string(index=False),
        "```",
        "",
        "### Counts by cell",
        "",
        "```",
    ]

    cells = DER / "cells_needing_runs.csv"
    if cells.exists():
        c = pd.read_csv(cells)
        lines.append(c[["family", "probe", "model", "missing_canonical_n", "run_type", "priority"]].to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.extend([
        "## 2. Uncanny / excluded from analysis (exists but don't use blindly)",
        "",
        "| Issue | Where | Action |",
        "|-------|-------|--------|",
        "| **GSM_021–040 in P1 raw** | GPT-4o, Llama | Duplicate reruns of GSM_001–020 — **exclude**; not remap to 041–060 |",
        "| **GSM P2 o4-mini** | `GSM_P2_phase1_o1mini.csv` | **44/44 exists** — merge into derivations (Step 16 not needed) |",
        "| **BW P1 GSM contamination** | `BW_P1_behavioral.csv` | Filter to bank in derivations (done) |",
        "| **BW P2 pilot** | 50/65 problems, 3/5 models | Scope claims; Step 20 protocol fix |",
        "| **BW TEP 87% missing** | 468/536 sessions | Aborted sessions — not blank rows |",
        "| **BW final_ok always False** | P2 TEP slice | Spearman undefined |",
        "| **ALGO P2B n=61** | Injection CSVs | By design — label denominator |",
        "| **P3 mechanistic** | Qwen 0.5B only | Exploratory — not five-model |",
        "| **GSM P2 five-model** | o4-mini in phase1_o1mini file | Five-model P2 OK after merge into loaders |",
        "",
        "Full flag list → `pre_api_unusable_flags.csv`",
        "",
        "## 3. BW P2 (not in master coverage table)",
        "",
        "```",
        pd.DataFrame(bw_p2).drop_duplicates(["probe", "model"])[
            ["probe", "model", "observed_canonical_n", "bank_canonical_n", "coverage_label"]
        ].to_string(index=False),
        "```",
        "",
        "## 4. TEP / P2 analysis usability",
        "",
    ])

    if not tep.empty:
        lines.append("```")
        lines.append(tep.to_string(index=False))
        lines.append("```")
    else:
        lines.append("_Run `tep_dissociation_analysis.py` first for TEP detail._")

    lines.extend([
        "",
        "## 5. API budget (estimated)",
        "",
        "```",
    ])
    if not budget.empty:
        lines.append(
            budget[["priority", "family", "probe", "model", "task", "units", "est_api_calls", "fix_type"]]
            .to_string(index=False)
        )
    lines.append("```")
    lines.extend([
        "",
        "### Priority interpretation",
        "",
        "- **P0:** ~~GSM P2 o4-mini~~ **recovered** from existing file — wire merged loader only",
        "- **P1 core:** ALGO P2A elicited ×3 models (~588 calls); GSM P1 GPT-4o/Llama missing GSM_041–064 (~336 calls)",
        "- **P2 optional:** BW P2 pilot extension (~2300 calls — Step 20; defer until protocol fixed)",
        "",
        "## 6. Output files",
        "",
        "| File | Contents |",
        "|------|----------|",
        "| `pre_api_slice_inventory.csv` | All slices + BW P2 pilot rows |",
        "| `pre_api_missing_ids.csv` | Every missing canonical/session ID |",
        "| `pre_api_unusable_flags.csv` | Exclusion / uncanny flags |",
        "| `pre_api_api_budget.csv` | API estimates by task |",
        "",
    ])
    (DER / "PRE_API_MASTER_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_recovery_markdown(recovery: pd.DataFrame) -> None:
    saved = int(recovery["est_api_saved"].sum()) if not recovery.empty else 0
    lines = [
        "# Pre-API recovery audit",
        "",
        "Before spending on API runs: check **existing raw files** and **sweep logs**.",
        "Regenerate: `python scripts/runs/pre_api_master_audit.py`",
        "",
        f"**Total API calls avoided so far:** ~{saved} (GSM P2 o4-mini recovered)",
        "",
        "## Revival checklist",
        "",
    ]
    if recovery.empty:
        lines.append("_No recovery rows._")
    else:
        lines.append(
            "| Gap | Status | Observed | Source | Log / note | Action | API saved |"
        )
        lines.append("|-----|--------|----------|--------|------------|--------|-----------|")
        for _, r in recovery.iterrows():
            lines.append(
                f"| {r['gap']} | `{r['status']}` | {r['observed']} | `{r['source_file']}` | "
                f"{r['log_ref']} | {r['action']} | {int(r['est_api_saved'])} |"
            )

    lines.extend([
        "",
        "## Log locations",
        "",
        "- `results/raw/new_model_sweep_logs/finish_o4mini.log` — o4-mini GSM P1/P2/BW/ALGO resume (2026-05-24)",
        "- `results/raw/new_model_sweep_logs/wait_and_finish.log` — credit-limit wait loop",
        "- `results/raw/_failed_archive/` — superseded failed CSV snapshots (compare before discard)",
        "",
        "## Status key",
        "",
        "- `recovered_no_api` — data exists; update derivations only",
        "- `remap_invalid` — do not relabel IDs; duplicates or wrong bank mapping",
        "- `needs_api` — no recoverable rows in raw or logs",
        "- `resume_not_full_rerun` — use `--resume` on existing output to retry ERROR rows",
        "",
    ])
    (DER / "PRE_API_RECOVERY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)

    coverage, gaps = master_coverage_table()
    bw_p2 = _bw_p2_inventory()
    unusable = _unusable_flags(coverage)
    missing = _missing_long(gaps)
    budget = _api_budget(coverage, missing)
    tep = _tep_unusable_summary()

    # Combined slice inventory
    inv = coverage.copy()
    inv["in_master_coverage_table"] = True
    if bw_p2:
        inv = pd.concat([inv, pd.DataFrame(bw_p2)], ignore_index=True)

    coverage.to_csv(DER / "master_coverage_table.csv", index=False)
    gaps.to_csv(DER / "master_coverage_gaps.csv", index=False)
    inv.to_csv(DER / "pre_api_slice_inventory.csv", index=False)
    missing.to_csv(DER / "pre_api_missing_ids.csv", index=False)
    unusable.to_csv(DER / "pre_api_unusable_flags.csv", index=False)
    budget.to_csv(DER / "pre_api_api_budget.csv", index=False)
    write_markdown(coverage, bw_p2, unusable, budget, missing, tep)

    recovery = _recovery_inventory()
    write_recovery_markdown(recovery)
    recovery.to_csv(DER / "pre_api_recovery_inventory.csv", index=False)

    print("Wrote pre-API master audit:")
    for name in [
        "PRE_API_MASTER_AUDIT.md",
        "PRE_API_RECOVERY_AUDIT.md",
        "pre_api_recovery_inventory.csv",
        "pre_api_slice_inventory.csv",
        "pre_api_missing_ids.csv",
        "pre_api_unusable_flags.csv",
        "pre_api_api_budget.csv",
    ]:
        print(f"  results/derived/{name}")

    api = budget[budget["fix_type"].str.startswith("api", na=False)] if not budget.empty else pd.DataFrame()
    core = budget[
        budget["family"].isin(["GSM", "ALGO"])
        & budget["fix_type"].str.startswith("api", na=False)
    ] if not budget.empty else pd.DataFrame()
    bw = budget[
        (budget["family"] == "BW") & budget["fix_type"].str.startswith("api", na=False)
    ] if not budget.empty else pd.DataFrame()
    print(f"\nSlices complete: {int(coverage['bank_complete'].astype(str).str.lower().eq('true').sum())}/{len(coverage)}")
    print(f"Missing ID rows: {len(missing)}")
    print(f"Unusable flags: {len(unusable)}")
    print(f"Est. API calls (core GSM+ALGO api_*): ~{int(core['est_api_calls'].sum()) if not core.empty else 0}")
    print(f"Est. API calls (BW P2 pilot extension): ~{int(bw['est_api_calls'].sum()) if not bw.empty else 0}")
    print(f"Est. API calls (all api_*): ~{int(api['est_api_calls'].sum()) if not api.empty else 0}")


if __name__ == "__main__":
    main()
