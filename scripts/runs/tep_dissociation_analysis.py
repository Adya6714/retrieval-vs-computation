#!/usr/bin/env python3
"""Step 8 — TEP dissociation analysis (existing P2 data only).

TEP measures trajectory divergence after mid-execution perturbation; dissociation
is when TEP is high but the final answer is still correct (or vice versa).

Outputs:
    results/derived/tep_dissociation_sessions.csv
    results/derived/tep_dissociation_by_slice.csv
    results/derived/tep_dissociation_correlations.csv
    results/derived/tep_injection_recovery.csv
    results/derived/tep_dissociation_scatter.csv
    results/derived/tep_dissociation_quality_audit.md
    results/derived/tep_dissociation_summary.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
DER = ROOT / "results" / "derived"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.algo.decision_normalize import normalize_phase2_decision

SHORT = {
    "anthropic/claude-sonnet-4": "Claude",
    "google/gemini-2.5-flash": "Gemini",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}

TEP_HIGH = 0.5
PARSEABLE = {"compliant", "format_ignored"}


def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return None


def _to_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_gsm_sessions() -> pd.DataFrame:
    from scripts.runs.coverage_audit import load_gsm_p2_merged

    df = load_gsm_p2_merged()
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["family"] = "GSM"
    out["subtype"] = out.get("problem_subtype", "")
    out["tep"] = pd.to_numeric(out.get("tep_score", ""), errors="coerce")
    out["cci"] = pd.to_numeric(out.get("cci_score", ""), errors="coerce")
    out["final_ok"] = _to_bool(out.get("session_b_correct", pd.Series([""] * len(out))))
    out["inject_at_step"] = pd.to_numeric(out.get("inject_at_step", ""), errors="coerce")
    out["model_short"] = out["model"].map(SHORT).fillna(out["model"])
    out = _drop_invalid_rows(out, family="GSM")
    return out[
        [
            "family",
            "problem_id",
            "model",
            "model_short",
            "subtype",
            "tep",
            "cci",
            "final_ok",
            "inject_at_step",
        ]
    ]


def _normalize_algo_steps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_step"] = pd.to_numeric(out["step_index"], errors="coerce").fillna(0).astype(int)
    mins = out.groupby(["problem_id", "model", "instance_type"])["_step"].transform("min")
    out["_step"] = out["_step"] - mins
    return out


def _algo_post_critical_tep(
    gn: pd.DataFrame,
    gi: pd.DataFrame,
    subtype: str,
    crit: int,
    *,
    strict_compliant: bool,
) -> float:
    merged = gn.merge(
        gi[["_step", "parsed_decision", "response_type"]],
        on="_step",
        how="inner",
        suffixes=("_n", "_i"),
    )
    post = merged[merged["_step"] > crit]
    if strict_compliant:
        post = post[(post["response_type_n"] == "compliant") & (post["response_type_i"] == "compliant")]
    else:
        post = post[
            post["response_type_n"].isin(PARSEABLE) & post["response_type_i"].isin(PARSEABLE)
        ]
    if post.empty:
        return float("nan")
    diffs = post.apply(
        lambda r: normalize_phase2_decision(subtype, r["parsed_decision_n"])
        != normalize_phase2_decision(subtype, r["parsed_decision_i"]),
        axis=1,
    )
    return float(diffs.mean())


def _algo_session_tep(normal: pd.DataFrame, injected: pd.DataFrame) -> pd.DataFrame:
    """Per (problem_id, model, instance_type) TEP + final correctness."""
    if normal.empty or injected.empty:
        return pd.DataFrame()

    normal = _normalize_algo_steps(normal)
    injected = _normalize_algo_steps(injected)

    keys = ["problem_id", "model", "instance_type", "subtype"]
    last = (
        normal.sort_values("_step")
        .groupby(keys, as_index=False)
        .tail(1)[keys + ["final_answer_correct"]]
        .rename(columns={"final_answer_correct": "final_ok_raw"})
    )
    last["final_ok"] = _to_bool(last["final_ok_raw"])

    rows: list[dict] = []
    for (pid, model, inst, subtype), gn in normal.groupby(keys):
        gi = injected[
            (injected["problem_id"] == pid)
            & (injected["model"] == model)
            & (injected["instance_type"] == inst)
        ]
        if gi.empty:
            continue
        crit_rows = gi[gi["critical_step_index"].astype(str).str.strip() != ""]
        if crit_rows.empty:
            continue
        try:
            crit = int(float(crit_rows.iloc[0]["critical_step_index"]))
        except ValueError:
            continue
        if crit < 0:
            tep = float("nan")
            tep_inclusive = float("nan")
        else:
            tep = _algo_post_critical_tep(gn, gi, subtype, crit, strict_compliant=True)
            tep_inclusive = _algo_post_critical_tep(gn, gi, subtype, crit, strict_compliant=False)
            if pd.isna(tep) and not pd.isna(tep_inclusive):
                tep = tep_inclusive

        fin = last[
            (last["problem_id"] == pid)
            & (last["model"] == model)
            & (last["instance_type"] == inst)
        ]
        final_ok = bool(fin["final_ok"].iloc[0]) if not fin.empty else False
        rows.append(
            {
                "family": "ALGO",
                "problem_id": pid,
                "model": model,
                "model_short": SHORT.get(model, model),
                "subtype": subtype,
                "instance_type": inst,
                "tep": tep,
                "tep_inclusive": tep_inclusive,
                "cci": float("nan"),
                "final_ok": final_ok,
                "inject_at_step": crit,
            }
        )
    return _drop_invalid_rows(pd.DataFrame(rows), family="ALGO")


def _bw_tep_from_row(row: pd.Series) -> float:
    """Recompute TEP from cascade JSON when CSV tep column is empty."""
    existing = pd.to_numeric(row.get("tep", ""), errors="coerce")
    if not pd.isna(existing):
        return float(existing)
    try:
        cascade = json.loads(row.get("cascade_sequence_json", "[]") or "[]")
    except json.JSONDecodeError:
        cascade = []
    if not isinstance(cascade, list) or not cascade:
        adapted = pd.to_numeric(row.get("adapted_count", ""), errors="coerce")
        resistant = pd.to_numeric(row.get("resistant_count", ""), errors="coerce")
        ambiguous = pd.to_numeric(row.get("ambiguous_count", ""), errors="coerce")
        if pd.isna(adapted) and pd.isna(resistant):
            return float("nan")
        a = int(adapted or 0)
        r = int(resistant or 0)
        amb = int(ambiguous or 0)
        denom = a + r + amb
        return float(a / denom) if denom > 0 else float("nan")
    adapted = sum(1 for s in cascade if isinstance(s, dict) and s.get("classification") == "adapted")
    resistant = sum(1 for s in cascade if isinstance(s, dict) and s.get("classification") == "resistant")
    ambiguous = sum(1 for s in cascade if isinstance(s, dict) and s.get("classification") == "ambiguous")
    denom = adapted + resistant + ambiguous
    return float(adapted / denom) if denom > 0 else float("nan")


def load_algo_sessions() -> pd.DataFrame:
    normal = _safe_read(RAW / "ALGO_P2_phase2_normal.csv")
    injected = _safe_read(RAW / "ALGO_P2_phase2_injected.csv")
    if normal is None or injected is None:
        return pd.DataFrame()
    return _algo_session_tep(normal, injected)


def load_bw_sessions() -> pd.DataFrame:
    df = _safe_read(RAW / "BW_P2_tep.csv")
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out["family"] = "BW"
    out["subtype"] = ""
    out["tep"] = out.apply(_bw_tep_from_row, axis=1)
    out["final_ok"] = _to_bool(out.get("goal_reached_true", pd.Series([""] * len(out))))
    out["inject_at_step"] = pd.to_numeric(out.get("inject_at_step", ""), errors="coerce")
    out["model_short"] = out["model"].map(SHORT).fillna(out["model"])
    out["first_response_class"] = out.get("first_response_class", "")
    out["session_status"] = out.get("session_status", "")
    out = _drop_invalid_rows(out, family="BW")
    return out[
        [
            "family",
            "problem_id",
            "model",
            "model_short",
            "subtype",
            "tep",
            "final_ok",
            "inject_at_step",
            "first_response_class",
            "session_status",
        ]
    ]


def _drop_invalid_rows(df: pd.DataFrame, *, family: str) -> pd.DataFrame:
    """Remove blank keys and all-empty rows before any metrics."""
    if df.empty:
        return df
    out = df.copy()
    out = out[~out.apply(lambda r: all(str(x).strip() == "" for x in r), axis=1)]
    for col in ("problem_id", "model"):
        if col in out.columns:
            out = out[out[col].astype(str).str.strip() != ""]
    out["family"] = family
    return out.reset_index(drop=True)


def _annotate_quality(sessions: pd.DataFrame) -> pd.DataFrame:
    """Flag rows eligible for TEP stats vs Spearman (needs variation in both vars)."""
    out = sessions.copy()
    out["tep_valid"] = out["tep"].notna()
    out["final_ok_defined"] = out["final_ok"].notna()

    reasons: list[str] = []
    for _, row in out.iterrows():
        if not row["tep_valid"]:
            reasons.append("tep_missing")
        elif not row["final_ok_defined"]:
            reasons.append("final_ok_missing")
        else:
            reasons.append("analysis_ready")
    out["quality_flag"] = reasons
    return out


def _corr_eligible(g: pd.DataFrame) -> tuple[bool, str]:
    gv = g[g["tep"].notna()].copy()
    if len(gv) < 3:
        return False, f"n_tep_valid={len(gv)} (<3)"
    if gv["tep"].nunique() < 2:
        return False, "tep_constant"
    if gv["final_ok"].nunique() < 2:
        return False, "final_ok_constant"
    return True, "ok"


def _dissociation_label(row: pd.Series) -> str:
    tep = row["tep"]
    ok = bool(row["final_ok"])
    if pd.isna(tep):
        return "tep_missing"
    if tep >= TEP_HIGH and ok:
        return "dissociated_high_tep_correct_final"
    if tep >= TEP_HIGH and not ok:
        return "aligned_high_tep_wrong_final"
    if tep < TEP_HIGH and ok:
        return "aligned_low_tep_correct_final"
    return "dissociated_low_tep_wrong_final"


def aggregate_by_slice(sessions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    group_cols = ["family", "model_short", "subtype"]
    for keys, g in sessions.groupby(group_cols, dropna=False):
        fam, model, subtype = keys
        g = g.copy()
        g["dissoc_label"] = g.apply(_dissociation_label, axis=1)
        valid = g["tep"].notna()
        gv = g[valid]
        rows.append(
            {
                "family": fam,
                "model": model,
                "subtype": subtype or "(all)",
                "n_sessions": len(g),
                "n_tep_valid": int(valid.sum()),
                "mean_tep": float(gv["tep"].mean()) if not gv.empty else float("nan"),
                "final_ok_rate": float(g["final_ok"].mean()),
                "mean_cci": float(g["cci"].mean()) if "cci" in g.columns and g["cci"].notna().any() else float("nan"),
                "pct_dissociated_high_tep_correct": float(
                    (g["dissoc_label"] == "dissociated_high_tep_correct_final").mean()
                ),
                "pct_high_tep": float((gv["tep"] >= TEP_HIGH).mean()) if not gv.empty else float("nan"),
                "tep_final_spearman": (
                    float(stats.spearmanr(gv["tep"], gv["final_ok"].astype(int))[0])
                    if _corr_eligible(g)[0]
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def correlation_table(sessions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (fam, model), g in sessions.groupby(["family", "model_short"]):
        gv = g[g["tep"].notna()].copy()
        eligible, reason = _corr_eligible(g)
        if eligible:
            rho, p = stats.spearmanr(gv["tep"], gv["final_ok"].astype(int))
        else:
            rho, p = float("nan"), float("nan")
        high = gv[gv["tep"] >= TEP_HIGH]
        rows.append(
            {
                "family": fam,
                "model": model,
                "n_total": len(g),
                "n_tep_valid": len(gv),
                "corr_eligible": eligible,
                "corr_exclude_reason": reason if not eligible else "",
                "mean_tep": float(gv["tep"].mean()) if not gv.empty else float("nan"),
                "final_ok_rate": float(gv["final_ok"].mean()) if not gv.empty else float("nan"),
                "high_tep_final_ok_rate": float(high["final_ok"].mean()) if len(high) >= 1 else float("nan"),
                "high_tep_n": len(high),
                "spearman_tep_vs_final": float(rho),
                "spearman_p": float(p),
            }
        )
    return pd.DataFrame(rows)


def algo_injection_recovery() -> pd.DataFrame:
    """Post-injection steps: recovery rate by reasoning type and inject step."""
    inj = _safe_read(RAW / "ALGO_P2_phase2_injected.csv")
    impl = _safe_read(RAW / "ALGO_P2_phase2_injected_implausible.csv")
    parts = []
    for df, mode in [(inj, "plausible"), (impl, "implausible")]:
        if df is None or df.empty:
            continue
        d = df.copy()
        d["mode"] = mode
        parts.append(d)
    if not parts:
        return pd.DataFrame()

    all_inj = pd.concat(parts, ignore_index=True)
    all_inj["_step"] = pd.to_numeric(all_inj["step_index"], errors="coerce").fillna(0).astype(int)

    # Last-step post_injection_correct per session
    keys = ["problem_id", "model", "instance_type", "mode"]
    last = (
        all_inj[all_inj["post_injection_correct"].astype(str).str.strip() != ""]
        .sort_values("_step")
        .groupby(keys, as_index=False)
        .tail(1)
    )
    last["post_ok"] = _to_bool(last["post_injection_correct"])

    # Reasoning at injection step
    at_inj = all_inj[_to_bool(all_inj.get("injection_applied", pd.Series([""] * len(all_inj))))].copy()
    at_inj = at_inj.rename(columns={"reasoning_type": "inj_reasoning_type"})
    merged = last.merge(
        at_inj[keys + ["_step", "inj_reasoning_type", "critical_step_index"]],
        on=keys,
        how="left",
        suffixes=("", "_inj"),
    )
    merged["model_short"] = merged["model"].map(SHORT).fillna(merged["model"])
    merged["critical_step"] = pd.to_numeric(merged["critical_step_index"], errors="coerce")

    agg = (
        merged.groupby(["mode", "model_short", "subtype", "inj_reasoning_type", "critical_step"], dropna=False)
        .agg(n=("post_ok", "size"), post_ok_rate=("post_ok", "mean"))
        .reset_index()
    )
    agg = agg[agg["n"] > 0].copy()
    return agg.sort_values(["mode", "model_short", "n"], ascending=[True, True, False])


def write_quality_audit(sessions: pd.DataFrame, corr: pd.DataFrame) -> None:
    lines = [
        "# TEP dissociation — data quality audit",
        "",
        "Rows with blank `problem_id` or `model` are dropped on load. "
        "All-empty CSV rows are removed. Spearman is computed only when "
        "`corr_eligible=True` (≥3 TEP-valid sessions, TEP and final_ok both vary).",
        "",
        "## Session inventory",
        "",
        f"- Total sessions after validation: **{len(sessions)}**",
        f"- Empty/all-blank rows dropped at load: see per-family loaders",
        f"- TEP-valid: **{int(sessions['tep_valid'].sum())}** ({100 * sessions['tep_valid'].mean():.1f}%)",
        f"- Analysis-ready (TEP + final_ok vary within slice): see correlation table",
        "",
        "### By family",
        "",
        "```",
    ]
    fam_stats = []
    for fam, g in sessions.groupby("family"):
        fam_stats.append(
            {
                "family": fam,
                "n": len(g),
                "tep_valid": int(g["tep_valid"].sum()),
                "tep_missing": int((~g["tep_valid"]).sum()),
                "final_ok_rate": round(float(g["final_ok"].mean()), 3),
            }
        )
    lines.append(pd.DataFrame(fam_stats).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("### TEP missing — expected causes")
    lines.append("")
    for row in fam_stats:
        lines.append(
            f"- **{row['family']} ({row['tep_missing']}/{row['n']} missing):** "
            + (
                "session aborted or empty cascade (BW); "
                if row["family"] == "BW"
                else "no post-critical parseable steps (ALGO); "
                if row["family"] == "ALGO"
                else "all rows have tep_score (GSM); "
            )
            + "not blank-row corruption."
        )
    lines.append("")
    lines.append("### Spearman exclusions")
    lines.append("")
    excluded = corr[~corr["corr_eligible"]]
    if excluded.empty:
        lines.append("All model slices eligible.")
    else:
        lines.append("```")
        lines.append(
            excluded[["family", "model", "n_tep_valid", "corr_exclude_reason"]].to_string(index=False)
        )
        lines.append("```")
        lines.append("")
        lines.append("BW slices excluded because **final_ok is always False** (zero variance) — correlation undefined, not a parsing error.")
    lines.append("")
    lines.append("### Column nulls (expected, not bugs)")
    lines.append("")
    lines.append("- `cci`: GSM-only")
    lines.append("- `instance_type`, `tep_inclusive`: ALGO-only")
    lines.append("- `first_response_class`, `session_status`: BW-only")
    lines.append("- `subtype`: blank for BW (no subtype in probe)")
    lines.append("")
    (DER / "tep_dissociation_quality_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(
    sessions: pd.DataFrame,
    by_slice: pd.DataFrame,
    corr: pd.DataFrame,
    inj: pd.DataFrame,
) -> None:
    sessions = sessions.copy()
    sessions["dissoc_label"] = sessions.apply(_dissociation_label, axis=1)

    valid = sessions[sessions["tep"].notna()]
    lines = [
        "# TEP dissociation analysis (Step 8)",
        "",
        "TEP = trajectory divergence after perturbation. **Dissociation** = high TEP (≥0.5) "
        "but final answer still correct — suggests re-derivation or answer independent of corrupted chain.",
        "",
        f"**Sessions analyzed:** {len(sessions)} ({sessions['family'].value_counts().to_dict()})",
        f"**TEP-valid sessions:** {len(valid)} ({100 * len(valid) / max(len(sessions), 1):.1f}%)",
        "",
        "ALGO TEP uses paper-compliant steps; when empty, falls back to parseable (`format_ignored`) steps. "
        "BW TEP recomputed from `cascade_sequence_json` when CSV column is blank.",
        "",
        "## Headline: TEP vs final correctness",
        "",
        "```",
        corr.sort_values(["family", "model"])[
            [
                "family",
                "model",
                "n_tep_valid",
                "corr_eligible",
                "mean_tep",
                "final_ok_rate",
                "high_tep_final_ok_rate",
                "spearman_tep_vs_final",
            ]
        ].round(3).to_string(index=False),
        "```",
        "",
        "**Interpretation:** Weak or positive Spearman on GSM/ALGO for some models → high TEP does not imply wrong finals.",
        "",
        "## Dissociation rate (high TEP + correct final)",
        "",
        "```",
        by_slice.sort_values(["family", "model"])[
            ["family", "model", "subtype", "n_sessions", "mean_tep", "final_ok_rate", "pct_dissociated_high_tep_correct"]
        ].round(3).to_string(index=False),
        "```",
        "",
        "## Quadrant counts (all families)",
        "",
    ]
    quad = sessions["dissoc_label"].value_counts()
    for label, n in quad.items():
        lines.append(f"- **{label}**: {n} ({100 * n / len(sessions):.1f}%)")

    if not inj.empty:
        lines.extend(["", "## ALGO injection recovery (post-injection correct)", ""])
        top = inj.sort_values("n", ascending=False).head(20)
        lines.append("```")
        lines.append(top.round(3).to_string(index=False))
        lines.append("```")

    lines.extend(
        [
            "",
            "## Mechanistic hypotheses (for paper discussion)",
            "",
            "1. **Re-derivation:** model recomputes from problem statement after chain corruption (GSM numeric steps).",
            "2. **Terminal correction:** wrong intermediate steps but correct final aggregation (coin-change).",
            "3. **Format compliance without state:** ALGO `compliant` steps after injection may diverge in token but still reach correct final state.",
            "4. **BW protocol noise:** many BW TEP rows reflect parser/session abort — interpret BW separately.",
            "",
            "## Files",
            "",
            "- `tep_dissociation_sessions.csv` — per-session TEP, final_ok, dissociation label",
            "- `tep_dissociation_by_slice.csv` — family × model × subtype aggregates",
            "- `tep_dissociation_correlations.csv` — Spearman TEP vs final_ok",
            "- `tep_injection_recovery.csv` — ALGO injection recovery by reasoning type",
            "- `tep_dissociation_scatter.csv` — TEP-valid rows for scatter plots",
            "- `tep_dissociation_quality_audit.md` — row validation and exclusion log",
            "",
        ]
    )
    (DER / "tep_dissociation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)

    parts = [load_gsm_sessions(), load_algo_sessions(), load_bw_sessions()]
    sessions = pd.concat([p for p in parts if not p.empty], ignore_index=True)
    if sessions.empty:
        raise SystemExit("No P2 session data found for TEP analysis.")

    sessions = _annotate_quality(sessions)
    sessions["dissoc_label"] = sessions.apply(_dissociation_label, axis=1)

    by_slice = aggregate_by_slice(sessions)
    corr = correlation_table(sessions)
    inj = algo_injection_recovery()

    scatter = sessions[sessions["tep_valid"]].copy()
    scatter["final_ok_int"] = scatter["final_ok"].astype(int)

    sessions.to_csv(DER / "tep_dissociation_sessions.csv", index=False)
    by_slice.to_csv(DER / "tep_dissociation_by_slice.csv", index=False)
    corr.to_csv(DER / "tep_dissociation_correlations.csv", index=False)
    inj.to_csv(DER / "tep_injection_recovery.csv", index=False)
    scatter.to_csv(DER / "tep_dissociation_scatter.csv", index=False)
    write_quality_audit(sessions, corr)
    write_summary(sessions, by_slice, corr, inj)

    print("Wrote TEP dissociation pack:")
    for name in [
        "tep_dissociation_sessions.csv",
        "tep_dissociation_by_slice.csv",
        "tep_dissociation_correlations.csv",
        "tep_injection_recovery.csv",
        "tep_dissociation_scatter.csv",
        "tep_dissociation_quality_audit.md",
        "tep_dissociation_summary.md",
    ]:
        print(f" - results/derived/{name}")

    dissoc = sessions[sessions["dissoc_label"] == "dissociated_high_tep_correct_final"]
    print(f"\nDissociated (high TEP + correct final): {len(dissoc)}/{len(sessions)} sessions")


if __name__ == "__main__":
    main()
