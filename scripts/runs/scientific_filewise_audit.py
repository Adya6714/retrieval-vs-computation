#!/usr/bin/env python3
"""Scientific file-wise audit across probe runs.

Goal:
- Analyze each run/metric file with small, concrete deductions.
- Compute cross-relations across probes/families/models.
- Produce reviewable markdown + machine-readable tables.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw"
DER = ROOT / "results" / "derived"

FAMILY_PREFIXES = ("ALGO_", "GSM_", "BW_")
CORRECT_COL_CANDIDATES = [
    "verified",
    "behavioral_correct",
    "final_answer_correct",
    "post_injection_correct",
    "session_b_correct",
]


def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return None


def _to_bool(x: Any) -> bool | None:
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _norm_variant(v: Any) -> str:
    s = str(v).strip()
    if not s:
        return s
    if s.lower() == "canonical":
        return "canonical"
    if s.lower().startswith("w") and s[1:].isdigit():
        return f"W{s[1:]}"
    return s


def _write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def list_targets() -> tuple[list[Path], list[Path]]:
    raw_files = sorted(
        [p for p in RAW.glob("*.csv") if p.name.startswith(FAMILY_PREFIXES)]
    )
    derived_files = sorted(
        [
            p
            for p in DER.glob("*.csv")
            if p.name.startswith(
                (
                    "ALGO_",
                    "GSM_",
                    "BW_",
                    "cross_",
                    "probe",
                    "P1_",
                    "o4mini_",
                    "gemini_",
                    "implausibility",
                    "algorithm_invocation",
                    "bw_violation",
                    "master_",
                    "coverage_",
                    "table_",
                    "deep_",
                )
            )
        ]
    )
    return raw_files, derived_files


def generic_profile(path: Path, df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "file": path.name,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "models": "",
        "n_models": 0,
        "n_problem_ids": 0,
        "n_subtypes": 0,
        "parse_failed_rate": float("nan"),
        "top_response_type": "",
        "top_variant": "",
        "primary_correct_col": "",
        "primary_correct_rate": float("nan"),
    }
    if "model" in df.columns:
        mods = sorted(df["model"].astype(str).replace("", np.nan).dropna().unique())
        out["models"] = ",".join(mods[:8]) + ("..." if len(mods) > 8 else "")
        out["n_models"] = len(mods)
    if "problem_id" in df.columns:
        out["n_problem_ids"] = int(df["problem_id"].nunique())
    if "subtype" in df.columns:
        out["n_subtypes"] = int(df["subtype"].nunique())
    elif "problem_subtype" in df.columns:
        out["n_subtypes"] = int(df["problem_subtype"].nunique())
    if "parse_status" in df.columns:
        fail = df["parse_status"].astype(str).str.contains("fail", case=False, na=False)
        out["parse_failed_rate"] = float(fail.mean()) if len(df) else float("nan")
    if "response_type" in df.columns and len(df):
        vc = df["response_type"].astype(str).value_counts()
        out["top_response_type"] = str(vc.index[0]) if len(vc) else ""
    if "variant_type" in df.columns and len(df):
        vc = df["variant_type"].map(_norm_variant).value_counts()
        out["top_variant"] = str(vc.index[0]) if len(vc) else ""

    for col in CORRECT_COL_CANDIDATES:
        if col in df.columns:
            vals = df[col].map(_to_bool)
            if vals.notna().sum() > 0:
                out["primary_correct_col"] = col
                out["primary_correct_rate"] = float(vals.mean())
                break
    return out


def p1_deductions(path: Path, df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if "variant_type" not in df.columns or "problem_id" not in df.columns:
        return lines

    # Determine correctness column
    corr_col = None
    for c in ["verified", "behavioral_correct"]:
        if c in df.columns:
            corr_col = c
            break
    if corr_col is None:
        return lines

    d = df.copy()
    d["variant_type"] = d["variant_type"].map(_norm_variant)
    d["_ok"] = d[corr_col].map(_to_bool)
    if "raw_response" in d.columns:
        d.loc[d["raw_response"].astype(str).str.startswith("ERROR:"), "_ok"] = False
    if "model" not in d.columns:
        d["model"] = "unknown"
    d = d.drop_duplicates(["problem_id", "model", "variant_type"], keep="last")

    # Variant-level rates
    vr = (
        d.groupby(["model", "variant_type"], as_index=False)["_ok"]
        .mean()
        .rename(columns={"_ok": "acc"})
    )
    # Canonical-conditioned variant keep
    kept_rows = []
    for model, g in d.groupby("model"):
        wide = g.pivot_table(index="problem_id", columns="variant_type", values="_ok", aggfunc="last")
        if "canonical" not in wide.columns:
            continue
        can = wide["canonical"]
        for v in ["W1", "W2", "W3", "W4", "W5", "W6"]:
            if v not in wide.columns:
                continue
            sub = wide[[v, "canonical"]].dropna()
            if sub.empty:
                continue
            cc = sub["canonical"].astype(bool)
            vv = sub[v].astype(bool)
            keep = (cc & vv).sum()
            drop = (cc & ~vv).sum()
            rescue = (~cc & vv).sum()
            kept_rows.append(
                {
                    "model": model,
                    "variant": v,
                    "drop_rate_given_canonical_correct": (drop / cc.sum()) if cc.sum() > 0 else np.nan,
                    "rescue_rate_given_canonical_wrong": (rescue / (~cc).sum()) if (~cc).sum() > 0 else np.nan,
                    "n_overlap": len(sub),
                }
            )
    kept = pd.DataFrame(kept_rows)
    if not kept.empty:
        worst = kept.sort_values("drop_rate_given_canonical_correct", ascending=False).iloc[0]
        best_rescue = kept.sort_values("rescue_rate_given_canonical_wrong", ascending=False).iloc[0]
        lines.append(
            f"- Worst canonical->variant fragility: `{worst['model']}` on `{worst['variant']}` with "
            f"drop-rate {worst['drop_rate_given_canonical_correct']:.3f} (n={int(worst['n_overlap'])})."
        )
        if np.isfinite(best_rescue["rescue_rate_given_canonical_wrong"]):
            lines.append(
                f"- Highest rescue effect: `{best_rescue['model']}` on `{best_rescue['variant']}` with "
                f"rescue-rate {best_rescue['rescue_rate_given_canonical_wrong']:.3f}."
            )

    # Global W3 remark when present
    if not vr.empty:
        piv = vr.pivot_table(index="model", columns="variant_type", values="acc", aggfunc="first")
        if "canonical" in piv.columns and "W3" in piv.columns:
            diff = (piv["canonical"] - piv["W3"]).dropna()
            if len(diff):
                lines.append(
                    f"- Mean canonical minus W3 gap across models: {float(diff.mean()):.3f} "
                    f"(max={float(diff.max()):.3f}, min={float(diff.min()):.3f})."
                )
    return lines


def p2_phase1_deductions(path: Path, df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for col in ["phase1_parseable", "greedy_assessment_correct", "critical_point_identified"]:
        if col in df.columns:
            vals = df[col].map(_to_bool)
            if vals.notna().sum():
                lines.append(f"- `{col}` rate: {float(vals.mean()):.3f} over {int(vals.notna().sum())} rows.")
    if "predicted_first_decision" in df.columns:
        non_empty = (df["predicted_first_decision"].astype(str).str.strip() != "").mean()
        lines.append(f"- `predicted_first_decision` non-empty rate: {float(non_empty):.3f}.")
    return lines


def p2_phase2_deductions(path: Path, df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if "response_type" in df.columns and len(df):
        vc = df["response_type"].astype(str).value_counts(normalize=True).head(4)
        top = ", ".join([f"{k}:{v:.2f}" for k, v in vc.items()])
        lines.append(f"- Top `response_type` distribution: {top}.")
    if "post_injection_correct" in df.columns:
        vals = df["post_injection_correct"].map(_to_bool)
        if vals.notna().sum():
            lines.append(f"- `post_injection_correct` rate: {float(vals.mean()):.3f} (final-step rows only).")
    if "final_answer_correct" in df.columns:
        vals = df["final_answer_correct"].map(_to_bool)
        if vals.notna().sum():
            lines.append(f"- `final_answer_correct` rate: {float(vals.mean()):.3f}.")
    if "injection_applied" in df.columns:
        vals = df["injection_applied"].map(_to_bool)
        if vals.notna().sum():
            lines.append(f"- `injection_applied=True` prevalence: {float(vals.mean()):.3f}.")
    return lines


def contamination_deductions(path: Path, df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    for c in ["instance_contamination_score", "contamination_score", "template_contamination_score"]:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum():
                q = x.quantile([0.1, 0.5, 0.9]).to_dict()
                lines.append(
                    f"- `{c}` quantiles q10/q50/q90: {q[0.1]:.3f}/{q[0.5]:.3f}/{q[0.9]:.3f}."
                )
    return lines


def mechanistic_deductions(path: Path, df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    if "crystallization_layer" in df.columns and "n_layers_processed" in df.columns:
        c = pd.to_numeric(df["crystallization_layer"], errors="coerce")
        n = pd.to_numeric(df["n_layers_processed"], errors="coerce")
        depth = c / n.replace(0, np.nan)
        if depth.notna().sum():
            lines.append(
                f"- Normalized crystallization depth mean/std: {float(depth.mean()):.3f}/{float(depth.std()):.3f}."
            )
    if "layer_cosine_similarities" in df.columns:
        non_empty = (df["layer_cosine_similarities"].astype(str).str.strip() != "").mean()
        lines.append(f"- `layer_cosine_similarities` non-empty rate: {float(non_empty):.3f}.")
    return lines


def derived_metric_deductions(path: Path, df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    metric_col = "metric_name" if "metric_name" in df.columns else ("metric" if "metric" in df.columns else "")
    if metric_col:
        vals = sorted(df[metric_col].astype(str).str.strip().replace("", np.nan).dropna().unique().tolist())
        if vals:
            lines.append(f"- Metrics present ({len(vals)}): " + ", ".join(vals[:12]) + (" ..." if len(vals) > 12 else "") + ".")
    for c in ["metric_value", "value", "accuracy", "mean_cci", "mean_tep"]:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            if x.notna().sum():
                lines.append(f"- `{c}` mean/min/max: {float(x.mean()):.3f}/{float(x.min()):.3f}/{float(x.max()):.3f}.")
    if "convergence_label" in df.columns:
        vc = df["convergence_label"].astype(str).value_counts(normalize=True).head(6)
        lines.append("- `convergence_label` distribution: " + ", ".join([f"{k}:{v:.2f}" for k, v in vc.items()]) + ".")
    if "tri_plus_label" in df.columns:
        vc = df["tri_plus_label"].astype(str).value_counts(normalize=True).head(6)
        lines.append("- `tri_plus_label` distribution: " + ", ".join([f"{k}:{v:.2f}" for k, v in vc.items()]) + ".")
    return lines


def file_deductions(path: Path, df: pd.DataFrame) -> list[str]:
    name = path.name
    lines: list[str] = []
    if "_P1_behavioral" in name:
        lines.extend(p1_deductions(path, df))
    elif "_P2_phase1_" in name:
        lines.extend(p2_phase1_deductions(path, df))
    elif "_P2_phase2_" in name or "_P2_cci" in name or "_P2_tep" in name:
        lines.extend(p2_phase2_deductions(path, df))
    elif "_P3_contamination" in name:
        lines.extend(contamination_deductions(path, df))
    elif "_P3_mechanistic" in name:
        lines.extend(mechanistic_deductions(path, df))
    else:
        lines.extend(derived_metric_deductions(path, df))
    return lines


def cross_relations() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    # Cross 1: P1 rename fragility vs contamination
    deep_tri = _safe_read(DER / "deep_triangulation_plus.csv")
    if deep_tri is not None and not deep_tri.empty:
        d = deep_tri.copy()
        for fam, g in d.groupby("family"):
            if "contam" in g.columns and "tri_plus_score" in g.columns:
                x = pd.to_numeric(g["contam"], errors="coerce")
                y = pd.to_numeric(g["tri_plus_score"], errors="coerce")
                m = x.notna() & y.notna()
                if m.sum() >= 5 and np.nanstd(x[m]) > 0 and np.nanstd(y[m]) > 0:
                    r = float(np.corrcoef(x[m], y[m])[0, 1])
                else:
                    r = float("nan")
                rows.append(
                    {
                        "relation": "contamination_vs_tri_plus_score",
                        "family": fam,
                        "model": "all",
                        "n": int(m.sum()),
                        "value": r,
                        "interpretation": "negative => higher contamination associates with retrieval-side votes",
                    }
                )

    # Cross 2: P2A phase link effectiveness from deep file
    p2a = _safe_read(DER / "deep_p2a_phase_link.csv")
    if p2a is not None and not p2a.empty:
        for _, r in p2a.iterrows():
            model = str(r.get("model", ""))
            subtype = str(r.get("subtype", ""))
            match = pd.to_numeric(pd.Series([r.get("first_decision_match_rate", np.nan)]), errors="coerce").iloc[0]
            final = pd.to_numeric(pd.Series([r.get("final_correct_rate", np.nan)]), errors="coerce").iloc[0]
            rows.append(
                {
                    "relation": "p2a_first_decision_match_rate",
                    "family": "ALGO",
                    "model": model,
                    "n": int(pd.to_numeric(pd.Series([r.get("n_sessions", 0)]), errors="coerce").fillna(0).iloc[0]),
                    "value": float(match) if np.isfinite(match) else float("nan"),
                    "interpretation": f"subtype={subtype}; final_correct_rate={final:.3f}" if np.isfinite(final) else f"subtype={subtype}",
                }
            )

    # Cross 3: injection plausible-implausible delta
    p2b = _safe_read(DER / "deep_p2b_reactivity_delta.csv")
    if p2b is not None and not p2b.empty:
        for _, r in p2b.iterrows():
            delta = pd.to_numeric(pd.Series([r.get("plausible_minus_implausible", np.nan)]), errors="coerce").iloc[0]
            rows.append(
                {
                    "relation": "p2b_plausible_minus_implausible",
                    "family": "ALGO",
                    "model": str(r.get("model", "")),
                    "n": 1,
                    "value": float(delta) if np.isfinite(delta) else float("nan"),
                    "interpretation": f"subtype={r.get('subtype','')}; positive => better on plausible than implausible.",
                }
            )

    # Cross 4: triangulation density
    tri_main = _safe_read(DER / "ALGO_P3_triangulation.csv")
    if tri_main is not None and not tri_main.empty and "convergence_label" in tri_main.columns:
        vc = tri_main["convergence_label"].value_counts(normalize=True)
        rows.append(
            {
                "relation": "triangulation_original_non_ambiguous_rate",
                "family": "ALGO",
                "model": "all",
                "n": int(len(tri_main)),
                "value": float(1.0 - vc.get("ambiguous", 0.0)),
                "interpretation": "fraction with label != ambiguous in current pipeline",
            }
        )
    tri_plus = _safe_read(DER / "deep_triangulation_plus.csv")
    if tri_plus is not None and not tri_plus.empty and "tri_plus_label" in tri_plus.columns:
        for fam, g in tri_plus.groupby("family"):
            vc = g["tri_plus_label"].value_counts(normalize=True)
            strong = float(vc.get("computation_signal_plus", 0.0) + vc.get("retrieval_signal_plus", 0.0))
            rows.append(
                {
                    "relation": "triangulation_plus_strong_signal_rate",
                    "family": fam,
                    "model": "all",
                    "n": int(len(g)),
                    "value": strong,
                    "interpretation": "fraction with strong +/-2 vote margin",
                }
            )

    return pd.DataFrame(rows)


def build_report(
    raw_files: list[Path],
    derived_files: list[Path],
    profiles: pd.DataFrame,
    all_deductions: pd.DataFrame,
    cross_df: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Scientific file-wise audit")
    lines.append("")
    lines.append("This report audits each file with concrete deductions, then summarizes cross-probe relations.")
    lines.append("")
    lines.append("## Global inventory")
    lines.append("")
    lines.append(f"- Raw files audited: {len(raw_files)}")
    lines.append(f"- Derived files audited: {len(derived_files)}")
    lines.append(f"- Total deductions generated: {len(all_deductions)}")
    lines.append("")

    # Concise top-level profile table as CSV text
    lines.append("## File profile table (compact)")
    lines.append("")
    cols = [
        "file",
        "rows",
        "cols",
        "n_models",
        "n_problem_ids",
        "n_subtypes",
        "primary_correct_col",
        "primary_correct_rate",
        "parse_failed_rate",
    ]
    lines.append(profiles[cols].sort_values("file").to_csv(index=False).strip())
    lines.append("")

    lines.append("## File-by-file deductions")
    lines.append("")
    for p in raw_files + derived_files:
        file = p.name
        lines.append(f"### `{file}`")
        prof = profiles[profiles["file"] == file]
        if not prof.empty:
            r = prof.iloc[0]
            lines.append(
                f"- Shape: rows={int(r['rows'])}, cols={int(r['cols'])}, models={int(r['n_models'])}, "
                f"problem_ids={int(r['n_problem_ids'])}."
            )
        sub = all_deductions[all_deductions["file"] == file]
        if sub.empty:
            lines.append("- No specialized deduction emitted (schema-only or sparse file).")
        else:
            for d in sub["deduction"].tolist():
                lines.append(d)
        lines.append("")

    lines.append("## Cross-probe relation summary")
    lines.append("")
    if cross_df.empty:
        lines.append("- No cross-relations computed.")
    else:
        lines.append(cross_df.to_csv(index=False).strip())
    lines.append("")

    _write_markdown(DER / "scientific_filewise_audit.md", lines)


def main() -> None:
    raw_files, derived_files = list_targets()

    profile_rows: list[dict[str, Any]] = []
    deductions_rows: list[dict[str, str]] = []

    for p in raw_files + derived_files:
        df = _safe_read(p)
        if df is None:
            continue
        profile_rows.append(generic_profile(p, df))
        deductions = file_deductions(p, df)
        for d in deductions:
            deductions_rows.append({"file": p.name, "deduction": d})

    profiles = pd.DataFrame(profile_rows)
    deductions_df = pd.DataFrame(deductions_rows)
    cross_df = cross_relations()

    profiles.to_csv(DER / "scientific_file_profiles.csv", index=False)
    deductions_df.to_csv(DER / "scientific_file_deductions.csv", index=False)
    cross_df.to_csv(DER / "scientific_cross_relations.csv", index=False)
    build_report(raw_files, derived_files, profiles, deductions_df, cross_df)

    print("Wrote scientific audit outputs:")
    for name in [
        "scientific_file_profiles.csv",
        "scientific_file_deductions.csv",
        "scientific_cross_relations.csv",
        "scientific_filewise_audit.md",
    ]:
        print(f" - results/derived/{name}")


if __name__ == "__main__":
    main()
