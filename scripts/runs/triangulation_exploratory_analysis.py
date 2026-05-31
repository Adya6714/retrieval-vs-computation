#!/usr/bin/env python3
"""Step 10 — Triangulation exploratory analysis (no threshold lock-in).

Characterizes v2 label behaviour before Step 13 design choices.

Outputs:
    results/derived/triangulation_label_distribution.csv
    results/derived/triangulation_vote_fire_rates.csv
    results/derived/triangulation_sweep_stability.csv
    results/derived/triangulation_legacy_v2_overlap.csv
    results/derived/triangulation_exploratory_summary.md
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DER = ROOT / "results" / "derived"
DATA = ROOT / "data" / "problems"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.runs.triangulation_v2 import (  # noqa: E402
    TriThresholds,
    apply_votes,
    build_signal_frame,
    load_p1_long,
)

VOTE_COLS = [
    "p1_rename_fragile",
    "p1_w3_keep",
    "p1_multi_variant",
    "p1_vri_high",
    "p2_cci_comp",
    "p2_cci_retr",
    "p2_match_first",
    "p2_crit_step",
    "p2_impl_recovery",
    "p2_impl_fail",
    "p3_contam_high",
    "p3_contam_low",
    "p3_depth_high",
]

EXCLUDE_MODELS = {"mock", "deepseek/deepseek-r1-distill-llama-70b"}


def _load_subtype_map() -> pd.DataFrame:
    rows: list[dict] = []
    for fam in ("ALGO", "GSM", "BW"):
        path = DATA / f"question_bank_{fam.lower()}.csv"
        if not path.exists():
            continue
        bank = pd.read_csv(path, dtype=str)
        canon = bank[bank["variant_type"].astype(str).str.lower() == "canonical"]
        for _, r in canon.drop_duplicates("problem_id").iterrows():
            rows.append(
                {
                    "family": fam,
                    "problem_id": r["problem_id"],
                    "subtype": r.get("problem_subtype", "") or "(all)",
                }
            )
    return pd.DataFrame(rows)


def _clean_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out[~out["model"].isin(EXCLUDE_MODELS)]
    out = out[out["model"].astype(str).str.strip() != ""]
    return out


def label_distribution(labels: pd.DataFrame) -> pd.DataFrame:
    lab = _clean_labels(labels)
    sub = _load_subtype_map()
    lab = lab.merge(sub, on=["family", "problem_id"], how="left")
    lab["subtype"] = lab["subtype"].fillna("(all)")
    rows: list[dict] = []
    for keys, g in lab.groupby(["family", "model", "subtype", "tri_v2_label"], dropna=False):
        fam, model, subtype, label = keys
        rows.append({"family": fam, "model": model, "subtype": subtype, "tri_v2_label": label, "n": len(g)})
    dist = pd.DataFrame(rows)
    if dist.empty:
        return dist
    totals = dist.groupby(["family", "model", "subtype"])["n"].transform("sum")
    dist["pct"] = dist["n"] / totals
    return dist.sort_values(["family", "model", "subtype", "n"], ascending=[True, True, True, False])


def vote_fire_rates(labels: pd.DataFrame) -> pd.DataFrame:
    lab = _clean_labels(labels)
    rows: list[dict] = []
    for fam, gfam in lab.groupby("family"):
        for col in VOTE_COLS:
            if col not in gfam.columns:
                continue
            vals = pd.to_numeric(gfam[col], errors="coerce").fillna(0)
            rows.append(
                {
                    "family": fam,
                    "scope": "family",
                    "model": "(all)",
                    "vote_signal": col,
                    "fire_rate": float((vals >= 0.5).mean()),
                    "n_instances": len(gfam),
                }
            )
        for model, gm in gfam.groupby("model"):
            for col in VOTE_COLS:
                if col not in gm.columns:
                    continue
                vals = pd.to_numeric(gm[col], errors="coerce").fillna(0)
                rows.append(
                    {
                        "family": fam,
                        "scope": "model",
                        "model": model,
                        "vote_signal": col,
                        "fire_rate": float((vals >= 0.5).mean()),
                        "n_instances": len(gm),
                    }
                )
    out = pd.DataFrame(rows)
    return out.sort_values(["family", "scope", "model", "fire_rate"], ascending=[True, True, True, False])


def _config_from_row(row: pd.Series) -> TriThresholds:
    return TriThresholds(
        w3_retrieval_max=float(row["w3_retrieval_max"]),
        w3_computation_min=float(row["w3_computation_min"]),
        contam_retrieval_min=float(row["contam_retrieval_min"]),
        contam_computation_max=float(row["contam_computation_max"]),
        cci_computation_min=float(row["cci_computation_min"]),
        cci_retrieval_max=float(row["cci_retrieval_max"]),
        min_votes=int(row["min_votes"]),
        vote_margin=int(row["vote_margin"]),
    )


def instance_stability(base: pd.DataFrame, sweep: pd.DataFrame) -> pd.DataFrame:
    """Per-instance label diversity across default, best sweep, and single-param neighbors."""
    if base.empty or sweep.empty:
        return pd.DataFrame()

    default = TriThresholds()
    configs: dict[str, TriThresholds] = {"default": default}

    best = sweep.sort_values("pct_strong_total", ascending=False).iloc[0]
    configs[f"best_sweep_id_{int(best['param_id'])}"] = _config_from_row(best)

    # Single-parameter neighbors of default
    configs["min_votes_3"] = TriThresholds(min_votes=3, vote_margin=2)
    configs["vote_margin_1"] = TriThresholds(min_votes=2, vote_margin=1)
    configs["contam_hi_0.5"] = TriThresholds(contam_retrieval_min=0.5)
    configs["cci_comp_0.6"] = TriThresholds(cci_computation_min=0.6)

    label_cols: dict[str, pd.Series] = {}
    for name, th in configs.items():
        labeled = apply_votes(base, th)
        key = labeled["family"].astype(str) + "|" + labeled["problem_id"] + "|" + labeled["model"].astype(str)
        label_cols[name] = pd.Series(labeled["tri_v2_label"].values, index=key)

    mat = pd.DataFrame(label_cols)
    mat = mat[~mat.index.str.split("|").str[2].isin(EXCLUDE_MODELS)]

    rows: list[dict] = []
    n_configs = len(configs)
    distinct = mat.nunique(axis=1)
    rows.append(
        {
            "metric": "instances_any_flip",
            "value": int((distinct > 1).sum()),
            "n_instances": len(mat),
            "pct": float((distinct > 1).mean()),
            "note": f"Distinct labels across {n_configs} reference configs",
        }
    )
    rows.append(
        {
            "metric": "instances_all_same",
            "value": int((distinct == 1).sum()),
            "n_instances": len(mat),
            "pct": float((distinct == 1).mean()),
            "note": "Stable label across reference configs",
        }
    )

    # Pairwise flip rates for key pairs
    pairs = [
        ("default", "best_sweep_id_" + str(int(best["param_id"]))),
        ("default", "min_votes_3"),
        ("default", "vote_margin_1"),
    ]
    for a, b in pairs:
        if a not in mat.columns or b not in mat.columns:
            continue
        flip = mat[a] != mat[b]
        rows.append(
            {
                "metric": f"flip_{a}_vs_{b}",
                "value": int(flip.sum()),
                "n_instances": len(mat),
                "pct": float(flip.mean()),
                "note": "Instance label changed between configs",
            }
        )

    # Sweep-level: adjacent param pairs (aggregate pct_strong delta)
    param_cols = [
        "w3_retrieval_max",
        "w3_computation_min",
        "contam_retrieval_min",
        "contam_computation_max",
        "cci_computation_min",
        "cci_retrieval_max",
        "min_votes",
        "vote_margin",
    ]
    adj_rows: list[dict] = []
    for col in param_cols:
        vals = sorted(sweep[col].unique())
        for i in range(len(vals) - 1):
            a, b = vals[i], vals[i + 1]
            sa = sweep[sweep[col] == a]["pct_strong_total"]
            sb = sweep[sweep[col] == b]["pct_strong_total"]
            adj_rows.append(
                {
                    "param": col,
                    "from_val": a,
                    "to_val": b,
                    "mean_pct_strong_from": float(sa.mean()),
                    "mean_pct_strong_to": float(sb.mean()),
                    "mean_delta_strong": float(sb.mean() - sa.mean()),
                    "n_configs_from": len(sa),
                    "n_configs_to": len(sb),
                }
            )
    adj_df = pd.DataFrame(adj_rows)

    stability = pd.DataFrame(rows)
    stability.attrs["adjacent_param_sweep"] = adj_df
    return stability


def legacy_v2_overlap(labels: pd.DataFrame) -> pd.DataFrame:
    lab = _clean_labels(labels)
    lab = lab[lab["family"] == "ALGO"].copy()
    lab = lab[lab["legacy_label"].astype(str).str.strip() != ""]

    def _strong_legacy(s: str) -> str:
        if s in {"retrieval_signal", "computation_signal"}:
            return "strong"
        if s in {"ambiguous", "mixed"}:
            return s
        return "other"

    def _strong_v2(s: str) -> str:
        if s in {"retrieval", "computation"}:
            return "strong"
        if s in {"weak_retrieval", "weak_computation"}:
            return "weak"
        if s in {"mixed", "insufficient"}:
            return s
        return "other"

    lab["legacy_bucket"] = lab["legacy_label"].map(_strong_legacy)
    lab["v2_bucket"] = lab["tri_v2_label"].map(_strong_v2)

    ct = (
        lab.groupby(["legacy_label", "tri_v2_label"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    ct["pct_of_overlap"] = ct["n"] / ct["n"].sum()

    summary = (
        lab.groupby(["legacy_bucket", "v2_bucket"], dropna=False)
        .size()
        .reset_index(name="n")
    )
    summary["pct"] = summary["n"] / summary["n"].sum()

    # attach summary as attrs on detailed crosstab
    ct.attrs["bucket_summary"] = summary
    return ct


def write_summary(
    dist: pd.DataFrame,
    votes: pd.DataFrame,
    stability: pd.DataFrame,
    overlap: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    lab = _clean_labels(labels)
    lines = [
        "# Triangulation exploratory analysis (Step 10)",
        "",
        "Characterization only — **do not** treat as official paper thresholds (Step 13).",
        "",
        f"**Instances (excl. mock):** {len(lab)}",
        "",
        "## Default v2 label mix (all families)",
        "",
        "```",
        lab["tri_v2_label"].value_counts().to_string(),
        "```",
        "",
        "## Label distribution by family × model (top slices)",
        "",
        "```",
    ]
    if not dist.empty:
        top = dist.sort_values("n", ascending=False).head(25)
        lines.append(top.round(3).to_string(index=False))
    lines.append("```")
    lines.append("")

    lines.extend(["## Vote signals — highest fire rates by family", ""])
    if not votes.empty:
        fam_votes = votes[votes["scope"] == "family"].sort_values(["family", "fire_rate"], ascending=[True, False])
        for fam in sorted(fam_votes["family"].unique()):
            sub = fam_votes[fam_votes["family"] == fam][["vote_signal", "fire_rate"]].head(5)
            lines.append(f"**{fam}:**")
            lines.append("```")
            lines.append(sub.round(3).to_string(index=False))
            lines.append("```")
            lines.append("")

    lines.extend(["## Stability (reference config set)", ""])
    if not stability.empty:
        lines.append("```")
        lines.append(stability.round(4).to_string(index=False))
        lines.append("```")
        adj = stability.attrs.get("adjacent_param_sweep")
        if adj is not None and not adj.empty:
            lines.extend(["", "### Sweep: mean Δ strong-label rate between adjacent param values", ""])
            lines.append("```")
            adj2 = adj.copy()
            adj2["abs_delta"] = adj2["mean_delta_strong"].abs()
            lines.append(adj2.sort_values("abs_delta", ascending=False).head(12).drop(columns=["abs_delta"]).round(4).to_string(index=False))
            lines.append("```")
    lines.append("")

    lines.extend(["## Legacy vs v2 overlap (ALGO, n=330 with legacy label)", ""])
    if not overlap.empty:
        bucket = overlap.attrs.get("bucket_summary")
        if bucket is not None:
            lines.append("**Bucket summary:**")
            lines.append("```")
            lines.append(bucket.round(3).to_string(index=False))
            lines.append("```")
            lines.append("")
        lines.append("**Full crosstab (top rows):**")
        lines.append("```")
        lines.append(overlap.sort_values("n", ascending=False).head(15).round(3).to_string(index=False))
        lines.append("```")

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- **~37% insufficient** under default thresholds — many instances lack enough firing votes.",
            "- **P3 contam_low** and **P1 w3_keep / multi_variant** (GSM) fire most often; **p2_match_first** fires rarely.",
            "- Legacy strong labels (~3%) vs v2 strong (~58% at best sweep) — legacy is much stricter.",
            "- Label flips most when **`vote_margin`** (1→2) or **`min_votes`** (2→3) change — Step 13 must lock these.",
            "",
            "## Files",
            "",
            "- `triangulation_label_distribution.csv`",
            "- `triangulation_vote_fire_rates.csv`",
            "- `triangulation_sweep_stability.csv`",
            "- `triangulation_legacy_v2_overlap.csv`",
            "",
        ]
    )
    (DER / "triangulation_exploratory_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)

    labels_path = DER / "triangulation_v2_labels.csv"
    sweep_path = DER / "triangulation_threshold_sweep.csv"
    if not labels_path.exists():
        raise SystemExit("Run triangulation_v2.py first.")

    labels = pd.read_csv(labels_path)
    sweep = pd.read_csv(sweep_path) if sweep_path.exists() else pd.DataFrame()

    p1 = load_p1_long()
    base = build_signal_frame(p1)

    dist = label_distribution(labels)
    votes = vote_fire_rates(labels)
    stability = instance_stability(base, sweep)
    overlap = legacy_v2_overlap(labels)

    dist.to_csv(DER / "triangulation_label_distribution.csv", index=False)
    votes.to_csv(DER / "triangulation_vote_fire_rates.csv", index=False)

    stability.to_csv(DER / "triangulation_sweep_stability.csv", index=False)
    adj = stability.attrs.get("adjacent_param_sweep")
    if adj is not None and not adj.empty:
        adj.to_csv(DER / "triangulation_sweep_param_sensitivity.csv", index=False)

    overlap.to_csv(DER / "triangulation_legacy_v2_overlap.csv", index=False)
    bucket = overlap.attrs.get("bucket_summary")
    if bucket is not None:
        bucket.to_csv(DER / "triangulation_legacy_v2_buckets.csv", index=False)

    write_summary(dist, votes, stability, overlap, labels)

    print("Wrote triangulation exploratory pack:")
    for name in [
        "triangulation_label_distribution.csv",
        "triangulation_vote_fire_rates.csv",
        "triangulation_sweep_stability.csv",
        "triangulation_sweep_param_sensitivity.csv",
        "triangulation_legacy_v2_overlap.csv",
        "triangulation_legacy_v2_buckets.csv",
        "triangulation_exploratory_summary.md",
    ]:
        p = DER / name
        if p.exists():
            print(f"  results/derived/{name}")


if __name__ == "__main__":
    main()
