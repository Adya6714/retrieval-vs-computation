#!/usr/bin/env python3
"""O3: Audit M1–N5 inferential stats; replace naive p with cluster-bootstrap p.

Policy: CI primary, p secondary; p must be coherent with its own percentile CI
(same cluster-resampled bootstrap distribution; H0: association = 0).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.behavioral.retention import MIN_CANONICAL_FOR_RETENTION  # noqa: E402
from probes.common.clones import cluster_ids_for  # noqa: E402
from probes.common.cluster_inference import (  # noqa: E402
    ci_excludes_zero,
    cluster_bootstrap_assoc,
    iid_bootstrap_assoc,
    sig_at,
)
from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
RAW = REPO_ROOT / "results" / "raw"
OUT = DER / "ALL_INFERENCE_AUDIT.csv"

PAPER_MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
N_BOOT = 5000
SEED = 42
ALPHA = 0.05


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _f(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _round(x, nd=4):
    if x is None or x == "":
        return ""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if v != v:
        return ""
    return round(v, nd)


def _load_p1(*families: str) -> pd.DataFrame:
    want = set(families) if families else {"GSM", "ALGO", "BW"}
    parts = []
    for path in sorted(DER.glob("*_P1_*rescored.csv")):
        if "review" in path.name.lower():
            continue
        if path.name.startswith("ALGO_"):
            fam = "ALGO"
        elif path.name.startswith("BW_"):
            fam = "BW"
        elif path.name.startswith("GSM_"):
            fam = "GSM"
        else:
            continue
        if fam not in want:
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "included" not in df.columns:
            continue
        df = df[_is_true(df["included"])].copy()
        df = filter_excluded(df, family=fam)
        df["family"] = fam
        df["model_short"] = df["model"].map(PAPER_MODELS).fillna(df["model"])
        df["variant"] = df["variant_type"].map(normalize_variant)
        ok = df["rescored_correct"] if "rescored_correct" in df.columns else df.get("verified", "")
        df["ok"] = _is_true(ok)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates(
        ["family", "problem_id", "variant", "model_short"], keep="last",
    )


def _load_contam(*families: str) -> pd.DataFrame:
    rows = []
    for fam in families:
        path = RAW / f"{fam}_P3_contamination.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        df["family"] = fam
        df["contamination_score"] = pd.to_numeric(df["contamination_score"], errors="coerce")
        rows.append(df[["family", "problem_id", "contamination_score"]])
    return pd.concat(rows, ignore_index=True).drop_duplicates(["family", "problem_id"])


def _attach_clusters(df: pd.DataFrame, family_col: str = "family") -> pd.DataFrame:
    out = df.copy()
    out["cluster_id"] = out["problem_id"].astype(str)
    algo = out[family_col] == "ALGO"
    if algo.any():
        out.loc[algo, "cluster_id"] = cluster_ids_for(out.loc[algo, "problem_id"].astype(str).tolist())
    return out


def _row(
    *,
    analysis_id: str,
    statistic: str,
    estimate,
    ci_low,
    ci_high,
    p_clustered,
    n,
    n_clusters,
    clustering_variable: str,
    previously_reported_p,
    null_test: str,
    notes: str = "",
) -> dict:
    prev = _f(previously_reported_p)
    pnew = _f(p_clustered)
    prev_sig = sig_at(prev, ALPHA)
    new_sig = sig_at(pnew, ALPHA)
    changed = False
    if prev_sig is not None and new_sig is not None:
        changed = bool(prev_sig != new_sig)
    # Also flag CI/p incoherence with *previous* reporting (CI vs naive p)
    lo, hi = _f(ci_low), _f(ci_high)
    ci_sig = ci_excludes_zero(lo, hi)
    if ci_sig is not None and new_sig is not None and ci_sig != new_sig:
        # Should not happen with coherent bootstrap p; record in notes
        notes = (notes + "; WARN_ci_p_mismatch").strip("; ")
    return {
        "analysis_id": analysis_id,
        "statistic": statistic,
        "estimate": _round(estimate),
        "ci_low": _round(ci_low),
        "ci_high": _round(ci_high),
        "p_clustered": _round(pnew, 4) if pnew == pnew else "",
        "n": n,
        "n_clusters": n_clusters,
        "clustering_variable": clustering_variable,
        "previously_reported_p": _round(prev, 4) if prev == prev else "",
        "changed_conclusion": bool(changed),
        "null_test": null_test,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# M1
# ---------------------------------------------------------------------------

def audit_m1(rows: list[dict]) -> pd.DataFrame:
    print("\n=== M1: W6 gap vs contamination ===")
    print("estimator=Spearman(contam, can-W6); null=H0 rho=0; CI=cluster bootstrap; OLD p=naive spearmanr")
    p1 = _load_p1("BW", "ALGO")
    contam = _load_contam("BW", "ALGO")
    can = p1[p1["variant"] == "canonical"][["family", "problem_id", "model_short", "ok"]].rename(
        columns={"ok": "canonical_ok"},
    )
    w6 = p1[p1["variant"] == "W6"][["family", "problem_id", "model_short", "ok"]].rename(
        columns={"ok": "w6_ok"},
    )
    merged = can.merge(w6, on=["family", "problem_id", "model_short"], how="inner")
    merged["w6_gap"] = merged["canonical_ok"].astype(int) - merged["w6_ok"].astype(int)
    merged = merged.merge(contam, on=["family", "problem_id"], how="left")
    merged = merged[merged["contamination_score"].notna()].copy()
    merged = _attach_clusters(merged)

    old = pd.read_csv(DER / "M1_w6_gap_vs_contamination.csv", dtype=str).fillna("")
    out_rows = []
    for fam in ("BW", "ALGO"):
        for model in sorted(merged.loc[merged["family"] == fam, "model_short"].unique()):
            sub = merged[(merged["family"] == fam) & (merged["model_short"] == model)]
            if len(sub) < 5:
                continue
            clust = "clone_family" if fam == "ALGO" else "problem_id"
            res = cluster_bootstrap_assoc(
                sub["contamination_score"], sub["w6_gap"], sub["cluster_id"],
                kind="spearman", n_boot=N_BOOT, seed=SEED,
            )
            prev = old[(old["family"] == fam) & (old["model"] == model)]
            prev_p = prev["p_value"].iloc[0] if len(prev) else ""
            print(
                f"  {fam}/{model}: rho={res['estimate']:.4f} CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] "
                f"p_old={prev_p} p_clust={res['p_clustered']:.4f} n={res['n']} n_clust={res['n_clusters']} "
                f"clustering={clust}"
            )
            rows.append(
                _row(
                    analysis_id=f"M1_{fam}_{model}",
                    statistic="spearman_contam_vs_w6_gap",
                    estimate=res["estimate"],
                    ci_low=res["ci_low"],
                    ci_high=res["ci_high"],
                    p_clustered=res["p_clustered"],
                    n=res["n"],
                    n_clusters=res["n_clusters"],
                    clustering_variable=clust,
                    previously_reported_p=prev_p,
                    null_test="cluster_bootstrap_H0_rho=0",
                )
            )
            out_rows.append(
                {
                    "family": fam,
                    "model": model,
                    "n": res["n"],
                    "n_clusters": res["n_clusters"],
                    "spearman_rho": _round(res["estimate"]),
                    "ci_low": _round(res["ci_low"]),
                    "ci_high": _round(res["ci_high"]),
                    "p_value": _round(res["p_clustered"]),
                    "p_value_method": "cluster_bootstrap_two_sided",
                    "contamination_column": "contamination_score",
                    "gap_definition": "canonical_ok_minus_w6_ok",
                    "bootstrap": f"cluster_by_{clust}",
                    "n_boot": N_BOOT,
                    "seed": SEED,
                }
            )
    pd.DataFrame(out_rows).to_csv(DER / "M1_w6_gap_vs_contamination.csv", index=False)
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# M2 / N1 — descriptive only
# ---------------------------------------------------------------------------

def audit_m2_n1(rows: list[dict]) -> None:
    print("\n=== M2 (K3 ALGO/BW matched) / N1 (BW stratified) ===")
    print("estimator=accuracy/structural deltas; null=NONE; no p-values reported; n=matched pairs")
    for path, aid in [
        (DER / "K3_bw_canonical_w6_matched_report.csv", "M2_BW_matched"),
        (DER / "K3_algo_canonical_w6_matched_report.csv", "M2_ALGO_matched"),
        (DER / "N1_bw_w6_stratified_accuracy.csv", "N1_BW_stratified"),
    ]:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        n = int(df["n_pairs"].iloc[0]) if "n_pairs" in df.columns else len(df)
        rows.append(
            _row(
                analysis_id=aid,
                statistic="descriptive_delta_only",
                estimate="",
                ci_low="",
                ci_high="",
                p_clustered="",
                n=n,
                n_clusters="",
                clustering_variable="n/a",
                previously_reported_p="",
                null_test="none_descriptive",
                notes="no inferential test in original report",
            )
        )
        print(f"  {aid}: descriptive only (n≈{n})")


# ---------------------------------------------------------------------------
# M3 construct validity
# ---------------------------------------------------------------------------

def audit_m3_construct(rows: list[dict]) -> None:
    print("\n=== M3 construct validity (phi/retention vs can_acc) ===")
    print("estimator=Spearman; null=H0 rho=0; CI=bootstrap by family; OLD p=naive spearmanr")
    phi = pd.read_csv(DER / "P1_phi_canonical_w3.csv", dtype=str).fillna("")
    for col in ["phi", "retention_w3", "acc_canonical"]:
        phi[col] = pd.to_numeric(phi[col], errors="coerce")
    phi = phi[phi["acc_canonical"] >= MIN_CANONICAL_FOR_RETENTION].copy()
    old = pd.read_csv(DER / "P1_construct_validity.csv", dtype=str).fillna("")
    out_rows = []
    for x, y, label, row_c in [
        ("phi", "acc_canonical", "discriminant_phi_vs_canonical_accuracy", "phi"),
        ("retention_w3", "acc_canonical", "discriminant_retention_vs_canonical_accuracy", "retention"),
    ]:
        sub = phi[[x, y, "family"]].dropna()
        res = cluster_bootstrap_assoc(
            sub[x], sub[y], sub["family"].astype(str),
            kind="spearman", n_boot=N_BOOT, seed=SEED,
        )
        prev = old[old["analysis"] == label]
        prev_p = prev["p_value"].iloc[0] if len(prev) else ""
        print(
            f"  {label}: rho={res['estimate']:.4f} CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] "
            f"p_old={prev_p} p_clust={res['p_clustered']:.4f} n={res['n']} n_clust={res['n_clusters']}"
        )
        rows.append(
            _row(
                analysis_id=f"M3_{label}",
                statistic="spearman",
                estimate=res["estimate"],
                ci_low=res["ci_low"],
                ci_high=res["ci_high"],
                p_clustered=res["p_clustered"],
                n=res["n"],
                n_clusters=res["n_clusters"],
                clustering_variable="family",
                previously_reported_p=prev_p,
                null_test="cluster_bootstrap_H0_rho=0",
            )
        )
        out_rows.append(
            {
                "analysis": label,
                "row_construct": row_c,
                "col_construct": "canonical_accuracy",
                "spearman_rho": _round(res["estimate"], 3),
                "ci_low": _round(res["ci_low"], 3),
                "ci_high": _round(res["ci_high"], 3),
                "p_value": _round(res["p_clustered"], 3),
                "p_value_method": "cluster_bootstrap_two_sided",
                "n_cells": res["n"],
                "can_acc_floor": MIN_CANONICAL_FOR_RETENTION,
                "bootstrap": "cluster_by_family",
                "n_boot": N_BOOT,
                "seed": SEED,
                "note": (
                    "retention-vs-phi omitted — same contingency table, not independent evidence"
                    if x == "phi"
                    else ""
                ),
            }
        )
    pd.DataFrame(out_rows).to_csv(DER / "P1_construct_validity.csv", index=False)


# ---------------------------------------------------------------------------
# N2 / M3 P2 convergence
# ---------------------------------------------------------------------------

def audit_n2(rows: list[dict]) -> None:
    print("\n=== N2/M3: P2 CCI vs W3 correctness ===")
    print("estimator=point-biserial(CCI,W3); null=H0 r=0; CI=cluster bootstrap; OLD p=naive pointbiserialr")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "p2_p1_convergence",
        REPO_ROOT / "scripts/consolidate/p2_p1_convergence.py",
    )
    p2mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(p2mod)

    old = pd.read_csv(DER / "P2_P1_convergence.csv", dtype=str).fillna("")
    frames = []
    family_frames = []
    out_rows = []

    if p2mod.GSM_P2_IN.exists():
        gsm_m = p2mod._merge_p2_p1(
            pd.read_csv(p2mod.GSM_P2_IN, dtype=str).fillna(""),
            p2mod._load_p1("GSM"),
            family="GSM",
        )
        frames.append(gsm_m)
        family_frames.append(("GSM", gsm_m))
    if p2mod.ALGO_P2_IN.exists():
        algo_m = p2mod._merge_p2_p1(
            pd.read_csv(p2mod.ALGO_P2_IN, dtype=str).fillna(""),
            p2mod._load_p1("ALGO"),
            family="ALGO",
        )
        frames.append(algo_m)
        family_frames.append(("ALGO", algo_m))

    def _pb_scopes(merged: pd.DataFrame, prefix: str, clust_var: str):
        for subset_name, sub in [
            (f"{prefix}_all_instances_with_p2", merged),
            (f"{prefix}_canonical_correct_subset", merged[merged["canonical_ok"].astype(bool)]),
        ]:
            if sub.empty:
                continue
            res = cluster_bootstrap_assoc(
                sub["cci"], sub["w3_correct"], sub["cluster_id"].astype(str),
                kind="pointbiserial", n_boot=N_BOOT, seed=SEED,
            )
            prev = old[old["scope"] == subset_name]
            prev_p = prev["p_value"].iloc[0] if len(prev) else ""
            print(
                f"  {subset_name}: r={res['estimate']:.4f} CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] "
                f"p_old={prev_p} p_clust={res['p_clustered']:.4f} n={res['n']} n_clust={res['n_clusters']}"
            )
            rows.append(
                _row(
                    analysis_id=f"N2_{subset_name}",
                    statistic="pointbiserial_cci_w3",
                    estimate=res["estimate"],
                    ci_low=res["ci_low"],
                    ci_high=res["ci_high"],
                    p_clustered=res["p_clustered"],
                    n=res["n"],
                    n_clusters=res["n_clusters"],
                    clustering_variable=clust_var,
                    previously_reported_p=prev_p,
                    null_test="cluster_bootstrap_H0_r=0",
                )
            )
            out_rows.append(
                {
                    "analysis": "pointbiserial_cci_w3_correct",
                    "scope": subset_name,
                    "statistic": _round(res["estimate"], 3),
                    "ci_low": _round(res["ci_low"], 3),
                    "ci_high": _round(res["ci_high"], 3),
                    "p_value": _round(res["p_clustered"], 4),
                    "p_value_method": "cluster_bootstrap_two_sided",
                    "n": res["n"],
                    "n_clusters": res["n_clusters"],
                    "note": "P2 CCI vs W3; CI+p from same cluster bootstrap",
                }
            )

    for fam, fr in family_frames:
        clust = "clone_family" if fam == "ALGO" else "problem_id"
        _pb_scopes(fr, fam, clust)

    if frames:
        pooled = pd.concat(frames, ignore_index=True)
        pooled["cluster_id"] = pooled.apply(lambda r: f"{r['family']}:{r['cluster_id']}", axis=1)
        _pb_scopes(pooled, "GSM_ALGO_pooled", "family:cluster")

    # across-model spearman (units = models; iid bootstrap)
    print("  across-model Spearman(mean_CCI, retention): units=models (iid bootstrap, not clone)")
    if p2mod.PHI_IN.exists() and family_frames:
        phi = pd.read_csv(p2mod.PHI_IN, dtype=str).fillna("")
        for col in ["retention_w3", "acc_canonical"]:
            phi[col] = pd.to_numeric(phi[col], errors="coerce")
        phi = phi[phi["acc_canonical"] >= MIN_CANONICAL_FOR_RETENTION].copy()
        for fam, merged in family_frames:
            cci_means = merged.groupby("model_short", as_index=False)["cci"].mean().rename(
                columns={"cci": "mean_cci", "model_short": "model"},
            )
            model_df = phi[phi["family"] == fam].merge(cci_means, on="model", how="inner")
            scope = f"{fam}_across_models"
            if len(model_df) < 2:
                continue
            res = iid_bootstrap_assoc(
                model_df["mean_cci"], model_df["retention_w3"],
                kind="spearman", n_boot=N_BOOT, seed=SEED,
            )
            prev = old[old["scope"] == scope]
            prev_p = prev["p_value"].iloc[0] if len(prev) else ""
            print(
                f"  {scope}: rho={res['estimate']:.4f} CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] "
                f"p_old={prev_p} p_boot={res['p_clustered']:.4f} n={res['n']}"
            )
            rows.append(
                _row(
                    analysis_id=f"N2_{scope}",
                    statistic="spearman_mean_cci_retention",
                    estimate=res["estimate"],
                    ci_low=res["ci_low"],
                    ci_high=res["ci_high"],
                    p_clustered=res["p_clustered"],
                    n=res["n"],
                    n_clusters=res["n"],
                    clustering_variable="model_iid",
                    previously_reported_p=prev_p,
                    null_test="iid_bootstrap_H0_rho=0",
                    notes="sampling unit=model; no clone clustering",
                )
            )
            out_rows.append(
                {
                    "analysis": "spearman_mean_cci_retention",
                    "scope": scope,
                    "statistic": _round(res["estimate"], 3),
                    "ci_low": _round(res["ci_low"], 3),
                    "ci_high": _round(res["ci_high"], 3),
                    "p_value": _round(res["p_clustered"], 4),
                    "p_value_method": "iid_bootstrap_two_sided",
                    "n": res["n"],
                    "n_clusters": res["n"],
                    "note": f"{fam} models; iid bootstrap over models",
                }
            )

    pd.DataFrame(out_rows).to_csv(DER / "P2_P1_convergence.csv", index=False)


# ---------------------------------------------------------------------------
# N3
# ---------------------------------------------------------------------------

def audit_n3(rows: list[dict]) -> None:
    print("\n=== N3: mech rank-shift vs W3 correctness ===")
    print("estimator=Spearman(rank_shift, w3_ok); null=H0 rho=0; CI=cluster bootstrap; OLD p=naive spearmanr")
    inst = pd.read_csv(DER / "N3_algo_mech_behavior_instances.csv", dtype=str).fillna("")
    inst["rank_shift_canonical_minus_w3"] = pd.to_numeric(inst["rank_shift_canonical_minus_w3"], errors="coerce")
    inst["w3_ok"] = inst["w3_ok"].astype(str).str.lower().isin({"true", "1", "yes"}).astype(int)
    if "cluster_id" not in inst.columns or inst["cluster_id"].eq("").all():
        inst["cluster_id"] = cluster_ids_for(inst["problem_id"].astype(str).tolist())
    old = pd.read_csv(DER / "N3_algo_mech_behavior_link.csv", dtype=str).fillna("")
    out_rows = []
    for model in sorted(inst["model"].unique()):
        sub = inst[inst["model"] == model].copy()
        if sub["w3_ok"].nunique() < 2 or sub["rank_shift_canonical_minus_w3"].nunique() < 2:
            prev = old[old["model"] == model]
            prev_p = prev["p_value"].iloc[0] if len(prev) else ""
            print(f"  {model}: insufficient variation (n={len(sub)})")
            rows.append(
                _row(
                    analysis_id=f"N3_{model}",
                    statistic="spearman_rank_shift_vs_w3",
                    estimate="",
                    ci_low="",
                    ci_high="",
                    p_clustered="",
                    n=len(sub),
                    n_clusters=sub["cluster_id"].nunique(),
                    clustering_variable="clone_family",
                    previously_reported_p=prev_p,
                    null_test="cluster_bootstrap_H0_rho=0",
                    notes="insufficient variation",
                )
            )
            out_rows.append(
                {
                    "model": model,
                    "n": len(sub),
                    "n_clusters": sub["cluster_id"].nunique(),
                    "spearman_rho": "",
                    "ci_low": "",
                    "ci_high": "",
                    "p_value": "",
                    "p_value_method": "cluster_bootstrap_two_sided",
                    "x": "rank_shift_canonical_minus_w3",
                    "y": "w3_correct",
                    "bootstrap": "cluster_by_clone_family",
                    "n_boot": N_BOOT,
                    "seed": SEED,
                    "note": "insufficient variation — correlation undefined",
                }
            )
            continue
        res = cluster_bootstrap_assoc(
            sub["rank_shift_canonical_minus_w3"], sub["w3_ok"], sub["cluster_id"],
            kind="spearman", n_boot=N_BOOT, seed=SEED,
        )
        prev = old[old["model"] == model]
        prev_p = prev["p_value"].iloc[0] if len(prev) else ""
        print(
            f"  {model}: rho={res['estimate']:.4f} CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] "
            f"p_old={prev_p} p_clust={res['p_clustered']:.4f} n={res['n']} n_clust={res['n_clusters']}"
        )
        rows.append(
            _row(
                analysis_id=f"N3_{model}",
                statistic="spearman_rank_shift_vs_w3",
                estimate=res["estimate"],
                ci_low=res["ci_low"],
                ci_high=res["ci_high"],
                p_clustered=res["p_clustered"],
                n=res["n"],
                n_clusters=res["n_clusters"],
                clustering_variable="clone_family",
                previously_reported_p=prev_p,
                null_test="cluster_bootstrap_H0_rho=0",
            )
        )
        out_rows.append(
            {
                "model": model,
                "n": res["n"],
                "n_clusters": res["n_clusters"],
                "spearman_rho": _round(res["estimate"]),
                "ci_low": _round(res["ci_low"]),
                "ci_high": _round(res["ci_high"]),
                "p_value": _round(res["p_clustered"]),
                "p_value_method": "cluster_bootstrap_two_sided",
                "x": "rank_shift_canonical_minus_w3",
                "y": "w3_correct",
                "bootstrap": "cluster_by_clone_family",
                "n_boot": N_BOOT,
                "seed": SEED,
                "note": "Validation only — not a mechanism claim",
            }
        )
    pd.DataFrame(out_rows).to_csv(DER / "N3_algo_mech_behavior_link.csv", index=False)


# ---------------------------------------------------------------------------
# N4 — already permutation; no clone clustering
# ---------------------------------------------------------------------------

def audit_n4(rows: list[dict]) -> None:
    print("\n=== N4: Kendall W variant ordering ===")
    print("estimator=Kendall W; null=row-permutation of ranks; clustering=N/A (rankers, not clones)")
    v2 = pd.read_csv(DER / "P1_variant_ordering_v2.csv", dtype=str).fillna("")
    # previously_reported_p from broken v1 (all 1.0)
    for _, r in v2.iterrows():
        aid = f"N4_{r['analysis']}_{r['family']}_{r['model']}"
        prev_p = "1.0"  # broken column-permutation version
        p = _f(r["p_value"])
        print(
            f"  {r['analysis']} {r['family']}/{r['model']}: W={r['kendall_W']} "
            f"p_old_broken=1.0 p_perm={r['p_value']} n_rankers={r['n_rankers']}"
        )
        rows.append(
            _row(
                analysis_id=aid,
                statistic="kendall_W",
                estimate=r["kendall_W"],
                ci_low="",
                ci_high="",
                p_clustered=p,
                n=r["n_rankers"],
                n_clusters=r["n_rankers"],
                clustering_variable="none_rank_permutation",
                previously_reported_p=prev_p,
                null_test="within_row_rank_permutation",
                notes="p from fixed within-row permutation (O2); not clone clustering",
            )
        )


# ---------------------------------------------------------------------------
# N5
# ---------------------------------------------------------------------------

def audit_n5(rows: list[dict]) -> None:
    print("\n=== N5: contamination vs W3 retention ===")
    print("estimator=Spearman(contam, retained|can); null=H0 rho=0; CI=cluster bootstrap; OLD p=naive spearmanr")
    p1 = _load_p1("GSM", "ALGO", "BW")
    contam = _load_contam("GSM", "ALGO", "BW")
    can = p1[p1["variant"] == "canonical"][["family", "problem_id", "model_short", "ok"]].rename(
        columns={"ok": "canonical_ok"},
    )
    w3 = p1[p1["variant"] == "W3"][["family", "problem_id", "model_short", "ok"]].rename(
        columns={"ok": "w3_ok"},
    )
    merged = can.merge(w3, on=["family", "problem_id", "model_short"], how="inner")
    merged = merged[merged["canonical_ok"]].copy()
    merged["retained_w3"] = merged["w3_ok"].astype(int)
    merged = merged.merge(contam, on=["family", "problem_id"], how="left")
    merged = merged[merged["contamination_score"].notna()].copy()
    merged = _attach_clusters(merged)

    old = pd.read_csv(DER / "N5_contamination_vs_retention.csv", dtype=str).fillna("")
    out_rows = []
    for fam in ("GSM", "ALGO", "BW"):
        for model in sorted(merged.loc[merged["family"] == fam, "model_short"].unique()):
            sub = merged[(merged["family"] == fam) & (merged["model_short"] == model)]
            if len(sub) < 5:
                continue
            clust = "clone_family" if fam == "ALGO" else "problem_id"
            prev = old[(old["family"] == fam) & (old["model"] == model)]
            prev_p = prev["p_value"].iloc[0] if len(prev) else ""
            if sub["retained_w3"].nunique() < 2 or sub["contamination_score"].nunique() < 2:
                print(f"  {fam}/{model}: insufficient variation n={len(sub)}")
                rows.append(
                    _row(
                        analysis_id=f"N5_{fam}_{model}",
                        statistic="spearman_contam_vs_retention",
                        estimate="",
                        ci_low="",
                        ci_high="",
                        p_clustered="",
                        n=len(sub),
                        n_clusters=sub["cluster_id"].nunique(),
                        clustering_variable=clust,
                        previously_reported_p=prev_p,
                        null_test="cluster_bootstrap_H0_rho=0",
                        notes="insufficient variation",
                    )
                )
                out_rows.append(
                    {
                        "family": fam,
                        "model": model,
                        "n": len(sub),
                        "n_clusters": sub["cluster_id"].nunique(),
                        "spearman_rho": "",
                        "ci_low": "",
                        "ci_high": "",
                        "p_value": "",
                        "p_value_method": "cluster_bootstrap_two_sided",
                        "subset": "canonical_correct_only",
                        "y": "w3_retained",
                        "contamination_column": "contamination_score",
                        "bootstrap": f"cluster_by_{clust}",
                        "n_boot": N_BOOT,
                        "seed": SEED,
                        "note": "insufficient variation",
                    }
                )
                continue
            res = cluster_bootstrap_assoc(
                sub["contamination_score"], sub["retained_w3"], sub["cluster_id"],
                kind="spearman", n_boot=N_BOOT, seed=SEED,
            )
            print(
                f"  {fam}/{model}: rho={res['estimate']:.4f} CI=[{res['ci_low']:.4f},{res['ci_high']:.4f}] "
                f"p_old={prev_p} p_clust={res['p_clustered']:.4f} n={res['n']} n_clust={res['n_clusters']}"
            )
            rows.append(
                _row(
                    analysis_id=f"N5_{fam}_{model}",
                    statistic="spearman_contam_vs_retention",
                    estimate=res["estimate"],
                    ci_low=res["ci_low"],
                    ci_high=res["ci_high"],
                    p_clustered=res["p_clustered"],
                    n=res["n"],
                    n_clusters=res["n_clusters"],
                    clustering_variable=clust,
                    previously_reported_p=prev_p,
                    null_test="cluster_bootstrap_H0_rho=0",
                )
            )
            out_rows.append(
                {
                    "family": fam,
                    "model": model,
                    "n": res["n"],
                    "n_clusters": res["n_clusters"],
                    "spearman_rho": _round(res["estimate"]),
                    "ci_low": _round(res["ci_low"]),
                    "ci_high": _round(res["ci_high"]),
                    "p_value": _round(res["p_clustered"]),
                    "p_value_method": "cluster_bootstrap_two_sided",
                    "subset": "canonical_correct_only",
                    "y": "w3_retained",
                    "contamination_column": "contamination_score",
                    "bootstrap": f"cluster_by_{clust}",
                    "n_boot": N_BOOT,
                    "seed": SEED,
                    "note": "",
                }
            )
    pd.DataFrame(out_rows).to_csv(DER / "N5_contamination_vs_retention.csv", index=False)


def main() -> None:
    rows: list[dict] = []
    audit_m1(rows)
    audit_m2_n1(rows)
    audit_m3_construct(rows)
    audit_n2(rows)
    audit_n3(rows)
    audit_n4(rows)
    audit_n5(rows)

    out = pd.DataFrame(rows)
    # column order per spec + extras for transparency
    cols = [
        "analysis_id", "statistic", "estimate", "ci_low", "ci_high",
        "p_clustered", "n", "n_clusters", "clustering_variable",
        "previously_reported_p", "changed_conclusion", "null_test", "notes",
    ]
    out = out[cols]
    out.to_csv(OUT, index=False)

    changed = out[out["changed_conclusion"] == True]  # noqa: E712
    print(f"\n=== Wrote {OUT} ({len(out)} rows) ===")
    print(f"Rows where clustering changed significance verdict (alpha={ALPHA}): {len(changed)}")
    if len(changed):
        print(changed[["analysis_id", "estimate", "ci_low", "ci_high", "previously_reported_p", "p_clustered"]].to_string(index=False))


if __name__ == "__main__":
    main()
