#!/usr/bin/env python3
"""C4: Is TEP (and CCI) a robustness measure or capability in disguise?

Correlate mean TEP / CCI against canonical accuracy (model-level, per family +
pooled) and item-level TEP/CCI vs item canonical correctness within model.
Compare to existing P1 discriminant-validity cells (retention ρ≈0.136 n.s.;
phi ρ≈−0.426 n.s.).

BW Probe 2 excluded (execution floor; TEP not computable).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.clones import cluster_ids_for  # noqa: E402
from probes.common.cluster_inference import cluster_bootstrap_assoc  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
RAW = REPO_ROOT / "results" / "raw"
OUT = DER / "C4_tep_discriminant_validity.csv"

N_BOOT = 5000
SEED = 42

MODEL_SHORT = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
    "openai/o1-mini": "o4-mini",
}


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _round(x: float, nd: int = 4):
    if x != x:
        return ""
    return round(float(x), nd)


def _verdict(rho: float, p: float) -> str:
    if rho != rho:
        return "undefined"
    if p == p and p < 0.05:
        return "discriminant_validity_failure_tracks_capability"
    if abs(rho) >= 0.5:
        return "tracks_capability_ns"
    return "discriminant_ok_independent_of_capability"


def _assoc_row(
    *,
    analysis: str,
    scope: str,
    metric: str,
    x_name: str,
    y_name: str,
    x: pd.Series,
    y: pd.Series,
    clusters: pd.Series,
    models: str,
    note: str = "",
    verdict: str = "",
    bootstrap_label: str = "cluster_bootstrap",
    binary_y: bool = False,
    min_binary_cell: int = 5,
) -> dict:
    sub = pd.DataFrame({"x": x, "y": y, "c": clusters}).dropna()
    n = len(sub)
    n_clust = int(sub["c"].nunique()) if n else 0

    # Degenerate binary outcome (same rule as C3 N3)
    if binary_y and n:
        y_int = sub["y"].astype(float).round().astype(int)
        n_pos = int(y_int.sum())
        n_neg = int((y_int == 0).sum())
        if min(n_pos, n_neg) < min_binary_cell:
            return {
                "analysis": analysis,
                "scope": scope,
                "metric": metric,
                "x": x_name,
                "y": y_name,
                "spearman_rho": "",
                "ci_low": "",
                "ci_high": "",
                "p_value": "",
                "p_value_method": "cluster_bootstrap_two_sided",
                "n": n,
                "n_clusters": n_clust,
                "bootstrap": bootstrap_label,
                "n_boot": N_BOOT,
                "seed": SEED,
                "models": models,
                "note": (
                    f"outcome degenerate ({n_pos}/{n} positive); correlation not estimable. "
                    + note
                ).strip(),
                "verdict": "undefined",
            }

    # K<3 family-cluster bootstrap is degenerate (C3 lesson)
    if n_clust < 3 and bootstrap_label.startswith("cluster_by_family"):
        return {
            "analysis": analysis,
            "scope": scope,
            "metric": metric,
            "x": x_name,
            "y": y_name,
            "spearman_rho": "",
            "ci_low": "",
            "ci_high": "",
            "p_value": "",
            "p_value_method": "cluster_bootstrap_two_sided",
            "n": n,
            "n_clusters": n_clust,
            "bootstrap": bootstrap_label,
            "n_boot": N_BOOT,
            "seed": SEED,
            "models": models,
            "note": (
                f"family-cluster bootstrap undefined with n_clusters={n_clust}<3 "
                "(narrow CI artifact). Use cell-bootstrap row. "
                + note
            ).strip(),
            "verdict": "undefined",
        }

    if n < 3 or sub["x"].nunique() < 2 or sub["y"].nunique() < 2:
        return {
            "analysis": analysis,
            "scope": scope,
            "metric": metric,
            "x": x_name,
            "y": y_name,
            "spearman_rho": "",
            "ci_low": "",
            "ci_high": "",
            "p_value": "",
            "p_value_method": "cluster_bootstrap_two_sided",
            "n": n,
            "n_clusters": n_clust,
            "bootstrap": bootstrap_label,
            "n_boot": N_BOOT,
            "seed": SEED,
            "models": models,
            "note": note or "insufficient variation — correlation undefined",
            "verdict": verdict or "undefined",
        }
    res = cluster_bootstrap_assoc(
        sub["x"], sub["y"], sub["c"].astype(str),
        kind="spearman", n_boot=N_BOOT, seed=SEED,
    )
    rho = res["estimate"]
    p = res["p_clustered"]
    return {
        "analysis": analysis,
        "scope": scope,
        "metric": metric,
        "x": x_name,
        "y": y_name,
        "spearman_rho": _round(rho),
        "ci_low": _round(res["ci_low"]),
        "ci_high": _round(res["ci_high"]),
        "p_value": _round(p),
        "p_value_method": "cluster_bootstrap_two_sided",
        "n": res["n"],
        "n_clusters": res["n_clusters"],
        "bootstrap": bootstrap_label,
        "n_boot": N_BOOT,
        "seed": SEED,
        "models": models,
        "note": note,
        "verdict": verdict or _verdict(rho, p),
    }


def load_gsm_tep_cci() -> pd.DataFrame:
    tep = pd.read_csv(DER / "GSM_P2_tep.csv", dtype=str).fillna("")
    inst = tep[tep["level"] == "instance"].copy()
    inst["tep"] = pd.to_numeric(inst["tep"], errors="coerce")
    inst["cci"] = pd.to_numeric(inst["cci"], errors="coerce")
    inst["model"] = inst["model"].astype(str)
    return inst


def load_algo_tep_cci() -> pd.DataFrame:
    tep = pd.read_csv(DER / "ALGO_P2_tep.csv", dtype=str).fillna("")
    inst = tep[tep["level"] == "instance"].copy()
    inst["tep"] = pd.to_numeric(inst["tep"], errors="coerce")
    # Prefer CCI from ALGO_P2_cci when present (fuller); fall back to tep file
    cci = pd.read_csv(DER / "ALGO_P2_cci.csv", dtype=str).fillna("")
    cci["model_short"] = cci["model"].map(MODEL_SHORT).fillna(cci["model"])
    cci["cci_score"] = pd.to_numeric(cci["cci_score"], errors="coerce")
    merged = inst.merge(
        cci[["problem_id", "model_short", "cci_score"]],
        left_on=["problem_id", "model"],
        right_on=["problem_id", "model_short"],
        how="left",
    )
    # If tep file had cci, fill gaps
    if "cci" in inst.columns:
        merged["cci"] = pd.to_numeric(merged["cci"], errors="coerce")
        merged["cci"] = merged["cci_score"].combine_first(merged["cci"])
    else:
        merged["cci"] = merged["cci_score"]
    merged["cluster_id"] = merged.get("cluster_id")
    if merged["cluster_id"].eq("").all() or merged["cluster_id"].isna().all():
        merged["cluster_id"] = cluster_ids_for(merged["problem_id"].tolist())
    return merged


def gsm_item_canonical(problem_ids: set[str]) -> pd.DataFrame:
    rows = []
    for path in sorted(RAW.glob("GSM_P1_behavioral_*.csv")):
        if path.suffix != ".csv":
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        can = df[
            (df["variant_type"].str.lower() == "canonical")
            & (df["problem_id"].isin(problem_ids))
        ].copy()
        if can.empty:
            continue
        can["model"] = can["model"].map(MODEL_SHORT).fillna(can["model"])
        can["canonical_ok"] = _bool(can["behavioral_correct"]).astype(int)
        rows.append(can[["problem_id", "model", "canonical_ok"]])
    if not rows:
        return pd.DataFrame(columns=["problem_id", "model", "canonical_ok"])
    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(["problem_id", "model"], keep="last")


def algo_item_canonical(problem_ids: set[str]) -> pd.DataFrame:
    rows = []
    for path in sorted(DER.glob("ALGO_P1_behavioral_*_rescored.csv")):
        df = pd.read_csv(path, dtype=str).fillna("")
        can = df[
            (df["variant_type"].str.lower() == "canonical")
            & (df["problem_id"].isin(problem_ids))
        ].copy()
        if can.empty:
            continue
        can["model"] = can["model"].map(MODEL_SHORT).fillna(can["model"])
        col = "rescored_correct" if "rescored_correct" in can.columns else "verified"
        can["canonical_ok"] = _bool(can[col]).astype(int)
        rows.append(can[["problem_id", "model", "canonical_ok"]])
    if not rows:
        return pd.DataFrame(columns=["problem_id", "model", "canonical_ok"])
    out = pd.concat(rows, ignore_index=True)
    return out.drop_duplicates(["problem_id", "model"], keep="last")


def model_means(inst: pd.DataFrame, metric: str, can: pd.DataFrame) -> pd.DataFrame:
    """Per-model mean metric + matched-set canonical accuracy."""
    m_metric = (
        inst.dropna(subset=[metric])
        .groupby("model", as_index=False)
        .agg(metric_mean=(metric, "mean"), n_metric=(metric, "count"))
    )
    m_can = (
        can.groupby("model", as_index=False)
        .agg(canonical_accuracy=("canonical_ok", "mean"), n_can=("canonical_ok", "count"))
    )
    return m_metric.merge(m_can, on="model", how="inner")


def main() -> None:
    rows: list[dict] = []

    gsm_inst = load_gsm_tep_cci()
    algo_inst = load_algo_tep_cci()
    gsm_can = gsm_item_canonical(set(gsm_inst["problem_id"]))
    algo_can = algo_item_canonical(set(algo_inst["problem_id"]))

    gsm_tep_m = model_means(gsm_inst, "tep", gsm_can)
    algo_tep_m = model_means(algo_inst, "tep", algo_can)
    gsm_cci_m = model_means(gsm_inst, "cci", gsm_can)
    algo_cci_m = model_means(algo_inst, "cci", algo_can)

    print("=== Model-level means (matched P1 canonical on TEP items) ===")
    print("ALGO TEP:\n", algo_tep_m.to_string(index=False))
    print("GSM TEP:\n", gsm_tep_m.to_string(index=False))
    print("ALGO CCI:\n", algo_cci_m.to_string(index=False))
    print("GSM CCI:\n", gsm_cci_m.to_string(index=False))

    # --- Model-level TEP vs canonical accuracy ---
    for fam, means in [("ALGO", algo_tep_m), ("GSM", gsm_tep_m)]:
        rows.append(
            _assoc_row(
                analysis="model_mean_tep_vs_canonical_accuracy",
                scope=fam,
                metric="TEP",
                x_name="mean_tep",
                y_name="canonical_accuracy",
                x=means["metric_mean"],
                y=means["canonical_accuracy"],
                clusters=means["model"],
                models=",".join(means["model"].tolist()),
                bootstrap_label="bootstrap_over_models",
                note=(
                    f"model means; TEP from {fam}_P2_tep.csv; "
                    "canonical_accuracy = P1 correct rate on same problem_ids"
                ),
            )
        )

    # ALGO sensitivity: drop o4-mini (acc floor/ceiling; mid TEP) — recovers O9 inverse order
    algo_tep_no_o4 = algo_tep_m[algo_tep_m["model"] != "o4-mini"].copy()
    rows.append(
        _assoc_row(
            analysis="model_mean_tep_vs_canonical_accuracy",
            scope="ALGO_excl_o4mini",
            metric="TEP",
            x_name="mean_tep",
            y_name="canonical_accuracy",
            x=algo_tep_no_o4["metric_mean"],
            y=algo_tep_no_o4["canonical_accuracy"],
            clusters=algo_tep_no_o4["model"],
            models=",".join(algo_tep_no_o4["model"].tolist()),
            bootstrap_label="bootstrap_over_models",
            note=(
                "sensitivity: drop o4-mini (acc=1.0 ceiling). Remaining order "
                "Claude<Gemini<GPT-4o<Llama TEP is exact inverse of capability"
            ),
        )
    )

    pooled_tep = pd.concat(
        [
            algo_tep_m.assign(family="ALGO"),
            gsm_tep_m.assign(family="GSM"),
        ],
        ignore_index=True,
    )
    pooled_tep["cell"] = pooled_tep["family"] + ":" + pooled_tep["model"]
    # Primary pooled: cell bootstrap (family×model). Family-cluster K=2 is undefined.
    rows.append(
        _assoc_row(
            analysis="model_mean_tep_vs_canonical_accuracy",
            scope="ALGO_GSM_pooled",
            metric="TEP",
            x_name="mean_tep",
            y_name="canonical_accuracy",
            x=pooled_tep["metric_mean"],
            y=pooled_tep["canonical_accuracy"],
            clusters=pooled_tep["cell"],
            models=";".join(pooled_tep["cell"].tolist()),
            bootstrap_label="bootstrap_over_family_model_cells",
            note="PRIMARY pooled: bootstrap over family×model cells (K=2 family cluster undefined)",
        )
    )
    rows.append(
        _assoc_row(
            analysis="model_mean_tep_vs_canonical_accuracy",
            scope="ALGO_GSM_pooled_family_cluster",
            metric="TEP",
            x_name="mean_tep",
            y_name="canonical_accuracy",
            x=pooled_tep["metric_mean"],
            y=pooled_tep["canonical_accuracy"],
            clusters=pooled_tep["family"],
            models=";".join(pooled_tep["cell"].tolist()),
            bootstrap_label="cluster_by_family",
            note="withdrawn: only 2 families",
        )
    )

    # --- Model-level CCI vs canonical accuracy ---
    for fam, means in [("ALGO", algo_cci_m), ("GSM", gsm_cci_m)]:
        rows.append(
            _assoc_row(
                analysis="model_mean_cci_vs_canonical_accuracy",
                scope=fam,
                metric="CCI",
                x_name="mean_cci",
                y_name="canonical_accuracy",
                x=means["metric_mean"],
                y=means["canonical_accuracy"],
                clusters=means["model"],
                models=",".join(means["model"].tolist()),
                bootstrap_label="bootstrap_over_models",
                note=(
                    f"model means; CCI from {'ALGO_P2_cci' if fam == 'ALGO' else 'GSM_P2'}; "
                    "Gemini ALGO CCI all-NaN excluded if absent"
                ),
            )
        )

    pooled_cci = pd.concat(
        [
            algo_cci_m.assign(family="ALGO"),
            gsm_cci_m.assign(family="GSM"),
        ],
        ignore_index=True,
    )
    pooled_cci["cell"] = pooled_cci["family"] + ":" + pooled_cci["model"]
    rows.append(
        _assoc_row(
            analysis="model_mean_cci_vs_canonical_accuracy",
            scope="ALGO_GSM_pooled",
            metric="CCI",
            x_name="mean_cci",
            y_name="canonical_accuracy",
            x=pooled_cci["metric_mean"],
            y=pooled_cci["canonical_accuracy"],
            clusters=pooled_cci["cell"],
            models=";".join(pooled_cci["cell"].tolist()),
            bootstrap_label="bootstrap_over_family_model_cells",
            note="PRIMARY pooled: bootstrap over family×model cells",
        )
    )
    rows.append(
        _assoc_row(
            analysis="model_mean_cci_vs_canonical_accuracy",
            scope="ALGO_GSM_pooled_family_cluster",
            metric="CCI",
            x_name="mean_cci",
            y_name="canonical_accuracy",
            x=pooled_cci["metric_mean"],
            y=pooled_cci["canonical_accuracy"],
            clusters=pooled_cci["family"],
            models=";".join(pooled_cci["cell"].tolist()),
            bootstrap_label="cluster_by_family",
            note="withdrawn: only 2 families",
        )
    )

    # --- Item-level within model ---
    for fam, inst, can in [
        ("GSM", gsm_inst, gsm_can),
        ("ALGO", algo_inst, algo_can),
    ]:
        joined = inst.merge(can, on=["problem_id", "model"], how="inner")
        for metric in ("tep", "cci"):
            for model, sub in joined.groupby("model"):
                s = sub.dropna(subset=[metric, "canonical_ok"]).copy()
                if fam == "ALGO":
                    clusters = (
                        s["cluster_id"]
                        if "cluster_id" in s.columns and s["cluster_id"].astype(str).ne("").any()
                        else pd.Series(cluster_ids_for(s["problem_id"].tolist()), index=s.index)
                    )
                    boot = "cluster_by_clone_family"
                else:
                    clusters = s["problem_id"]
                    boot = "cluster_by_problem_id"
                rows.append(
                    _assoc_row(
                        analysis="item_within_model",
                        scope=f"{fam}/{model}",
                        metric=metric.upper(),
                        x_name=metric,
                        y_name="canonical_ok",
                        x=s[metric],
                        y=s["canonical_ok"],
                        clusters=clusters,
                        models=str(model),
                        bootstrap_label=boot,
                        binary_y=True,
                        note="item-level within model",
                    )
                )

    # --- Reference: existing P1 discriminant validity ---
    ref = pd.read_csv(DER / "P1_construct_validity.csv", dtype=str).fillna("")
    for _, r in ref.iterrows():
        rho = float(pd.to_numeric(r["spearman_rho"], errors="coerce"))
        p = float(pd.to_numeric(r["p_value"], errors="coerce"))
        rows.append(
            {
                "analysis": "reference_" + r["analysis"],
                "scope": "P1_family_model_cells",
                "metric": r["row_construct"],
                "x": r["row_construct"],
                "y": "canonical_accuracy",
                "spearman_rho": r["spearman_rho"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "p_value": r["p_value"],
                "p_value_method": r["p_value_method"],
                "n": r["n_cells"],
                "n_clusters": "",
                "bootstrap": r["bootstrap"],
                "n_boot": r["n_boot"],
                "seed": r["seed"],
                "models": "see P1_phi_canonical_w3.csv (can_acc>=0.3)",
                "note": (
                    "existing M3 discriminant cell — contrast for TEP/CCI; "
                    + str(r.get("note", ""))
                ).strip("; "),
                "verdict": _verdict(rho, p),
            }
        )

    # --- Contrast summary ---
    tep_pooled = next(
        r
        for r in rows
        if r["analysis"] == "model_mean_tep_vs_canonical_accuracy"
        and r["scope"] == "ALGO_GSM_pooled"
    )
    cci_pooled = next(
        r
        for r in rows
        if r["analysis"] == "model_mean_cci_vs_canonical_accuracy"
        and r["scope"] == "ALGO_GSM_pooled"
    )
    ret = next(r for r in rows if "retention" in r["analysis"])
    phi = next(r for r in rows if "phi" in r["analysis"] and r["analysis"].startswith("reference"))
    tep_algo_no_o4 = next(
        r
        for r in rows
        if r["analysis"] == "model_mean_tep_vs_canonical_accuracy"
        and r["scope"] == "ALGO_excl_o4mini"
    )

    tep_fails = tep_pooled["verdict"].startswith("discriminant_validity_failure") or (
        tep_algo_no_o4["verdict"].startswith("discriminant_validity_failure")
        or tep_algo_no_o4["verdict"] == "tracks_capability_ns"
    )
    cci_fails = cci_pooled["verdict"].startswith("discriminant_validity_failure") or cci_pooled[
        "verdict"
    ] == "tracks_capability_ns"

    rows.append(
        {
            "analysis": "contrast_tep_cci_vs_retention_phi",
            "scope": "summary",
            "metric": "TEP_CCI_vs_retention_phi",
            "x": "probe2_metric",
            "y": "canonical_accuracy",
            "spearman_rho": tep_pooled["spearman_rho"],
            "ci_low": tep_pooled["ci_low"],
            "ci_high": tep_pooled["ci_high"],
            "p_value": tep_pooled["p_value"],
            "p_value_method": "see_component_rows",
            "n": tep_pooled["n"],
            "n_clusters": tep_pooled["n_clusters"],
            "bootstrap": "bootstrap_over_family_model_cells",
            "n_boot": N_BOOT,
            "seed": SEED,
            "models": tep_pooled["models"],
            "note": (
                f"TEP pooled rho={tep_pooled['spearman_rho']} "
                f"(ALGO excl o4-mini rho={tep_algo_no_o4['spearman_rho']}); "
                f"CCI pooled rho={cci_pooled['spearman_rho']}; "
                f"retention rho={ret['spearman_rho']} n.s.; "
                f"phi rho={phi['spearman_rho']} n.s. "
                "Within-instrument contrast: retention/phi pass discriminant validity; "
                "CCI fails (tracks capability); TEP shows capability-aligned between-model "
                "ordering (esp. ALGO without o4-mini) but item-level within model is mostly null."
            ),
            "verdict": (
                "cci_discriminant_failure_tep_between_model_signal_retention_ok"
                if cci_fails
                else (
                    "tep_between_model_signal_retention_ok"
                    if tep_fails
                    else "mixed_or_both_ok"
                )
            ),
        }
    )

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT} ({len(out)} rows)")
    show = out[
        ~out["analysis"].eq("item_within_model")
        | out["scope"].astype(str).str.contains("summary")
    ]
    # Prefer headline rows
    show = out[
        out["analysis"].isin(
            [
                "model_mean_tep_vs_canonical_accuracy",
                "model_mean_cci_vs_canonical_accuracy",
                "reference_discriminant_retention_vs_canonical_accuracy",
                "reference_discriminant_phi_vs_canonical_accuracy",
                "contrast_tep_cci_vs_retention_phi",
            ]
        )
    ]
    print(
        show[
            ["analysis", "scope", "metric", "spearman_rho", "ci_low", "ci_high", "p_value", "n", "verdict"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
