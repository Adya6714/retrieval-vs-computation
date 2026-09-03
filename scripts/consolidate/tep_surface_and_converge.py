#!/usr/bin/env python3
"""O9: Surface GSM TEP, compute ALGO TEP, document BW P2 floor, test convergence.

TEP = fraction of post-injection steps whose content differs from the uninjected run
(EF-05; GSM `_compute_tep`; ALGO join of normal vs injected parsed decisions).

Does NOT compute BW TEP — BW Probe 2 is a measurement failure (execution floor).
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

from probes.algo.decision_normalize import normalize_phase2_decision  # noqa: E402
from probes.common.clones import cluster_ids_for  # noqa: E402
from probes.common.cluster_inference import (  # noqa: E402
    cluster_bootstrap_assoc,
)
from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
RAW = REPO_ROOT / "results" / "raw"

GSM_CCI = RAW / "GSM_P2_cci.csv"
ALGO_NORMAL = RAW / "ALGO_P2_phase2_normal.csv"
ALGO_NORMAL_GEM = RAW / "ALGO_P2_phase2_normal_gemini.csv"
ALGO_INJ = RAW / "ALGO_P2_phase2_injected.csv"
ALGO_INJ_GEM = RAW / "ALGO_P2_phase2_injected_gemini.csv"
BW_CCI = RAW / "BW_P2_cci.csv"

OUT_GSM = DER / "GSM_P2_tep.csv"
OUT_ALGO = DER / "ALGO_P2_tep.csv"
OUT_ALGO_STEPS = DER / "ALGO_P2_tep_steps.csv"
OUT_ALGO_SESS = DER / "ALGO_P2_tep_sessions.csv"
OUT_CONV = DER / "TEP_P1_convergence.csv"
OUT_TEP_CCI = DER / "TEP_CCI_correlation.csv"
OUT_BW_FLOOR = DER / "BW_P2_floor_documentation.csv"
OUT_FAIL = DER / "MEASUREMENT_FAILURES.csv"

PAPER_MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
PARSEABLE = {"compliant", "partial", "format_ignored"}
N_BOOT = 5000
SEED = 42


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _short(s: pd.Series) -> pd.Series:
    return s.map(PAPER_MODELS).fillna(s)


def _cluster_boot_mean(vals: np.ndarray, cluster_ids: list[str]) -> tuple[float, float, float]:
    vals = np.asarray(vals, dtype=float)
    clusters = sorted(set(cluster_ids))
    grouped = {c: [i for i, cid in enumerate(cluster_ids) if cid == c] for c in clusters}
    est = float(np.nanmean(vals))
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT, dtype=float)
    for i in range(N_BOOT):
        draw = rng.choice(clusters, size=len(clusters), replace=True)
        idx = [j for c in draw for j in grouped[c]]
        boots[i] = float(np.nanmean(vals[idx])) if idx else float("nan")
    finite = boots[np.isfinite(boots)]
    if len(finite) == 0:
        return est, float("nan"), float("nan")
    return est, float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


# ---------------------------------------------------------------------------
# 1. GSM TEP surface
# ---------------------------------------------------------------------------

def surface_gsm_tep() -> pd.DataFrame:
    df = pd.read_csv(GSM_CCI, dtype=str).fillna("")
    df["model_short"] = _short(df["model"])
    df["tep"] = pd.to_numeric(df["tep_score"], errors="coerce")
    df["cci"] = pd.to_numeric(df["cci_score"], errors="coerce")
    df["w3_proxy_session_b_correct"] = _is_true(df["session_b_correct"])
    df = df[df["tep"].notna()].copy()
    df["cluster_id"] = df["problem_id"].astype(str)

    rows = []
    for model, sub in df.groupby("model_short"):
        est, lo, hi = _cluster_boot_mean(
            sub["tep"].to_numpy(), sub["cluster_id"].astype(str).tolist(),
        )
        rows.append(
            {
                "family": "GSM",
                "model": model,
                "n": len(sub),
                "n_clusters": sub["cluster_id"].nunique(),
                "tep_mean": round(est, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "tep_median": round(float(sub["tep"].median()), 4),
                "bootstrap": "cluster_by_problem_id",
                "n_boot": N_BOOT,
                "seed": SEED,
                "source": "results/raw/GSM_P2_cci.csv",
            }
        )
    summary = pd.DataFrame(rows)
    # instance-level for convergence joins
    inst = df[
        [
            "problem_id",
            "model",
            "model_short",
            "tep",
            "cci",
            "inject_at_step",
            "injected_value",
            "session_b_correct",
            "contamination_pole",
            "difficulty",
            "cluster_id",
        ]
    ].copy()
    inst["family"] = "GSM"
    # write summary + attach instance rows below with level column
    out_rows = []
    for _, r in summary.iterrows():
        out_rows.append({**r.to_dict(), "level": "model_summary"})
    for _, r in inst.iterrows():
        out_rows.append(
            {
                "family": "GSM",
                "level": "instance",
                "model": r["model_short"],
                "model_full": r["model"],
                "problem_id": r["problem_id"],
                "n": "",
                "n_clusters": "",
                "tep_mean": "",
                "tep": r["tep"],
                "cci": r["cci"],
                "ci_low": "",
                "ci_high": "",
                "tep_median": "",
                "inject_at_step": r["inject_at_step"],
                "cluster_id": r["cluster_id"],
                "bootstrap": "",
                "n_boot": "",
                "seed": "",
                "source": "results/raw/GSM_P2_cci.csv",
            }
        )
    out = pd.DataFrame(out_rows)
    out.to_csv(OUT_GSM, index=False)
    print(f"Wrote {OUT_GSM} ({len(summary)} model rows, {len(inst)} instances)")
    print(summary.to_string(index=False))
    return inst


# ---------------------------------------------------------------------------
# 2. ALGO TEP
# ---------------------------------------------------------------------------

def _load_algo_normal() -> pd.DataFrame:
    parts = []
    if ALGO_NORMAL.exists():
        df = pd.read_csv(ALGO_NORMAL, dtype=str).fillna("")
        df = df[_short(df["model"]) != "Gemini"]
        parts.append(df)
    if ALGO_NORMAL_GEM.exists():
        parts.append(pd.read_csv(ALGO_NORMAL_GEM, dtype=str).fillna(""))
    out = pd.concat(parts, ignore_index=True)
    out["step_index_int"] = pd.to_numeric(out["step_index"], errors="coerce").astype("Int64")
    out["response_type"] = out["response_type"].str.strip().str.lower()
    return out


def _load_algo_injected() -> pd.DataFrame:
    """Plausible injection only (exclude implausible condition)."""
    parts = []
    if ALGO_INJ.exists():
        df = pd.read_csv(ALGO_INJ, dtype=str).fillna("")
        # Prefer dedicated Gemini rerun when present
        if ALGO_INJ_GEM.exists():
            df = df[_short(df["model"]) != "Gemini"]
        parts.append(df)
    if ALGO_INJ_GEM.exists():
        parts.append(pd.read_csv(ALGO_INJ_GEM, dtype=str).fillna(""))
    out = pd.concat(parts, ignore_index=True)
    out["step_index_int"] = pd.to_numeric(out["step_index"], errors="coerce").astype("Int64")
    out["critical_step_index_int"] = pd.to_numeric(out["critical_step_index"], errors="coerce")
    out["response_type"] = out["response_type"].str.strip().str.lower()
    return out


def compute_algo_tep() -> pd.DataFrame:
    normal = _load_algo_normal()
    injected = _load_algo_injected()
    print(f"ALGO normal rows={len(normal)} injected rows={len(injected)}")

    step_rows: list[dict] = []
    sess_rows: list[dict] = []

    keys = ["problem_id", "model", "subtype", "instance_type"]
    for (pid, model, subtype, inst), gn in normal.groupby(keys):
        gi = injected[
            (injected["problem_id"] == pid)
            & (injected["model"] == model)
            & (injected["instance_type"] == inst)
        ]
        if gi.empty:
            continue
        crit_vals = gi["critical_step_index_int"].dropna()
        if crit_vals.empty:
            continue
        crit = int(crit_vals.iloc[0])
        if crit < 0:
            continue

        merged = gn.merge(
            gi[
                [
                    "step_index_int",
                    "parsed_decision",
                    "response_type",
                    "injection_applied",
                    "true_state",
                    "injected_state",
                    "diverged_from_normal",
                    "post_injection_correct",
                ]
            ],
            on="step_index_int",
            how="inner",
            suffixes=("_n", "_i"),
        )
        # post-injection: steps strictly after critical (Appendix D / audit)
        post = merged[merged["step_index_int"] > crit].copy()
        if post.empty:
            continue

        used_fallback = False
        compliant = post[
            (post["response_type_n"] == "compliant") & (post["response_type_i"] == "compliant")
        ]
        compare = compliant
        if compare.empty:
            compare = post[
                post["response_type_n"].isin(PARSEABLE) & post["response_type_i"].isin(PARSEABLE)
            ]
            used_fallback = True
        if compare.empty:
            continue

        diverged_flags = []
        for _, r in compare.iterrows():
            dn = normalize_phase2_decision(subtype, str(r["parsed_decision_n"]))
            di = normalize_phase2_decision(subtype, str(r["parsed_decision_i"]))
            diverged = dn != di
            diverged_flags.append(diverged)
            step_rows.append(
                {
                    "problem_id": pid,
                    "model": model,
                    "model_short": PAPER_MODELS.get(model, model),
                    "subtype": subtype,
                    "instance_type": inst,
                    "step_index": int(r["step_index_int"]),
                    "critical_step_index": crit,
                    "parsed_decision_normal": r["parsed_decision_n"],
                    "parsed_decision_injected": r["parsed_decision_i"],
                    "normalized_normal": dn,
                    "normalized_injected": di,
                    "diverged": bool(diverged),
                    "response_type_normal": r["response_type_n"],
                    "response_type_injected": r["response_type_i"],
                    "comparison_mode": "parseable_fallback" if used_fallback else "strict_compliant",
                }
            )

        tep = float(np.mean(diverged_flags)) if diverged_flags else float("nan")
        # final correctness from normal run last step
        gn_sorted = gn.sort_values("step_index_int")
        final_ok = bool(_is_true(gn_sorted["final_answer_correct"]).iloc[-1]) if "final_answer_correct" in gn_sorted.columns else False
        # post-injection correctness if available
        pic = gi["post_injection_correct"]
        pic_vals = pic[pic.astype(str).str.strip() != ""]
        post_inj_ok = ""
        if len(pic_vals):
            post_inj_ok = bool(_is_true(pic_vals).iloc[-1])

        sess_rows.append(
            {
                "family": "ALGO",
                "problem_id": pid,
                "model": model,
                "model_short": PAPER_MODELS.get(model, model),
                "subtype": subtype,
                "instance_type": inst,
                "critical_step_index": crit,
                "tep": round(tep, 4) if tep == tep else "",
                "n_post_steps_compared": len(diverged_flags),
                "n_diverged": int(sum(diverged_flags)),
                "comparison_mode": "parseable_fallback" if used_fallback else "strict_compliant",
                "final_answer_correct_normal": final_ok,
                "post_injection_correct": post_inj_ok,
                "n_injected_raw_rows": len(gi),
                "n_normal_raw_rows": len(gn),
            }
        )

    steps = pd.DataFrame(step_rows)
    sessions = pd.DataFrame(sess_rows)
    steps.to_csv(OUT_ALGO_STEPS, index=False)
    sessions.to_csv(OUT_ALGO_SESS, index=False)
    print(f"Wrote {OUT_ALGO_STEPS} ({len(steps)} rows)")
    print(f"Wrote {OUT_ALGO_SESS} ({len(sessions)} rows)")

    # model summary with cluster bootstrap
    sessions = sessions[sessions["tep"].astype(str).str.len() > 0].copy()
    sessions["tep"] = pd.to_numeric(sessions["tep"], errors="coerce")
    sessions["cluster_id"] = cluster_ids_for(sessions["problem_id"].astype(str).tolist())

    rows = []
    for model, sub in sessions.groupby("model_short"):
        est, lo, hi = _cluster_boot_mean(
            sub["tep"].to_numpy(), sub["cluster_id"].astype(str).tolist(),
        )
        rows.append(
            {
                "family": "ALGO",
                "level": "model_summary",
                "model": model,
                "n": len(sub),
                "n_clusters": sub["cluster_id"].nunique(),
                "tep_mean": round(est, 4),
                "ci_low": round(lo, 4),
                "ci_high": round(hi, 4),
                "tep_median": round(float(sub["tep"].median()), 4),
                "bootstrap": "cluster_by_clone_family",
                "n_boot": N_BOOT,
                "seed": SEED,
                "source": "ALGO_P2_phase2_normal*.csv × ALGO_P2_phase2_injected*.csv",
            }
        )

    out_rows = list(rows)
    for _, r in sessions.iterrows():
        out_rows.append(
            {
                "family": "ALGO",
                "level": "instance",
                "model": r["model_short"],
                "model_full": r["model"],
                "problem_id": r["problem_id"],
                "subtype": r["subtype"],
                "instance_type": r["instance_type"],
                "tep": r["tep"],
                "cci": "",
                "n": "",
                "n_clusters": "",
                "tep_mean": "",
                "ci_low": "",
                "ci_high": "",
                "tep_median": "",
                "critical_step_index": r["critical_step_index"],
                "n_post_steps_compared": r["n_post_steps_compared"],
                "comparison_mode": r["comparison_mode"],
                "cluster_id": r["cluster_id"],
                "bootstrap": "",
                "n_boot": "",
                "seed": "",
                "source": "normal×injected join",
            }
        )
    # attach CCI from ALGO_P2_cci when available
    cci_path = DER / "ALGO_P2_cci.csv"
    if cci_path.exists():
        cci = pd.read_csv(cci_path, dtype=str).fillna("")
        cci["model_short"] = _short(cci["model"])
        cci["cci"] = pd.to_numeric(cci["cci_score"], errors="coerce")
        cci_map = {
            (str(r["problem_id"]), r["model_short"]): r["cci"]
            for _, r in cci.iterrows()
        }
        for row in out_rows:
            if row.get("level") == "instance":
                row["cci"] = cci_map.get((str(row["problem_id"]), row["model"]), "")

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT_ALGO, index=False)
    print(f"Wrote {OUT_ALGO}")
    print(pd.DataFrame(rows).to_string(index=False))
    return sessions


# ---------------------------------------------------------------------------
# 3. BW floor (no TEP)
# ---------------------------------------------------------------------------

def document_bw_floor() -> None:
    cci = pd.read_csv(BW_CCI, dtype=str).fillna("")
    cci["model_short"] = _short(cci["model"])
    cci = cci[cci["model_short"].isin(PAPER_MODELS.values())].copy()
    cci["goal_reached"] = _is_true(cci["goal_reached"])
    cci["partial_goal"] = pd.to_numeric(cci["partial_goal_achievement"], errors="coerce")
    cci["semantic_validity"] = pd.to_numeric(cci["semantic_validity_rate"], errors="coerce")
    cci["cci"] = pd.to_numeric(cci["cci"], errors="coerce")

    rows = []
    for model, sub in cci.groupby("model_short"):
        rows.append(
            {
                "family": "BW",
                "model": model,
                "n_sessions": len(sub),
                "goal_reached_rate": round(float(sub["goal_reached"].mean()), 4),
                "partial_goal_achievement_mean": round(float(sub["partial_goal"].mean()), 4),
                "partial_goal_achievement_min": round(float(sub["partial_goal"].min()), 4),
                "partial_goal_achievement_max": round(float(sub["partial_goal"].max()), 4),
                "semantic_validity_rate_mean": round(float(sub["semantic_validity"].mean()), 4),
                "cci_mean": round(float(sub["cci"].mean()), 4),
                "tep_computed": False,
                "verdict": "measurement_failure_execution_floor",
                "note": (
                    "BW Probe 2 sessions are abort/format-error dominated; "
                    "goal_reached≈0; partial_goal in ~0.01–0.14 range. "
                    "Do not analyze TEP or treat CCI as a model fingerprint."
                ),
                "source": "results/raw/BW_P2_cci.csv",
            }
        )
    floor = pd.DataFrame(rows)
    floor.to_csv(OUT_BW_FLOOR, index=False)
    print(f"Wrote {OUT_BW_FLOOR}")
    print(floor[["model", "goal_reached_rate", "partial_goal_achievement_mean", "cci_mean"]].to_string(index=False))

    # measurement-failure table
    fail_row = {
        "probe": "P2",
        "family": "BW",
        "metric": "CCI/TEP",
        "status": "measurement_failure",
        "evidence": (
            f"goal_reached=0 for all models; "
            f"partial_goal_mean="
            + ",".join(f"{r['model']}={r['partial_goal_achievement_mean']}" for _, r in floor.iterrows())
        ),
        "action": "do_not_compute_TEP; report floor only",
        "source_doc": "BW_P2_floor_documentation.csv",
    }
    if OUT_FAIL.exists():
        fail = pd.read_csv(OUT_FAIL, dtype=str).fillna("")
        fail = fail[~((fail["probe"] == "P2") & (fail["family"] == "BW"))]
        fail = pd.concat([fail, pd.DataFrame([fail_row])], ignore_index=True)
    else:
        fail = pd.DataFrame([fail_row])
    fail.to_csv(OUT_FAIL, index=False)
    print(f"Wrote/updated {OUT_FAIL}")


# ---------------------------------------------------------------------------
# 4–5. Convergence: TEP↔W3, TEP↔CCI
# ---------------------------------------------------------------------------

def _load_p1_w3(family: str) -> pd.DataFrame:
    parts = []
    for path in sorted(DER.glob(f"{family}_P1_*rescored.csv")):
        if "review" in path.name.lower():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "included" not in df.columns:
            continue
        df = df[_is_true(df["included"])].copy()
        df = filter_excluded(df, family=family)
        df["variant"] = df["variant_type"].map(normalize_variant)
        ok = df["rescored_correct"] if "rescored_correct" in df.columns else df.get("verified", "")
        df["ok"] = _is_true(ok)
        df["model_short"] = _short(df["model"])
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    p1 = pd.concat(parts, ignore_index=True).drop_duplicates(
        ["problem_id", "variant", "model_short"], keep="last",
    )
    can = p1[p1["variant"] == "canonical"][["problem_id", "model_short", "ok"]].rename(
        columns={"ok": "canonical_ok"},
    )
    w3 = p1[p1["variant"] == "W3"][["problem_id", "model_short", "ok"]].rename(
        columns={"ok": "w3_ok"},
    )
    return can.merge(w3, on=["problem_id", "model_short"], how="inner")


def run_convergence(gsm_inst: pd.DataFrame, algo_sess: pd.DataFrame) -> None:
    conv_rows = []
    corr_rows = []

    frames = []

    # GSM
    gsm_p1 = _load_p1_w3("GSM")
    gsm = gsm_inst.merge(gsm_p1, on=["problem_id", "model_short"], how="inner")
    gsm["family"] = "GSM"
    gsm["cluster_id"] = gsm["problem_id"].astype(str)
    gsm["w3_correct"] = gsm["w3_ok"].astype(int)
    frames.append(gsm)

    # ALGO
    algo_p1 = _load_p1_w3("ALGO")
    algo = algo_sess.copy()
    algo["tep"] = pd.to_numeric(algo["tep"], errors="coerce")
    algo = algo.merge(algo_p1, on=["problem_id", "model_short"], how="inner")
    algo["family"] = "ALGO"
    algo["w3_correct"] = algo["w3_ok"].astype(int)
    # CCI join
    cci_path = DER / "ALGO_P2_cci.csv"
    if cci_path.exists():
        cci = pd.read_csv(cci_path, dtype=str).fillna("")
        cci["model_short"] = _short(cci["model"])
        cci["cci"] = pd.to_numeric(cci["cci_score"], errors="coerce")
        algo = algo.merge(
            cci[["problem_id", "model_short", "cci"]],
            on=["problem_id", "model_short"],
            how="left",
        )
    else:
        algo["cci"] = np.nan
    frames.append(algo)

    def _pb_rows(frame: pd.DataFrame, scope: str, clust_var: str) -> None:
        for label, sub in [
            (f"{scope}_all_with_tep", frame),
            (f"{scope}_canonical_correct", frame[frame["canonical_ok"].astype(bool)]),
        ]:
            sub = sub[sub["tep"].notna()].copy()
            if len(sub) < 5 or sub["w3_correct"].nunique() < 2:
                continue
            res = cluster_bootstrap_assoc(
                sub["tep"], sub["w3_correct"], sub["cluster_id"].astype(str),
                kind="pointbiserial", n_boot=N_BOOT, seed=SEED,
            )
            conv_rows.append(
                {
                    "analysis": "pointbiserial_tep_w3_correct",
                    "scope": label,
                    "statistic": round(res["estimate"], 4),
                    "ci_low": round(res["ci_low"], 4) if res["ci_low"] == res["ci_low"] else "",
                    "ci_high": round(res["ci_high"], 4) if res["ci_high"] == res["ci_high"] else "",
                    "p_clustered": round(res["p_clustered"], 4) if res["p_clustered"] == res["p_clustered"] else "",
                    "n": res["n"],
                    "n_clusters": res["n_clusters"],
                    "clustering_variable": clust_var,
                    "note": "CI+p from same cluster bootstrap; CI primary",
                }
            )

    _pb_rows(gsm, "GSM", "problem_id")
    _pb_rows(algo, "ALGO", "clone_family")

    pooled = pd.concat(
        [
            gsm[["family", "problem_id", "model_short", "tep", "cci", "w3_correct", "canonical_ok", "cluster_id"]],
            algo[["family", "problem_id", "model_short", "tep", "cci", "w3_correct", "canonical_ok", "cluster_id"]],
        ],
        ignore_index=True,
    )
    pooled["cluster_id"] = pooled.apply(lambda r: f"{r['family']}:{r['cluster_id']}", axis=1)
    _pb_rows(pooled, "GSM_ALGO_pooled", "family:cluster")

    # TEP vs CCI
    for fam, frame, clust in [
        ("GSM", gsm, "problem_id"),
        ("ALGO", algo, "clone_family"),
        ("GSM_ALGO_pooled", pooled, "family:cluster"),
    ]:
        sub = frame[frame["tep"].notna() & frame["cci"].notna()].copy()
        if len(sub) < 5 or sub["tep"].nunique() < 2 or sub["cci"].nunique() < 2:
            corr_rows.append(
                {
                    "family": fam,
                    "statistic": "spearman_tep_cci",
                    "estimate": "",
                    "ci_low": "",
                    "ci_high": "",
                    "p_clustered": "",
                    "n": len(sub),
                    "n_clusters": "",
                    "clustering_variable": clust,
                    "note": "insufficient variation",
                }
            )
            continue
        res = cluster_bootstrap_assoc(
            sub["tep"], sub["cci"], sub["cluster_id"].astype(str),
            kind="spearman", n_boot=N_BOOT, seed=SEED,
        )
        xv = sub["tep"].astype(float).to_numpy()
        yv = sub["cci"].astype(float).to_numpy()
        cids = sub["cluster_id"].astype(str).tolist()
        pearson_r, _ = stats.pearsonr(xv, yv)
        clusters = sorted(set(cids))
        grouped = {c: [i for i, cid in enumerate(cids) if cid == c] for c in clusters}
        rng = np.random.default_rng(SEED)
        boots = np.empty(N_BOOT, dtype=float)
        for i in range(N_BOOT):
            draw = rng.choice(clusters, size=len(clusters), replace=True)
            idx = [j for c in draw for j in grouped[c]]
            if len(idx) < 5 or len(set(xv[idx])) < 2 or len(set(yv[idx])) < 2:
                boots[i] = float("nan")
            else:
                boots[i], _ = stats.pearsonr(xv[idx], yv[idx])
        finite = boots[np.isfinite(boots)]
        from probes.common.cluster_inference import bootstrap_p_two_sided

        corr_rows.append(
            {
                "family": fam,
                "statistic": "spearman_tep_cci",
                "estimate": round(res["estimate"], 4),
                "ci_low": round(res["ci_low"], 4),
                "ci_high": round(res["ci_high"], 4),
                "p_clustered": round(res["p_clustered"], 4),
                "n": res["n"],
                "n_clusters": res["n_clusters"],
                "clustering_variable": clust,
                "note": "disagreement with CCI would mean Probe 2 is multi-construct",
            }
        )
        corr_rows.append(
            {
                "family": fam,
                "statistic": "pearson_tep_cci",
                "estimate": round(float(pearson_r), 4),
                "ci_low": round(float(np.percentile(finite, 2.5)), 4) if len(finite) else "",
                "ci_high": round(float(np.percentile(finite, 97.5)), 4) if len(finite) else "",
                "p_clustered": round(bootstrap_p_two_sided(finite), 4) if len(finite) else "",
                "n": len(sub),
                "n_clusters": len(clusters),
                "clustering_variable": clust,
                "note": "disagreement with CCI would mean Probe 2 is multi-construct",
            }
        )

    conv = pd.DataFrame(conv_rows)
    conv.to_csv(OUT_CONV, index=False)
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(OUT_TEP_CCI, index=False)
    print(f"Wrote {OUT_CONV}")
    print(conv.to_string(index=False))
    print(f"Wrote {OUT_TEP_CCI}")
    print(corr.to_string(index=False))


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    print("=== 1. GSM TEP ===")
    gsm_inst = surface_gsm_tep()
    print("\n=== 2. ALGO TEP ===")
    algo_sess = compute_algo_tep()
    print("\n=== 3. BW floor (no TEP) ===")
    document_bw_floor()
    print("\n=== 4–5. Convergence ===")
    run_convergence(gsm_inst, algo_sess)


if __name__ == "__main__":
    main()
