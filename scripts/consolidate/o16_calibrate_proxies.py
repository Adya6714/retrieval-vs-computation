#!/usr/bin/env python3
"""O16 Part C: Calibrate proxies against corpus ground truth + GT retention test.

Inputs:
  - results/derived/O16_corpus_ground_truth.csv          (Part A)
  - results/raw/{GSM,ALGO,BW}_P3_contamination.csv       (field Infini-gram proxy, RPJ)
  - results/raw/O16_open_model_scores.csv                (Part B: O5+O15 on Pythia/OLMo)
    optional fallback: results/raw/O15_surprisal_contamination.csv

Outputs:
  - O16_proxy_calibration.csv         ROC AUC of proxies vs GT membership
  - O16_groundtruth_retention_test.csv  N5-style W3 retention on GT members

Pre-registered:
  - Contaminated (GT) instances should show LOWER W3 retention among
    canonical-correct cells (negative association).
  - Proxies that recover GT membership (AUC≫0.5) are validated; else unreliable.

Closed-model caveat (paper): GT membership is unidentified for Claude, GPT-4o,
Gemini, o4-mini, DeepSeek — permanent limitation of contamination research on
closed models.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.clones import cluster_ids_for  # noqa: E402
from probes.common.cluster_inference import (  # noqa: E402
    bootstrap_p_two_sided,
    cluster_bootstrap_assoc,
)
from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
RAW = REPO_ROOT / "results" / "raw"

GT_PATH = DER / "O16_corpus_ground_truth.csv"
OUT_CAL = DER / "O16_proxy_calibration.csv"
OUT_RET = DER / "O16_groundtruth_retention_test.csv"

N_BOOT = 5000
SEED = 42

PAPER_MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
    "deepseek/deepseek-r1-distill-llama-70b": "DeepSeek",
}

# Open-model lineage ↔ corpus used as GT for that model's surprisal.
MODEL_CORPUS = {
    "EleutherAI/pythia-2.8b": "pile",
    "allenai/OLMo-2-0425-1B": "dolma",
}


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _load_gt() -> pd.DataFrame:
    if not GT_PATH.exists():
        raise FileNotFoundError(
            f"Missing {GT_PATH}. Run scripts/consolidate/o16_corpus_ground_truth.py first."
        )
    df = pd.read_csv(GT_PATH)
    df["family"] = df["family"].astype(str).str.strip().str.upper()
    df["problem_id"] = df["problem_id"].astype(str).str.strip()
    df["corpus"] = df["corpus"].astype(str).str.strip().str.lower()
    for col in (
        "exact_match_found",
        "near_exact_match_found",
        "ground_truth_member",
    ):
        df[col] = _is_true(df[col]) if df[col].dtype == object else df[col].astype(bool)
    df["matched_contamination_score"] = pd.to_numeric(
        df["matched_contamination_score"], errors="coerce"
    )
    return df


def _load_rpj_proxy() -> pd.DataFrame:
    rows = []
    for fam, name in (
        ("GSM", "GSM_P3_contamination.csv"),
        ("ALGO", "ALGO_P3_contamination.csv"),
        ("BW", "BW_P3_contamination.csv"),
    ):
        path = RAW / name
        df = pd.read_csv(path)
        df["family"] = fam
        df["problem_id"] = df["problem_id"].astype(str).str.strip()
        df["contamination_score_rpj"] = pd.to_numeric(
            df["contamination_score"], errors="coerce"
        )
        df = df.drop_duplicates("problem_id", keep="first")
        rows.append(df[["family", "problem_id", "contamination_score_rpj"]])
    return pd.concat(rows, ignore_index=True)


def _load_open_scores() -> pd.DataFrame:
    """Part B scores; optional O15 fallback for statement surprisal only."""
    candidates = [
        RAW / "O16_open_model_scores.csv",
        REPO_ROOT / "colab_out" / "O16_open_model_scores.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is not None:
        df = pd.read_csv(path)
        print(f"[load] open-model scores from {path}")
        return df

    # Fallback: O15 surprisal only (canonical).
    o15_cands = [
        RAW / "O15_surprisal_contamination.csv",
        DER / "O15_surprisal_contamination.csv",
    ]
    o15 = next((p for p in o15_cands if p.exists()), None)
    if o15 is None:
        print("[load] no Part B / O15 scores yet — surprisal proxies skipped")
        return pd.DataFrame()
    df = pd.read_csv(o15)
    df = df[df["variant"].astype(str).str.lower().eq("canonical")].copy()
    df = df[df["model"].isin(MODEL_CORPUS.keys())].copy()
    # Alias O15 columns into O16 naming.
    rename = {
        "mean_nll": "o15_mean_nll",
        "min_k_mean_logprob": "o15_min_k_mean_logprob",
        "min_k_mean_nll": "o15_min_k_mean_nll",
        "n_tokens": "o15_n_tokens",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "o15_mean_nll" in df.columns and "o15_n_tokens" in df.columns:
        # residualize within model
        parts = []
        for _, g in df.groupby("model"):
            sub = g.copy()
            x = pd.to_numeric(sub["o15_n_tokens"], errors="coerce").to_numpy(float)
            y = pd.to_numeric(sub["o15_mean_nll"], errors="coerce").to_numpy(float)
            m = np.isfinite(x) & np.isfinite(y)
            resid = np.full(len(sub), np.nan)
            if m.sum() >= 3 and np.unique(x[m]).size >= 2:
                b, a = np.polyfit(x[m], y[m], 1)
                resid[m] = y[m] - (a + b * x[m])
            sub["o15_residual_mean_nll"] = resid
            parts.append(sub)
        df = pd.concat(parts, ignore_index=True)
    print(f"[load] O15 fallback from {o15}")
    return df


def _load_p1() -> pd.DataFrame:
    parts = []
    for path in sorted(DER.glob("*_P1_*rescored.csv")):
        if "review" in path.name.lower():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "included" not in df.columns:
            continue
        df = df[_is_true(df["included"])].copy()
        if path.name.startswith("ALGO_"):
            fam = "ALGO"
        elif path.name.startswith("BW_"):
            fam = "BW"
        elif path.name.startswith("GSM_"):
            fam = "GSM"
        else:
            continue
        df = filter_excluded(df, family=fam)
        df["family"] = fam
        df["model_short"] = df["model"].map(PAPER_MODELS).fillna(df["model"])
        df["variant"] = df["variant_type"].map(normalize_variant)
        ok = (
            df["rescored_correct"]
            if "rescored_correct" in df.columns
            else df.get("verified", "")
        )
        df["ok"] = _is_true(ok)
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates(
        ["family", "problem_id", "variant", "model_short"], keep="last"
    )


def _cluster_ids(family: str, problem_ids: pd.Series) -> list[str]:
    pids = problem_ids.astype(str).tolist()
    if family == "ALGO":
        return cluster_ids_for(pids)
    return pids


def _roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(s) & np.isfinite(y)
    y, s = y[mask], s[mask]
    if len(y) < 5 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def cluster_bootstrap_auc(
    y_true: np.ndarray,
    scores: np.ndarray,
    cluster_ids: list[str],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(scores, dtype=float)
    cids = [str(c) for c in cluster_ids]
    mask = np.isfinite(y) & np.isfinite(s)
    y, s, cids = y[mask], s[mask], [c for c, m in zip(cids, mask) if m]
    estimate = _roc_auc(y, s)
    clusters = sorted(set(cids))
    if len(clusters) < 2 or not np.isfinite(estimate):
        return {
            "auc": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_clustered": float("nan"),
            "n": int(len(y)),
            "n_clusters": len(clusters),
            "n_positive": int(np.nansum(y)),
        }
    grouped = {c: [i for i, cid in enumerate(cids) if cid == c] for c in clusters}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = rng.choice(clusters, size=len(clusters), replace=True)
        idx = [j for c in draw for j in grouped[c]]
        boots[i] = _roc_auc(y[idx], s[idx])
    finite = boots[np.isfinite(boots)]
    # H0: AUC = 0.5
    centered = finite - 0.5
    return {
        "auc": estimate,
        "ci_low": float(np.percentile(finite, 2.5)) if len(finite) else float("nan"),
        "ci_high": float(np.percentile(finite, 97.5)) if len(finite) else float("nan"),
        "p_clustered": bootstrap_p_two_sided(centered) if len(finite) else float("nan"),
        "n": int(len(y)),
        "n_clusters": len(clusters),
        "n_positive": int(np.nansum(y)),
    }


def calibrate(gt: pd.DataFrame, rpj: pd.DataFrame, open_scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    # --- Matched Infini-gram proximity (same corpus as GT) ---
    for corpus, gcorp in gt.groupby("corpus"):
        for family, g in gcorp.groupby("family"):
            y = g["ground_truth_member"].astype(int).to_numpy()
            score = g["matched_contamination_score"].to_numpy(dtype=float)
            clusters = _cluster_ids(str(family), g["problem_id"])
            res = cluster_bootstrap_auc(y, score, clusters, seed=SEED + hash(f"m|{corpus}|{family}") % 10000)
            rows.append(
                {
                    "analysis": "proxy_roc_auc",
                    "family": family,
                    "corpus_gt": corpus,
                    "proxy": "infinigram_matched_contamination_score",
                    "proxy_model": "",
                    "expected_direction": "higher_score_more_member",
                    "n": res["n"],
                    "n_clusters": res["n_clusters"],
                    "n_positive_gt": res["n_positive"],
                    "auc": round(res["auc"], 6) if np.isfinite(res["auc"]) else "",
                    "ci_low": round(res["ci_low"], 6) if np.isfinite(res["ci_low"]) else "",
                    "ci_high": round(res["ci_high"], 6) if np.isfinite(res["ci_high"]) else "",
                    "p_auc_ne_0.5": round(res["p_clustered"], 6)
                    if np.isfinite(res["p_clustered"])
                    else "",
                    "verdict": _auc_verdict(res["auc"], res["ci_low"], res["ci_high"]),
                    "note": (
                        "SANITY/partially-circular: matched score is longest_ngram_frac; "
                        "GT near-exact also thresholds that length"
                    ),
                }
            )

    # --- Field RPJ Infini-gram proxy vs each corpus GT ---
    for corpus, gcorp in gt.groupby("corpus"):
        merged = gcorp.merge(rpj, on=["family", "problem_id"], how="inner")
        for family, g in merged.groupby("family"):
            y = g["ground_truth_member"].astype(int).to_numpy()
            score = g["contamination_score_rpj"].to_numpy(dtype=float)
            clusters = _cluster_ids(str(family), g["problem_id"])
            res = cluster_bootstrap_auc(
                y, score, clusters, seed=SEED + hash(f"rpj|{corpus}|{family}") % 10000
            )
            rows.append(
                {
                    "analysis": "proxy_roc_auc",
                    "family": family,
                    "corpus_gt": corpus,
                    "proxy": "infinigram_rpj_contamination_score",
                    "proxy_model": "",
                    "expected_direction": "higher_score_more_member",
                    "n": res["n"],
                    "n_clusters": res["n_clusters"],
                    "n_positive_gt": res["n_positive"],
                    "auc": round(res["auc"], 6) if np.isfinite(res["auc"]) else "",
                    "ci_low": round(res["ci_low"], 6) if np.isfinite(res["ci_low"]) else "",
                    "ci_high": round(res["ci_high"], 6) if np.isfinite(res["ci_high"]) else "",
                    "p_auc_ne_0.5": round(res["p_clustered"], 6)
                    if np.isfinite(res["p_clustered"])
                    else "",
                    "verdict": _auc_verdict(res["auc"], res["ci_low"], res["ci_high"]),
                    "note": "field Probe-3 proxy (RedPajama index) vs open-corpus GT",
                }
            )

    # --- O15 / O5 open-model surprisal (score inverted for NLL → membership) ---
    if open_scores.empty:
        rows.append(
            {
                "analysis": "proxy_roc_auc",
                "family": "ALL",
                "corpus_gt": "",
                "proxy": "o15/o5_open_models",
                "proxy_model": "",
                "expected_direction": "",
                "n": 0,
                "n_clusters": "",
                "n_positive_gt": "",
                "auc": "",
                "ci_low": "",
                "ci_high": "",
                "p_auc_ne_0.5": "",
                "verdict": "scores_missing",
                "note": "Run Colab o16_open_model_calibration.ipynb (Part B)",
            }
        )
        return pd.DataFrame(rows)

    os_df = open_scores.copy()
    os_df["family"] = os_df["family"].astype(str).str.strip().str.upper()
    os_df["problem_id"] = os_df["problem_id"].astype(str).str.strip()
    os_df["model"] = os_df["model"].astype(str).str.strip()

    # Measures: higher = more contaminated for ROC. Invert NLLs.
    measure_specs = []
    for col, invert, label in [
        ("o15_mean_nll", True, "o15_mean_nll"),
        ("o15_residual_mean_nll", True, "o15_residual_mean_nll"),
        ("o15_min_k_mean_logprob", False, "o15_min_k_mean_logprob"),
        ("o5_mean_nll_gold", True, "o5_mean_nll_gold"),
        ("o5_mean_logprob", False, "o5_mean_logprob"),
    ]:
        if col in os_df.columns:
            measure_specs.append((col, invert, label))

    for model, corpus in MODEL_CORPUS.items():
        g_model = os_df[os_df["model"].eq(model)]
        if g_model.empty:
            continue
        g_gt = gt[gt["corpus"].eq(corpus)]
        merged = g_model.merge(g_gt, on=["family", "problem_id"], how="inner")
        for family, g in merged.groupby("family"):
            y = g["ground_truth_member"].astype(int).to_numpy()
            clusters = _cluster_ids(str(family), g["problem_id"])
            for col, invert, label in measure_specs:
                raw = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
                score = (-raw) if invert else raw
                res = cluster_bootstrap_auc(
                    y,
                    score,
                    clusters,
                    seed=SEED + hash(f"{model}|{family}|{label}") % 10000,
                )
                rows.append(
                    {
                        "analysis": "proxy_roc_auc",
                        "family": family,
                        "corpus_gt": corpus,
                        "proxy": label,
                        "proxy_model": model,
                        "expected_direction": (
                            "lower_nll_more_member"
                            if invert
                            else "higher_logprob_more_member"
                        ),
                        "n": res["n"],
                        "n_clusters": res["n_clusters"],
                        "n_positive_gt": res["n_positive"],
                        "auc": round(res["auc"], 6) if np.isfinite(res["auc"]) else "",
                        "ci_low": round(res["ci_low"], 6)
                        if np.isfinite(res["ci_low"])
                        else "",
                        "ci_high": round(res["ci_high"], 6)
                        if np.isfinite(res["ci_high"])
                        else "",
                        "p_auc_ne_0.5": round(res["p_clustered"], 6)
                        if np.isfinite(res["p_clustered"])
                        else "",
                        "verdict": _auc_verdict(res["auc"], res["ci_low"], res["ci_high"]),
                        "note": "open-corpus model surprisal vs same-corpus GT",
                    }
                )

    # Summary gate over primary proxies
    cal = pd.DataFrame(rows)
    primary = cal[
        cal["proxy"].isin(
            [
                "infinigram_rpj_contamination_score",
                "o15_residual_mean_nll",
                "o15_min_k_mean_logprob",
            ]
        )
        & cal["verdict"].isin(["validates_proxy", "fails_to_predict_gt", "null_compatible"])
    ]
    if not primary.empty:
        vc = primary["verdict"].value_counts().to_dict()
        rows.append(
            {
                "analysis": "preregistered_summary",
                "family": "ALL",
                "corpus_gt": "pile+dolma",
                "proxy": "primary_set",
                "proxy_model": "",
                "expected_direction": "",
                "n": len(primary),
                "n_clusters": "",
                "n_positive_gt": "",
                "auc": "",
                "ci_low": "",
                "ci_high": "",
                "p_auc_ne_0.5": "",
                "verdict": (
                    f"validates={vc.get('validates_proxy', 0)}; "
                    f"fails={vc.get('fails_to_predict_gt', 0)}; "
                    f"null={vc.get('null_compatible', 0)} / {len(primary)}"
                ),
                "note": (
                    "Closed models (Claude/GPT-4o/Gemini/o4-mini/DeepSeek) have no "
                    "identifiable corpus GT — permanent limitation."
                ),
            }
        )
    return pd.DataFrame(rows)


def _auc_verdict(auc: float, ci_low: float, ci_high: float) -> str:
    if not np.isfinite(auc):
        return "insufficient_data"
    # CI entirely above 0.5 with AUC>=0.6 → validates
    if np.isfinite(ci_low) and ci_low > 0.5 and auc >= 0.6:
        return "validates_proxy"
    if np.isfinite(ci_high) and ci_high < 0.5:
        return "fails_to_predict_gt"  # worse than chance / inverted
    if abs(auc - 0.5) < 0.05 or (
        np.isfinite(ci_low) and np.isfinite(ci_high) and ci_low <= 0.5 <= ci_high
    ):
        return "null_compatible"
    if auc >= 0.6:
        return "suggestive_but_ci_includes_0.5"
    return "null_compatible"


def retention_test(gt: pd.DataFrame, p1: pd.DataFrame) -> pd.DataFrame:
    """N5 without proxy: GT member vs W3 retention (canonical-correct only)."""
    if p1.empty:
        return pd.DataFrame(
            [
                {
                    "analysis": "gt_retention",
                    "note": "No rescored P1 files found",
                    "verdict": "p1_missing",
                }
            ]
        )

    # Union GT across corpora: member if member in any searched corpus.
    gt_any = (
        gt.groupby(["family", "problem_id"], as_index=False)["ground_truth_member"]
        .max()
        .rename(columns={"ground_truth_member": "gt_member_any"})
    )
    # Also per-corpus labels.
    gt_wide = gt.pivot_table(
        index=["family", "problem_id"],
        columns="corpus",
        values="ground_truth_member",
        aggfunc="max",
    ).reset_index()
    gt_wide.columns = [
        c if c in ("family", "problem_id") else f"gt_member_{c}" for c in gt_wide.columns
    ]

    can = p1[p1["variant"] == "canonical"][
        ["family", "problem_id", "model_short", "ok"]
    ].rename(columns={"ok": "canonical_ok"})
    w3 = p1[p1["variant"] == "W3"][
        ["family", "problem_id", "model_short", "ok"]
    ].rename(columns={"ok": "w3_ok"})
    merged = can.merge(w3, on=["family", "problem_id", "model_short"], how="inner")
    merged = merged[merged["canonical_ok"]].copy()
    merged["retained_w3"] = merged["w3_ok"].astype(int)
    merged = merged.merge(gt_any, on=["family", "problem_id"], how="inner")
    merged = merged.merge(gt_wide, on=["family", "problem_id"], how="left")
    merged["gt_member_any"] = merged["gt_member_any"].astype(int)

    rows: list[dict] = []
    for gt_col, label in [
        ("gt_member_any", "any_open_corpus"),
        ("gt_member_pile", "pile"),
        ("gt_member_dolma", "dolma"),
    ]:
        if gt_col not in merged.columns:
            continue
        work = merged.dropna(subset=[gt_col]).copy()
        work[gt_col] = work[gt_col].astype(int)
        for family, gfam in work.groupby("family"):
            for model, g in gfam.groupby("model_short"):
                if len(g) < 5 or g[gt_col].nunique() < 2 or g["retained_w3"].nunique() < 2:
                    rows.append(
                        {
                            "analysis": "gt_retention",
                            "family": family,
                            "model": model,
                            "gt_definition": label,
                            "n": len(g),
                            "n_gt_member": int(g[gt_col].sum()) if len(g) else 0,
                            "retention_gt_member": "",
                            "retention_gt_clean": "",
                            "delta_retention_member_minus_clean": "",
                            "spearman_rho": "",
                            "ci_low": "",
                            "ci_high": "",
                            "p_clustered": "",
                            "verdict": "insufficient_variation",
                            "note": "Field claim: contaminated ⇒ lower W3 retention",
                            "closed_model_caveat": (
                                "GT unavailable for closed API models; this test uses "
                                "open-corpus membership labels on the shared item set."
                            ),
                        }
                    )
                    continue
                mem = g[g[gt_col] == 1]["retained_w3"]
                clean = g[g[gt_col] == 0]["retained_w3"]
                delta = float(mem.mean() - clean.mean())
                clusters = _cluster_ids(str(family), g["problem_id"])
                # Association: GT member (1) vs retention — expect negative rho
                assoc = cluster_bootstrap_assoc(
                    g[gt_col],
                    g["retained_w3"],
                    clusters,
                    kind="pointbiserial",
                    n_boot=N_BOOT,
                    seed=SEED + hash(f"ret|{family}|{model}|{label}") % 10000,
                )
                rho = assoc["estimate"]
                if np.isfinite(rho) and np.isfinite(assoc["ci_high"]) and assoc["ci_high"] < 0:
                    verdict = "supports_field_claim"
                elif np.isfinite(rho) and np.isfinite(assoc["ci_low"]) and assoc["ci_low"] > 0:
                    verdict = "contradicts_field_claim"
                else:
                    verdict = "null_compatible"
                rows.append(
                    {
                        "analysis": "gt_retention",
                        "family": family,
                        "model": model,
                        "gt_definition": label,
                        "n": assoc["n"],
                        "n_gt_member": int(g[gt_col].sum()),
                        "retention_gt_member": round(float(mem.mean()), 6),
                        "retention_gt_clean": round(float(clean.mean()), 6),
                        "delta_retention_member_minus_clean": round(delta, 6),
                        "spearman_rho": round(rho, 6) if np.isfinite(rho) else "",
                        "ci_low": round(assoc["ci_low"], 6)
                        if np.isfinite(assoc["ci_low"])
                        else "",
                        "ci_high": round(assoc["ci_high"], 6)
                        if np.isfinite(assoc["ci_high"])
                        else "",
                        "p_clustered": round(assoc["p_clustered"], 6)
                        if np.isfinite(assoc["p_clustered"])
                        else "",
                        "verdict": verdict,
                        "note": "point-biserial GT_member vs W3 retention; expect negative",
                        "closed_model_caveat": (
                            "GT unavailable for closed API models; this test uses "
                            "open-corpus membership labels on the shared item set."
                        ),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    gt = _load_gt()
    rpj = _load_rpj_proxy()
    open_scores = _load_open_scores()
    p1 = _load_p1()

    cal = calibrate(gt, rpj, open_scores)
    cal.to_csv(OUT_CAL, index=False)
    print(f"[write] {OUT_CAL} ({len(cal)} rows)")

    ret = retention_test(gt, p1)
    ret.to_csv(OUT_RET, index=False)
    print(f"[write] {OUT_RET} ({len(ret)} rows)")

    if "verdict" in cal.columns:
        print("[calibration verdicts]\n", cal["verdict"].value_counts().to_string())
    if "verdict" in ret.columns:
        print("[retention verdicts]\n", ret["verdict"].value_counts().to_string())


if __name__ == "__main__":
    main()
