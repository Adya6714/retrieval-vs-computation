#!/usr/bin/env python3
"""O15: Surprisal contamination vs Infini-gram (second Probe-3 measure).

Reads Colab output ``results/raw/O15_surprisal_contamination.csv``, adds
length-controlled NLL residuals (OLS of mean_nll ~ n_tokens within
family×model), writes derived surprisal table, and Spearman-correlates each
surprisal measure with existing Infini-gram ``contamination_score`` per family
(cluster-bootstrap CIs). Families are never pooled (windows 8 vs 13).

Pre-registered reading (do not revise after seeing results):
  - Agreement with Infini-gram → validates Probe 3's contamination proxy.
  - Disagreement → the field's standard n-gram proxy is unreliable here.
Report which.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.cluster_inference import cluster_bootstrap_assoc  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
RAW = REPO_ROOT / "results" / "raw"

SURPRISAL_CANDIDATES = [
    RAW / "O15_surprisal_contamination.csv",
    REPO_ROOT / "colab_out" / "O15_surprisal_contamination.csv",
]
OUT_SURP = DER / "O15_surprisal_contamination.csv"
OUT_CORR = DER / "O15_surprisal_vs_infinigram.csv"

N_BOOT = 5000
SEED = 42
INFINIGRAM_WINDOW = {"GSM": 8, "ALGO": 13, "BW": 13}

# Surprisal columns correlated against Infini-gram contamination_score.
# Sign expectation under "both measure contamination":
#   mean_nll / residual_mean_nll / min_k_mean_nll : negative (more overlap → less NLL)
#   min_k_mean_logprob : positive (more overlap → higher bottom-k logprob)
MEASURES = [
    ("mean_nll", "negative"),
    ("residual_mean_nll", "negative"),
    ("min_k_mean_logprob", "positive"),
    ("min_k_mean_nll", "negative"),
]

# |ρ| above this with CI excluding 0 → "agrees"; CI includes 0 → "null";
# significant wrong-sign → "disagrees".
AGREE_ABS_RHO = 0.10


def _load_surprisal() -> pd.DataFrame:
    path = next((p for p in SURPRISAL_CANDIDATES if p.exists()), None)
    if path is None:
        tried = "\n  ".join(str(p) for p in SURPRISAL_CANDIDATES)
        raise FileNotFoundError(
            "Missing O15 Colab output. Tried:\n  "
            f"{tried}\n"
            "Copy colab_out/O15_surprisal_contamination.csv → results/raw/."
        )
    df = pd.read_csv(path)
    print(f"[load] surprisal from {path} ({len(df)} rows)")
    for col in ("mean_nll", "n_tokens", "min_k_mean_logprob", "min_k_mean_nll"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["family"] = df["family"].astype(str).str.strip().str.upper()
    df["problem_id"] = df["problem_id"].astype(str).str.strip()
    df["variant"] = df["variant"].astype(str).str.strip()
    df["model"] = df["model"].astype(str).str.strip()
    df["clone_family"] = df["clone_family"].astype(str).str.strip()
    # Normalize variant labels.
    df.loc[df["variant"].str.lower().eq("canonical"), "variant"] = "canonical"
    return df


def _residualize(df: pd.DataFrame) -> pd.DataFrame:
    """OLS residual of mean_nll on n_tokens within family×model."""
    parts: list[pd.DataFrame] = []
    for (_, _), g in df.groupby(["family", "model"], sort=False):
        sub = g.copy()
        x = sub["n_tokens"].to_numpy(dtype=float)
        y = sub["mean_nll"].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        resid = np.full(len(sub), np.nan, dtype=float)
        if mask.sum() >= 3 and np.unique(x[mask]).size >= 2:
            # y = a + b x + e  → residual e
            b, a = np.polyfit(x[mask], y[mask], 1)
            resid[mask] = y[mask] - (a + b * x[mask])
            sub["nll_length_slope"] = b
            sub["nll_length_intercept"] = a
        else:
            sub["nll_length_slope"] = np.nan
            sub["nll_length_intercept"] = np.nan
        sub["residual_mean_nll"] = resid
        parts.append(sub)
    return pd.concat(parts, ignore_index=True)


def _load_infinigram() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fam, name in (
        ("GSM", "GSM_P3_contamination.csv"),
        ("ALGO", "ALGO_P3_contamination.csv"),
        ("BW", "BW_P3_contamination.csv"),
    ):
        path = RAW / name
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        df["family"] = fam
        df["problem_id"] = df["problem_id"].astype(str).str.strip()
        df["contamination_score"] = pd.to_numeric(
            df["contamination_score"], errors="coerce"
        )
        if "max_ngram_length" in df.columns:
            df["max_ngram_length"] = pd.to_numeric(
                df["max_ngram_length"], errors="coerce"
            )
        else:
            df["max_ngram_length"] = np.nan
        # Dedup ALGO duplicate problem_ids (keep first).
        df = df.drop_duplicates("problem_id", keep="first")
        rows.append(
            df[
                [
                    "family",
                    "problem_id",
                    "contamination_score",
                    "max_ngram_length",
                ]
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    print(f"[load] Infini-gram {len(out)} unique problem_ids")
    return out


def _verdict(rho: float, ci_low: float, ci_high: float, expected_sign: str) -> str:
    if not np.isfinite(rho):
        return "insufficient_data"
    ci_excludes_0 = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)
    if not ci_excludes_0:
        return "null_compatible"
    sign = "positive" if rho > 0 else "negative"
    if abs(rho) < AGREE_ABS_RHO:
        return "significant_but_negligible"
    if sign == expected_sign:
        return "agrees_with_infinigram"
    return "disagrees_with_infinigram"


def correlate(surp: pd.DataFrame, ig: pd.DataFrame) -> pd.DataFrame:
    # Infini-gram scores the canonical problem text → primary join on canonical.
    can = surp[surp["variant"].str.lower().eq("canonical")].copy()
    merged = can.merge(ig, on=["family", "problem_id"], how="inner")
    merged = merged[merged["contamination_score"].notna()].copy()

    rows: list[dict] = []
    for (family, model), g in merged.groupby(["family", "model"], sort=False):
        short = str(g["model_short"].iloc[0]) if "model_short" in g.columns else model
        quantized = str(g["quantized"].iloc[0]) if "quantized" in g.columns else ""
        clusters = g["clone_family"].astype(str).to_numpy()
        # One row per problem after canonical filter; still cluster on clone_family.
        y = g["contamination_score"].to_numpy(dtype=float)

        for measure, expected in MEASURES:
            if measure not in g.columns:
                continue
            x = g[measure].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 5:
                rows.append(
                    {
                        "family": family,
                        "model": model,
                        "model_short": short,
                        "quantized": quantized,
                        "measure": measure,
                        "expected_sign_if_both_contam": expected,
                        "n": int(mask.sum()),
                        "n_clusters": int(len(set(clusters[mask].tolist()))),
                        "spearman_rho": "",
                        "ci_low": "",
                        "ci_high": "",
                        "p_clustered": "",
                        "infinigram_max_window": INFINIGRAM_WINDOW.get(str(family), ""),
                        "subset": "canonical_problem_statements",
                        "bootstrap": "cluster_on_clone_family",
                        "n_boot": N_BOOT,
                        "seed": SEED,
                        "verdict": "insufficient_data",
                    }
                )
                continue

            seed = (
                hash(f"O15|{family}|{model}|{measure}") % (2**31 - 1)
            )
            assoc = cluster_bootstrap_assoc(
                x[mask],
                y[mask],
                clusters[mask].tolist(),
                kind="spearman",
                n_boot=N_BOOT,
                seed=int(seed),
            )
            rho = assoc["estimate"]
            verdict = _verdict(
                rho, assoc["ci_low"], assoc["ci_high"], expected
            )
            rows.append(
                {
                    "family": family,
                    "model": model,
                    "model_short": short,
                    "quantized": quantized,
                    "measure": measure,
                    "expected_sign_if_both_contam": expected,
                    "n": assoc["n"],
                    "n_clusters": assoc["n_clusters"],
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
                    "infinigram_max_window": INFINIGRAM_WINDOW.get(str(family), ""),
                    "subset": "canonical_problem_statements",
                    "bootstrap": "cluster_on_clone_family",
                    "n_boot": N_BOOT,
                    "seed": SEED,
                    "verdict": verdict,
                }
            )

    # Pre-registered summary across primary (non-quantized) cells.
    corr_df = pd.DataFrame(rows)
    primary = corr_df[
        corr_df["quantized"].astype(str).str.lower().isin({"false", "0", ""})
        & corr_df["measure"].isin(["residual_mean_nll", "min_k_mean_logprob"])
    ]
    if not primary.empty:
        vc = primary["verdict"].value_counts().to_dict()
        n_agree = int(vc.get("agrees_with_infinigram", 0))
        n_disagree = int(vc.get("disagrees_with_infinigram", 0))
        n_null = int(vc.get("null_compatible", 0)) + int(
            vc.get("significant_but_negligible", 0)
        )
        n_tot = len(primary)
        if n_agree > n_disagree and n_agree >= max(1, n_tot // 3):
            gate = (
                f"AGREEMENT_DOMINATES ({n_agree}/{n_tot} agree, {n_disagree} disagree, "
                f"{n_null} null) → Probe 3 Infini-gram validated by independent surprisal"
            )
        elif n_disagree > n_agree and n_disagree >= max(1, n_tot // 3):
            gate = (
                f"DISAGREEMENT_DOMINATES ({n_disagree}/{n_tot} disagree, {n_agree} agree, "
                f"{n_null} null) → field standard n-gram proxy unreliable here"
            )
        else:
            gate = (
                f"MIXED/NULL ({n_agree} agree, {n_disagree} disagree, {n_null} null "
                f"of {n_tot}) → Probe 3 proxy not clearly validated nor refuted"
            )
        rows.append(
            {
                "family": "ALL",
                "model": "PRIMARY_fp16",
                "model_short": "PRIMARY_fp16",
                "quantized": "false",
                "measure": "residual_mean_nll+min_k_mean_logprob",
                "expected_sign_if_both_contam": "mixed",
                "n": n_tot,
                "n_clusters": "",
                "spearman_rho": "",
                "ci_low": "",
                "ci_high": "",
                "p_clustered": "",
                "infinigram_max_window": "family-specific_8_or_13",
                "subset": "canonical_problem_statements",
                "bootstrap": "cluster_on_clone_family",
                "n_boot": N_BOOT,
                "seed": SEED,
                "verdict": gate,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    surp = _load_surprisal()
    surp = _residualize(surp)
    surp.to_csv(OUT_SURP, index=False)
    print(f"[write] {OUT_SURP} ({len(surp)} rows)")

    ig = _load_infinigram()
    corr = correlate(surp, ig)
    corr.to_csv(OUT_CORR, index=False)
    print(f"[write] {OUT_CORR} ({len(corr)} rows)")

    summary = corr[corr["family"].eq("ALL")]
    if not summary.empty:
        print("[preregistered]", summary.iloc[0]["verdict"])
    else:
        cells = corr[corr["family"].ne("ALL")]
        print("[verdicts]\n", cells.groupby("verdict").size().to_string())


if __name__ == "__main__":
    main()
