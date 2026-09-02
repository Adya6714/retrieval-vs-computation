#!/usr/bin/env python3
"""M3: Probe 2 / Probe 1 convergence — report non-convergence with bootstrap CIs.

point-biserial(CCI, W3_correct) is the primary instance-level test.
Across-model Spearman(CCI, retention) uses GSM model aggregates; n=4 because
o4-mini has no GSM P2 per-instance CCI data.

Do NOT treat retention and phi as convergent — they are algebraically linked
from the same canonical/W3 contingency table (see p1_construct_validity.py).
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

from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402
from probes.behavioral.retention import MIN_CANONICAL_FOR_RETENTION  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
RAW = REPO_ROOT / "results" / "raw"
PHI_IN = DER / "P1_phi_canonical_w3.csv"
P2_IN = RAW / "GSM_P2_cci.csv"
OUT = DER / "P2_P1_convergence.csv"

PAPER_MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
}
N_BOOT = 5000
SEED = 42


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _load_gsm_p1() -> pd.DataFrame:
    parts = []
    for path in sorted(DER.glob("GSM_P1_*rescored.csv")):
        df = pd.read_csv(path, dtype=str).fillna("")
        if "included" not in df.columns:
            continue
        df = df[_is_true(df["included"])].copy()
        df = filter_excluded(df, family="GSM")
        df["variant"] = df["variant_type"].map(normalize_variant)
        ok = df["rescored_correct"] if "rescored_correct" in df.columns else df.get("verified", "")
        df["ok"] = _is_true(ok)
        df["model_short"] = df["model"].map(
            {
                "anthropic/claude-sonnet-4": "Claude",
                "openai/gpt-4o": "GPT-4o",
                "google/gemini-2.5-flash": "Gemini",
                "meta-llama/llama-3.1-8b-instruct": "Llama",
                "openai/o4-mini": "o4-mini",
            },
        )
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates(
        ["problem_id", "variant", "model_short"], keep="last",
    )


def _bootstrap_pointbiserial(x: pd.Series, y: pd.Series, cluster_ids: list[str]) -> tuple[float, float, float]:
    r, _ = stats.pointbiserialr(y.astype(float), x.astype(float))
    clusters = sorted(set(cluster_ids))
    grouped = {c: [i for i, cid in enumerate(cluster_ids) if cid == c] for c in clusters}
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT, dtype=float)
    xv = x.astype(float).values
    yv = y.astype(float).values
    for i in range(N_BOOT):
        draw = rng.choice(clusters, size=len(clusters), replace=True)
        idx = [j for c in draw for j in grouped[c]]
        if len(idx) < 5 or len(set(yv[idx])) < 2:
            boots[i] = float("nan")
        else:
            boots[i], _ = stats.pointbiserialr(yv[idx], xv[idx])
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return float(r), float("nan"), float("nan")
    return float(r), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def _bootstrap_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    rho, _ = stats.spearmanr(x, y)
    n = len(x)
    rng = np.random.default_rng(SEED)
    boots = np.empty(N_BOOT, dtype=float)
    idx = np.arange(n)
    for i in range(N_BOOT):
        draw = rng.choice(idx, size=n, replace=True)
        if len(set(x.iloc[draw])) < 2 or len(set(y.iloc[draw])) < 2:
            boots[i] = float("nan")
        else:
            boots[i], _ = stats.spearmanr(x.iloc[draw], y.iloc[draw])
    boots = boots[np.isfinite(boots)]
    if len(boots) == 0:
        return float(rho), float("nan"), float("nan")
    return float(rho), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def main() -> None:
    if not P2_IN.exists():
        raise FileNotFoundError(P2_IN)
    p1 = _load_gsm_p1()
    p2 = pd.read_csv(P2_IN, dtype=str).fillna("")
    p2["model_short"] = p2["model"].map(PAPER_MODELS)
    p2 = p2[p2["model_short"].notna()].copy()
    p2["cci"] = pd.to_numeric(p2["cci_score"], errors="coerce")

    w3 = p1[p1["variant"] == "W3"][["problem_id", "model_short", "ok"]].rename(columns={"ok": "w3_ok"})
    can = p1[p1["variant"] == "canonical"][["problem_id", "model_short", "ok"]].rename(columns={"ok": "canonical_ok"})
    merged = p2.merge(w3, on=["problem_id", "model_short"], how="inner")
    merged = merged.merge(can, on=["problem_id", "model_short"], how="left")
    merged["w3_correct"] = merged["w3_ok"].astype(int)
    merged = merged[merged["cci"].notna()].copy()

    rows: list[dict] = []

    # Pooled instance-level
    sub = merged.copy()
    r, lo, hi = _bootstrap_pointbiserial(
        sub["cci"], sub["w3_correct"], sub["problem_id"].astype(str).tolist(),
    )
    _, p = stats.pointbiserialr(sub["w3_correct"].astype(float), sub["cci"].astype(float))
    rows.append(
        {
            "analysis": "pointbiserial_cci_w3_correct",
            "scope": "GSM_all_instances_with_p2",
            "statistic": round(r, 3),
            "ci_low": round(lo, 3) if lo == lo else "",
            "ci_high": round(hi, 3) if hi == hi else "",
            "p_value": round(float(p), 3),
            "n": len(sub),
            "note": "P2 CCI vs W3 correctness; cluster-bootstrap by problem_id",
        }
    )

    # Within canonical-correct subset
    sub_cc = merged[merged["canonical_ok"].astype(bool)].copy()
    if not sub_cc.empty:
        r, lo, hi = _bootstrap_pointbiserial(
            sub_cc["cci"], sub_cc["w3_correct"], sub_cc["problem_id"].astype(str).tolist(),
        )
        _, p = stats.pointbiserialr(sub_cc["w3_correct"].astype(float), sub_cc["cci"].astype(float))
        rows.append(
            {
                "analysis": "pointbiserial_cci_w3_correct",
                "scope": "GSM_canonical_correct_subset",
                "statistic": round(r, 3),
                "ci_low": round(lo, 3) if lo == lo else "",
                "ci_high": round(hi, 3) if hi == hi else "",
                "p_value": round(float(p), 3),
                "n": len(sub_cc),
                "note": "Same test restricted to canonical-correct instances",
            }
        )

    # Across models: mean CCI vs retention (GSM only; models with P2)
    if PHI_IN.exists():
        phi = pd.read_csv(PHI_IN, dtype=str).fillna("")
        phi = phi[phi["family"] == "GSM"].copy()
        for col in ["retention_w3", "acc_canonical"]:
            phi[col] = pd.to_numeric(phi[col], errors="coerce")
        phi = phi[phi["acc_canonical"] >= MIN_CANONICAL_FOR_RETENTION].copy()
        phi = phi[phi["model"].isin(PAPER_MODELS.values())].copy()

        cci_means = merged.groupby("model_short", as_index=False)["cci"].mean().rename(
            columns={"cci": "mean_cci"},
        )
        model_df = phi.merge(cci_means, left_on="model", right_on="model_short", how="inner")
        if len(model_df) >= 3:
            rho, lo, hi = _bootstrap_spearman(model_df["mean_cci"], model_df["retention_w3"])
            _, p = stats.spearmanr(model_df["mean_cci"], model_df["retention_w3"])
            rows.append(
                {
                    "analysis": "spearman_mean_cci_retention",
                    "scope": "GSM_across_models",
                    "statistic": round(rho, 3),
                    "ci_low": round(lo, 3) if lo == lo else "",
                    "ci_high": round(hi, 3) if hi == hi else "",
                    "p_value": round(float(p), 3),
                    "n": len(model_df),
                    "note": "o4-mini excluded — no GSM P2 per-instance CCI (n=4 models)",
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(out)} rows)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
