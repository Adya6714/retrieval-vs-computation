#!/usr/bin/env python3
"""M3/N2: Probe 2 / Probe 1 convergence — report non-convergence with bootstrap CIs.

point-biserial(CCI, W3_correct) at instance level for GSM, ALGO, and pooled.
Across-model Spearman(CCI, retention) uses model aggregates.

Do NOT treat retention and phi as convergent — they are algebraically linked
from the same canonical/W3 contingency table (see p1_construct_validity.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.behavioral.retention import MIN_CANONICAL_FOR_RETENTION  # noqa: E402
from probes.common.clones import cluster_ids_for  # noqa: E402
from probes.common.cluster_inference import (  # noqa: E402
    cluster_bootstrap_assoc,
    iid_bootstrap_assoc,
)
from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
RAW = REPO_ROOT / "results" / "raw"
PHI_IN = DER / "P1_phi_canonical_w3.csv"
GSM_P2_IN = RAW / "GSM_P2_cci.csv"
ALGO_P2_IN = DER / "ALGO_P2_cci.csv"
OUT = DER / "P2_P1_convergence.csv"

MODEL_MAP = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
N_BOOT = 5000
SEED = 42


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _load_p1(family: str) -> pd.DataFrame:
    glob = {"GSM": "GSM_P1_*", "ALGO": "ALGO_P1_*", "BW": "BW_P1_*"}.get(family, f"{family}_P1_*")
    parts = []
    for path in sorted(DER.glob(f"{glob}rescored.csv")):
        if "review" in path.name.lower():
            continue
        df = pd.read_csv(path, dtype=str).fillna("")
        if "included" not in df.columns:
            continue
        df = df[_is_true(df["included"])].copy()
        df = filter_excluded(df, family=family)
        df["family"] = family
        df["variant"] = df["variant_type"].map(normalize_variant)
        ok = df["rescored_correct"] if "rescored_correct" in df.columns else df.get("verified", "")
        df["ok"] = _is_true(ok)
        df["model_short"] = df["model"].map(MODEL_MAP).fillna(df["model"])
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True).drop_duplicates(
        ["family", "problem_id", "variant", "model_short"], keep="last",
    )


def _cluster_ids(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return []
    if frame["family"].iloc[0] == "ALGO":
        return cluster_ids_for(frame["problem_id"].astype(str).tolist())
    return frame["problem_id"].astype(str).tolist()


def _merge_p2_p1(p2: pd.DataFrame, p1: pd.DataFrame, *, family: str) -> pd.DataFrame:
    p2 = p2.copy()
    p2["model_short"] = p2["model"].map(MODEL_MAP).fillna(p2["model"])
    p2["cci"] = pd.to_numeric(p2["cci_score"], errors="coerce")
    p2 = p2[p2["model_short"].notna()].copy()
    w3 = p1[p1["variant"] == "W3"][["problem_id", "model_short", "ok"]].rename(columns={"ok": "w3_ok"})
    can = p1[p1["variant"] == "canonical"][["problem_id", "model_short", "ok"]].rename(
        columns={"ok": "canonical_ok"},
    )
    merged = p2.merge(w3, on=["problem_id", "model_short"], how="inner")
    merged = merged.merge(can, on=["problem_id", "model_short"], how="left")
    merged["family"] = family
    merged["w3_correct"] = merged["w3_ok"].astype(int)
    merged = merged[merged["cci"].notna()].copy()
    merged["cluster_id"] = _cluster_ids(merged)
    return merged


def _append_pointbiserial_rows(
    rows: list[dict],
    merged: pd.DataFrame,
    *,
    scope_prefix: str,
) -> None:
    sub = merged.copy()
    if sub.empty:
        return
    for scope, frame, note in [
        (
            f"{scope_prefix}_all_instances_with_p2",
            sub,
            "P2 CCI vs W3; CI+p from same cluster bootstrap",
        ),
        (
            f"{scope_prefix}_canonical_correct_subset",
            merged[merged["canonical_ok"].astype(bool)].copy(),
            "Same test restricted to canonical-correct instances",
        ),
    ]:
        if frame.empty:
            continue
        res = cluster_bootstrap_assoc(
            frame["cci"],
            frame["w3_correct"],
            frame["cluster_id"].astype(str),
            kind="pointbiserial",
            n_boot=N_BOOT,
            seed=SEED,
        )
        rows.append(
            {
                "analysis": "pointbiserial_cci_w3_correct",
                "scope": scope,
                "statistic": round(res["estimate"], 3),
                "ci_low": round(res["ci_low"], 3) if res["ci_low"] == res["ci_low"] else "",
                "ci_high": round(res["ci_high"], 3) if res["ci_high"] == res["ci_high"] else "",
                "p_value": round(res["p_clustered"], 4) if res["p_clustered"] == res["p_clustered"] else "",
                "p_value_method": "cluster_bootstrap_two_sided",
                "n": res["n"],
                "n_clusters": res["n_clusters"],
                "note": note,
            }
        )


def main() -> None:
    rows: list[dict] = []
    frames: list[pd.DataFrame] = []

    family_frames: list[tuple[str, pd.DataFrame]] = []

    if GSM_P2_IN.exists():
        gsm_p1 = _load_p1("GSM")
        gsm_p2 = pd.read_csv(GSM_P2_IN, dtype=str).fillna("")
        gsm_m = _merge_p2_p1(gsm_p2, gsm_p1, family="GSM")
        frames.append(gsm_m)
        family_frames.append(("GSM", gsm_m))
        _append_pointbiserial_rows(rows, gsm_m, scope_prefix="GSM")

    if ALGO_P2_IN.exists():
        algo_p1 = _load_p1("ALGO")
        algo_p2 = pd.read_csv(ALGO_P2_IN, dtype=str).fillna("")
        algo_m = _merge_p2_p1(algo_p2, algo_p1, family="ALGO")
        frames.append(algo_m)
        family_frames.append(("ALGO", algo_m))
        _append_pointbiserial_rows(rows, algo_m, scope_prefix="ALGO")

    if frames:
        pooled = pd.concat(frames, ignore_index=True)
        pooled["cluster_id"] = pooled.apply(
            lambda r: f"{r['family']}:{r['cluster_id']}", axis=1,
        )
        _append_pointbiserial_rows(rows, pooled, scope_prefix="GSM_ALGO_pooled")

    if PHI_IN.exists() and frames:
        phi = pd.read_csv(PHI_IN, dtype=str).fillna("")
        for col in ["retention_w3", "acc_canonical"]:
            phi[col] = pd.to_numeric(phi[col], errors="coerce")
        phi = phi[phi["acc_canonical"] >= MIN_CANONICAL_FOR_RETENTION].copy()

        for fam, merged in family_frames:
            if merged.empty:
                continue
            cci_means = merged.groupby("model_short", as_index=False)["cci"].mean().rename(
                columns={"cci": "mean_cci", "model_short": "model"},
            )
            sub_phi = phi[phi["family"] == fam].copy()
            model_df = sub_phi.merge(cci_means, on="model", how="inner")
            if len(model_df) < 2:
                continue
            res = iid_bootstrap_assoc(
                model_df["mean_cci"],
                model_df["retention_w3"],
                kind="spearman",
                n_boot=N_BOOT,
                seed=SEED,
            )
            note = f"{fam} models with P2 CCI and phi retention (can_acc>={MIN_CANONICAL_FOR_RETENTION})"
            if fam == "GSM":
                note += "; o4-mini excluded from GSM P2 CCI"
            if fam == "ALGO":
                note += "; Gemini CCI all NaN; Llama below canonical floor"
            rows.append(
                {
                    "analysis": "spearman_mean_cci_retention",
                    "scope": f"{fam}_across_models",
                    "statistic": round(res["estimate"], 3),
                    "ci_low": round(res["ci_low"], 3) if res["ci_low"] == res["ci_low"] else "",
                    "ci_high": round(res["ci_high"], 3) if res["ci_high"] == res["ci_high"] else "",
                    "p_value": round(res["p_clustered"], 4) if res["p_clustered"] == res["p_clustered"] else "",
                    "p_value_method": "iid_bootstrap_two_sided",
                    "n": res["n"],
                    "n_clusters": res["n"],
                    "note": note,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(out)} rows)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
