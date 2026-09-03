#!/usr/bin/env python3
"""O11: Partial Spearman of contamination vs W3 retention, controlling for difficulty.

Recomputes every N5 cell (family × model, canonical-correct subset) with:
  raw Spearman ρ, partial Spearman ρ (controls), Δ = partial − raw,
  cluster-bootstrap CIs throughout. Families are never pooled
  (GSM Infini-gram window 8; ALGO/BW window 13).

Controls (per family):
  - BW: Fast Downward optimal plan length; n_goal_clauses
  - ALGO: difficulty_numeric / problem-size parameter
  - GSM: structural difficulty proxy (n numeric literals in statement)
  - all: problem-statement whitespace token count
  - all: item-level canonical pass rate across models
    (model×item canonical accuracy is fixed at 1 under the N5 filter,
     so the cross-model pass rate is the usable accuracy-based control)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.clones import cluster_ids_for  # noqa: E402
from probes.common.cluster_inference import bootstrap_p_two_sided  # noqa: E402
from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402
from probes.contamination.algo_instance_metrics import extract_algo_metrics  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
RAW = REPO_ROOT / "results" / "raw"
BANK = REPO_ROOT / "data" / "problems"
OUT = DER / "O11_contamination_partial_correlations.csv"

PAPER_MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
    "deepseek/deepseek-r1-distill-llama-70b": "DeepSeek",
}
N_BOOT = 5000
SEED = 42
FAMILIES = ("GSM", "ALGO", "BW")
# Do not pool: windows differ by family (documented on every row).
INFINIGRAM_WINDOW = {"GSM": 8, "ALGO": 13, "BW": 13}


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _token_count(text: str) -> int:
    return len(str(text).split())


def _n_numeric_literals(text: str) -> int:
    import re

    return len(re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?", str(text)))


def _rank_avg(a: np.ndarray) -> np.ndarray:
    return stats.rankdata(a, method="average").astype(float)


def partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray | None,
) -> float:
    """Spearman partial correlation via rank residualization (Pearson of residuals)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 5 or len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    rx, ry = _rank_avg(x), _rank_avg(y)
    if Z is None or Z.size == 0 or Z.ndim != 2 or Z.shape[1] == 0:
        r, _ = stats.spearmanr(x, y)
        return float(r)
    # drop zero-variance / all-nan covariate columns
    cols = []
    for j in range(Z.shape[1]):
        col = Z[:, j].astype(float)
        if np.isfinite(col).sum() < 5:
            continue
        col = np.where(np.isfinite(col), col, np.nanmedian(col[np.isfinite(col)]))
        if np.nanstd(col) < 1e-12:
            continue
        cols.append(_rank_avg(col))
    if not cols:
        r, _ = stats.spearmanr(x, y)
        return float(r)
    RZ = np.column_stack(cols)
    # residualize with intercept
    ones = np.ones((len(rx), 1))
    A = np.hstack([ones, RZ])
    bx, *_ = np.linalg.lstsq(A, rx, rcond=None)
    by, *_ = np.linalg.lstsq(A, ry, rcond=None)
    ex = rx - A @ bx
    ey = ry - A @ by
    if np.std(ex) < 1e-12 or np.std(ey) < 1e-12:
        return float("nan")
    r, _ = stats.pearsonr(ex, ey)
    return float(r)


def cluster_bootstrap_partial(
    x: np.ndarray,
    y: np.ndarray,
    Z: np.ndarray | None,
    cluster_ids: list[str],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    estimate = partial_spearman(x, y, Z)
    clusters = sorted(set(cluster_ids))
    grouped = {c: [i for i, cid in enumerate(cluster_ids) if cid == c] for c in clusters}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = rng.choice(clusters, size=len(clusters), replace=True)
        idx = [j for c in draw for j in grouped[c]]
        Zb = None if Z is None else Z[idx]
        boots[i] = partial_spearman(x[idx], y[idx], Zb)
    finite = boots[np.isfinite(boots)]
    if len(finite) == 0:
        return {
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_clustered": float("nan"),
            "n": int(len(x)),
            "n_clusters": len(clusters),
        }
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "p_clustered": bootstrap_p_two_sided(finite),
        "n": int(len(x)),
        "n_clusters": len(clusters),
    }


def load_p1() -> pd.DataFrame:
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
        # normalize DeepSeek long name if present unmapped
        df.loc[df["model"].str.contains("deepseek", case=False, na=False), "model_short"] = "DeepSeek"
        df["variant"] = df["variant_type"].map(normalize_variant)
        ok = df["rescored_correct"] if "rescored_correct" in df.columns else df.get("verified", "")
        df["ok"] = _is_true(ok)
        parts.append(df)
    return pd.concat(parts, ignore_index=True).drop_duplicates(
        ["family", "problem_id", "variant", "model_short"], keep="last",
    )


def load_contamination() -> pd.DataFrame:
    rows = []
    for fam in FAMILIES:
        path = RAW / f"{fam}_P3_contamination.csv"
        df = pd.read_csv(path, dtype=str).fillna("")
        df["family"] = fam
        df["contamination_score"] = pd.to_numeric(df["contamination_score"], errors="coerce")
        df["difficulty_numeric"] = pd.to_numeric(df.get("difficulty_numeric", pd.Series(dtype=str)), errors="coerce")
        df["max_ngram_length"] = pd.to_numeric(df.get("max_ngram_length", pd.Series(dtype=str)), errors="coerce")
        keep = ["family", "problem_id", "contamination_score", "difficulty_numeric", "max_ngram_length"]
        if "problem_text" in df.columns:
            keep.append("problem_text")
        rows.append(df[keep])
    return pd.concat(rows, ignore_index=True).drop_duplicates(["family", "problem_id"])


def load_covariates() -> pd.DataFrame:
    """Item-level difficulty covariates from banks + K3 FD metrics."""
    rows: list[dict] = []

    # BW: K3 FD + goal clauses; bank text for tokens
    k3 = pd.read_csv(DER / "K3_bw_canonical_w6_instances.csv")
    bw_bank = pd.read_csv(BANK / "question_bank_bw.csv", dtype=str).fillna("")
    bw_can = bw_bank[bw_bank["variant_type"].map(normalize_variant) == "canonical"]
    bw_text = dict(zip(bw_can["problem_id"], bw_can["problem_text"]))
    for r in k3.itertuples(index=False):
        pid = str(r.problem_id)
        text = bw_text.get(pid, "")
        rows.append(
            {
                "family": "BW",
                "problem_id": pid,
                "token_count": _token_count(text),
                "difficulty_proxy": float(r.canonical_fd_optimal_plan_length)
                if pd.notna(r.canonical_fd_optimal_plan_length)
                else float("nan"),
                "difficulty_proxy_name": "fd_optimal_plan_length",
                "size_param": float(r.canonical_n_goal_clauses)
                if pd.notna(r.canonical_n_goal_clauses)
                else float("nan"),
                "size_param_name": "n_goal_clauses",
            }
        )

    # ALGO: bank difficulty_params → size; contam difficulty_numeric preferred
    algo_bank = pd.read_csv(BANK / "question_bank_algo.csv", dtype=str).fillna("")
    algo_can = algo_bank[algo_bank["variant_type"].map(normalize_variant) == "canonical"]
    for r in algo_can.itertuples(index=False):
        pid = str(r.problem_id)
        metrics = extract_algo_metrics(
            r.difficulty_params,
            problem_subtype=r.problem_subtype,
            verifier_function=r.verifier_function,
        )
        # problem size parameter: CC target, SP n_nodes, WIS n_intervals
        if "target" in metrics and metrics.get("n_denominations") is not None:
            size = float(metrics["target"])
            size_name = "cc_target"
        elif metrics.get("n_nodes") is not None:
            size = float(metrics["n_nodes"])
            size_name = "sp_n_nodes"
        elif metrics.get("n_intervals") is not None:
            size = float(metrics["n_intervals"])
            size_name = "wis_n_intervals"
        else:
            size = float("nan")
            size_name = "algo_size_unknown"
        # difficulty proxy: difficulty_numeric from params if present
        try:
            params = json.loads(r.difficulty_params) if str(r.difficulty_params).strip() else {}
        except json.JSONDecodeError:
            params = {}
        diff = params.get("difficulty_numeric")
        rows.append(
            {
                "family": "ALGO",
                "problem_id": pid,
                "token_count": _token_count(r.problem_text),
                "difficulty_proxy": float(diff) if diff is not None and str(diff).strip() != "" else size,
                "difficulty_proxy_name": "difficulty_numeric" if diff is not None else size_name,
                "size_param": float(diff) if diff is not None and str(diff).strip() != "" else size,
                "size_param_name": "difficulty_numeric_as_size",
            }
        )

    # GSM: numeric-literal count as difficulty proxy; no goal-clause analogue
    gsm_bank = pd.read_csv(BANK / "question_bank_gsm.csv", dtype=str).fillna("")
    gsm_can = gsm_bank[gsm_bank["variant_type"].map(normalize_variant) == "canonical"]
    for r in gsm_can.itertuples(index=False):
        pid = str(r.problem_id)
        text = r.problem_text
        rows.append(
            {
                "family": "GSM",
                "problem_id": pid,
                "token_count": _token_count(text),
                "difficulty_proxy": float(_n_numeric_literals(text)),
                "difficulty_proxy_name": "n_numeric_literals",
                "size_param": float(_token_count(text)),  # no separate size; mirrored for schema
                "size_param_name": "token_count_as_size",
            }
        )

    return pd.DataFrame(rows)


def build_frame(p1: pd.DataFrame, contam: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    can = p1[p1["variant"] == "canonical"][["family", "problem_id", "model_short", "ok"]].rename(
        columns={"ok": "canonical_ok"},
    )
    w3 = p1[p1["variant"] == "W3"][["family", "problem_id", "model_short", "ok"]].rename(
        columns={"ok": "w3_ok"},
    )
    merged = can.merge(w3, on=["family", "problem_id", "model_short"], how="inner")

    # item-level canonical pass rate across models (accuracy-based difficulty)
    item_pass = (
        merged.groupby(["family", "problem_id"])["canonical_ok"]
        .mean()
        .rename("item_canonical_pass_rate")
        .reset_index()
    )
    merged = merged.merge(item_pass, on=["family", "problem_id"], how="left")

    # N5 subset: retention given canonical correct
    merged = merged[merged["canonical_ok"]].copy()
    merged["retained_w3"] = merged["w3_ok"].astype(int)
    merged["model_canonical_ok"] = 1  # fixed by filter

    merged = merged.merge(contam, on=["family", "problem_id"], how="left")
    merged = merged[merged["contamination_score"].notna()].copy()
    merged = merged.merge(cov, on=["family", "problem_id"], how="left")

    # Prefer contamination-file difficulty_numeric for ALGO when present
    algo = merged["family"] == "ALGO"
    if algo.any():
        dn = merged.loc[algo, "difficulty_numeric"]
        use = dn.notna()
        merged.loc[algo & use, "difficulty_proxy"] = dn[use].astype(float)
        merged.loc[algo & use, "difficulty_proxy_name"] = "difficulty_numeric"

    merged["cluster_id"] = merged["problem_id"].astype(str)
    algo_mask = merged["family"] == "ALGO"
    if algo_mask.any():
        merged.loc[algo_mask, "cluster_id"] = cluster_ids_for(
            merged.loc[algo_mask, "problem_id"].astype(str).tolist(),
        )
    return merged


def controls_for(sub: pd.DataFrame) -> tuple[np.ndarray | None, list[str], str]:
    """Build covariate matrix; drop zero-variance columns. Return (Z, names, note)."""
    fam = sub["family"].iloc[0]
    candidates: list[tuple[str, pd.Series]] = [
        ("difficulty_proxy", sub["difficulty_proxy"]),
        ("token_count", sub["token_count"]),
        ("item_canonical_pass_rate", sub["item_canonical_pass_rate"]),
    ]
    # BW: separate goal-clause count. ALGO: difficulty_numeric already is the
    # size parameter — do not add heterogeneous subtype scales (target vs nodes).
    # GSM: no separate size facet.
    if fam == "BW":
        candidates.append(("size_param", sub["size_param"]))

    note_parts = [
        "model_canonical_ok omitted (zero variance under N5 canonical_correct_only filter); "
        "item_canonical_pass_rate used as accuracy-based difficulty control",
    ]
    if fam == "ALGO":
        note_parts.append(
            "ALGO size control = difficulty_numeric (bank problem-size parameter); "
            "subtype-raw sizes not pooled"
        )
    names: list[str] = []
    cols: list[np.ndarray] = []
    for name, series in candidates:
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        if np.isfinite(arr).sum() < 5:
            note_parts.append(f"{name} dropped (too few finite)")
            continue
        med = float(np.nanmedian(arr))
        arr = np.where(np.isfinite(arr), arr, med)
        if np.std(arr) < 1e-12:
            note_parts.append(f"{name} dropped (zero variance)")
            continue
        if name == "difficulty_proxy":
            label = str(sub["difficulty_proxy_name"].mode().iloc[0])
        elif name == "size_param":
            label = str(sub["size_param_name"].mode().iloc[0])
        else:
            label = name
        names.append(label)
        cols.append(arr)

    if not cols:
        return None, [], "; ".join(note_parts)
    return np.column_stack(cols), names, "; ".join(note_parts)


def _fmt(x: float | None, nd: int = 4) -> float | str:
    if x is None or x != x:
        return ""
    return round(float(x), nd)


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    p1 = load_p1()
    contam = load_contamination()
    cov = load_covariates()
    frame = build_frame(p1, contam, cov)

    rows: list[dict] = []
    for fam in FAMILIES:
        for model in sorted(frame.loc[frame["family"] == fam, "model_short"].dropna().unique()):
            sub = frame[(frame["family"] == fam) & (frame["model_short"] == model)].copy()
            if sub.empty or len(sub) < 5:
                continue
            clust = "clone_family" if fam == "ALGO" else "problem_id"
            x = sub["contamination_score"].to_numpy(dtype=float)
            y = sub["retained_w3"].to_numpy(dtype=float)
            cids = sub["cluster_id"].astype(str).tolist()

            base = {
                "family": fam,
                "model": model,
                "n": len(sub),
                "n_clusters": sub["cluster_id"].nunique(),
                "subset": "canonical_correct_only",
                "y": "w3_retained",
                "contamination_column": "contamination_score",
                "infinigram_max_window": INFINIGRAM_WINDOW[fam],
                "bootstrap": f"cluster_by_{clust}",
                "n_boot": N_BOOT,
                "seed": SEED,
                "difficulty_proxy_name": sub["difficulty_proxy_name"].iloc[0],
                "size_param_name": sub["size_param_name"].iloc[0],
            }

            if len(np.unique(y)) < 2 or len(np.unique(x)) < 2:
                rows.append(
                    {
                        **base,
                        "raw_rho": "",
                        "raw_ci_low": "",
                        "raw_ci_high": "",
                        "raw_p": "",
                        "partial_rho": "",
                        "partial_ci_low": "",
                        "partial_ci_high": "",
                        "partial_p": "",
                        "delta_partial_minus_raw": "",
                        "controls_used": "",
                        "raw_ci_excludes_zero_positive": False,
                        "partial_ci_excludes_zero_positive": False,
                        "verdict": "insufficient_variation",
                        "note": "insufficient variation in retention or contamination",
                    }
                )
                continue

            Z, ctrl_names, ctrl_note = controls_for(sub)

            raw = cluster_bootstrap_partial(x, y, None, cids, n_boot=N_BOOT, seed=SEED)
            # independent seed offset for partial so CI draws aren't identical stream reuse issues
            part = cluster_bootstrap_partial(x, y, Z, cids, n_boot=N_BOOT, seed=SEED + 1)

            raw_r = raw["estimate"]
            part_r = part["estimate"]
            delta = (part_r - raw_r) if (raw_r == raw_r and part_r == part_r) else float("nan")

            # Did the positive N5 signal survive?
            raw_pos = bool(raw_r == raw_r and raw_r > 0)
            part_pos = bool(part_r == part_r and part_r > 0)
            raw_sig = bool(raw["ci_low"] == raw["ci_low"] and raw["ci_low"] > 0)
            part_sig = bool(part["ci_low"] == part["ci_low"] and part["ci_low"] > 0)

            if raw_sig and part_sig:
                verdict = "positive_survives_controls"
            elif raw_sig and not part_sig:
                verdict = "positive_raw_vanishes_after_controls"
            elif (not raw_sig) and part_sig:
                verdict = "partial_positive_emerges_after_controls"
            elif raw_pos and part_pos and abs(delta) < 0.05:
                verdict = "positive_point_stable_ci_includes_zero"
            elif raw_pos and (not part_pos or (delta == delta and delta < -0.05)):
                verdict = "positive_point_attenuated_or_flipped"
            else:
                verdict = "not_a_positive_cell"

            rows.append(
                {
                    **base,
                    "raw_rho": _fmt(raw_r),
                    "raw_ci_low": _fmt(raw["ci_low"]),
                    "raw_ci_high": _fmt(raw["ci_high"]),
                    "raw_p": _fmt(raw["p_clustered"]),
                    "partial_rho": _fmt(part_r),
                    "partial_ci_low": _fmt(part["ci_low"]),
                    "partial_ci_high": _fmt(part["ci_high"]),
                    "partial_p": _fmt(part["p_clustered"]),
                    "delta_partial_minus_raw": _fmt(delta),
                    "controls_used": "|".join(ctrl_names),
                    "raw_ci_excludes_zero_positive": raw_sig,
                    "partial_ci_excludes_zero_positive": part_sig,
                    "verdict": verdict,
                    "note": ctrl_note,
                }
            )

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(out)} rows)")
    cols = [
        "family", "model", "n", "raw_rho", "raw_ci_low", "raw_ci_high",
        "partial_rho", "partial_ci_low", "partial_ci_high", "delta_partial_minus_raw",
        "controls_used", "verdict",
    ]
    print(out[cols].to_string(index=False))
    print("\n--- Focus: previously positive N5 cells ---")
    focus = out[
        out.apply(
            lambda r: (r["family"], r["model"])
            in {
                ("ALGO", "Claude"),
                ("ALGO", "Gemini"),
                ("BW", "DeepSeek"),
                ("BW", "o4-mini"),
            },
            axis=1,
        )
    ]
    print(focus[cols].to_string(index=False))


if __name__ == "__main__":
    main()
