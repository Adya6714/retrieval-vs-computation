#!/usr/bin/env python3
"""N4: P1 variant difficulty ordering stability via Kendall's W."""

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

DER = REPO_ROOT / "results" / "derived"
OUT = DER / "P1_variant_ordering.csv"

VARIANTS = ["W1", "W2", "W3", "W4", "W5", "W6"]
N_PERM = 5000
SEED = 42


def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


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
        df["variant"] = df["variant_type"].map(normalize_variant)
        ok = df["rescored_correct"] if "rescored_correct" in df.columns else df.get("verified", "")
        df["ok"] = _is_true(ok)
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    return out.drop_duplicates(["family", "problem_id", "variant", "model"], keep="last")


def _kendall_w(rank_matrix: np.ndarray) -> float:
    """Kendall's W for m rankers x n items (1-indexed ranks)."""
    m, n = rank_matrix.shape
    if m < 2 or n < 2:
        return float("nan")
    rank_sums = rank_matrix.sum(axis=0)
    mean_sum = rank_sums.mean()
    ss = np.sum((rank_sums - mean_sum) ** 2)
    return float(12 * ss / (m**2 * (n**3 - n)))


def _perm_p_value(rank_matrix: np.ndarray, observed: float, rng: np.random.Generator) -> float:
    m, n = rank_matrix.shape
    if m < 2 or n < 2 or not np.isfinite(observed):
        return float("nan")
    count = 0
    for _ in range(N_PERM):
        perm = np.empty_like(rank_matrix)
        for j in range(n):
            perm[:, j] = rng.permutation(rank_matrix[:, j])
        if _kendall_w(perm) >= observed - 1e-12:
            count += 1
    return float((count + 1) / (N_PERM + 1))


def _rank_matrix(sub: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    acc = sub.groupby(["model", "variant"], as_index=False)["ok"].mean()
    models = sorted(acc["model"].unique())
    present_variants = [v for v in VARIANTS if v in set(acc["variant"])]
    mat = np.zeros((len(models), len(present_variants)))
    for i, model in enumerate(models):
        row = acc[acc["model"] == model].set_index("variant")["ok"]
        vals = row.reindex(present_variants).astype(float)
        mat[i, :] = stats.rankdata(-vals.to_numpy(), method="average")
    # Drop rankers with missing variant coverage
    keep = np.all(np.isfinite(mat), axis=1)
    mat = mat[keep, :]
    models = [m for m, k in zip(models, keep) if k]
    return mat, models, present_variants


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    p1 = _load_p1()
    p1 = p1[p1["variant"].isin(VARIANTS)].copy()
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []

    for fam in sorted(p1["family"].unique()):
        sub_f = p1[p1["family"] == fam]
        mat, models, variants = _rank_matrix(sub_f)
        w = _kendall_w(mat)
        p = _perm_p_value(mat, w, rng)
        rows.append(
            {
                "analysis": "within_family_across_models",
                "family": fam,
                "model": "--",
                "kendall_W": round(w, 4) if w == w else "",
                "p_value": round(p, 4) if p == p else "",
                "n_rankers": len(models),
                "n_variants": len(variants),
                "variants": "|".join(variants),
                "n_perm": N_PERM,
                "seed": SEED,
            }
        )

    for model in sorted(p1["model"].unique()):
        sub_m = p1[p1["model"] == model]
        fams = sorted(sub_m["family"].unique())
        if len(fams) < 2:
            continue
        # accuracy per (family, variant) for this model
        acc = sub_m.groupby(["family", "variant"], as_index=False)["ok"].mean()
        variants = [v for v in VARIANTS if v in set(acc["variant"])]
        mat = np.zeros((len(fams), len(variants)))
        for i, fam in enumerate(fams):
            row = acc[acc["family"] == fam].set_index("variant")["ok"]
            vals = row.reindex(variants).astype(float)
            mat[i, :] = stats.rankdata(-vals.to_numpy(), method="average")
        keep = np.all(np.isfinite(mat), axis=1)
        mat = mat[keep, :]
        fams = [f for f, k in zip(fams, keep) if k]
        if len(fams) < 2:
            continue
        w = _kendall_w(mat)
        p = _perm_p_value(mat, w, rng)
        rows.append(
            {
                "analysis": "within_model_across_families",
                "family": "--",
                "model": model,
                "kendall_W": round(w, 4) if w == w else "",
                "p_value": round(p, 4) if p == p else "",
                "n_rankers": len(fams),
                "n_variants": len(variants),
                "variants": "|".join(variants),
                "n_perm": N_PERM,
                "seed": SEED,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT} ({len(out)} rows)")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
