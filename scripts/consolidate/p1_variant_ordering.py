#!/usr/bin/env python3
"""N4: P1 variant difficulty ordering stability via Kendall's W.

Null: independently permute each ranker's ranks across items (within-row).
p = (1 + #{W_perm >= W_obs}) / (1 + N_PERM).

Also emits pairwise Spearman of W1–W6 accuracy rankings per family.
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

DER = REPO_ROOT / "results" / "derived"
OUT = DER / "P1_variant_ordering_v2.csv"
OUT_PAIR = DER / "P1_variant_ordering_pairwise.csv"
# Keep legacy path as a copy pointer for older docs
OUT_LEGACY = DER / "P1_variant_ordering.csv"

VARIANTS = ["W1", "W2", "W3", "W4", "W5", "W6"]
N_PERM = 10000
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
    """Kendall's W for m rankers x n items (ranks along each row)."""
    m, n = rank_matrix.shape
    if m < 2 or n < 2:
        return float("nan")
    rank_sums = rank_matrix.sum(axis=0)
    mean_sum = rank_sums.mean()
    ss = np.sum((rank_sums - mean_sum) ** 2)
    return float(12 * ss / (m**2 * (n**3 - n)))


def _perm_p_value(
    rank_matrix: np.ndarray,
    observed: float,
    rng: np.random.Generator,
    n_perm: int = N_PERM,
) -> float:
    """Permutation p for H0: no concordance.

    Null: independently permute ranks *within each row* (each ranker's
    assignment of ranks across items). Column-wise permutation would leave
    column sums — and thus W — invariant, yielding p ≡ 1.
    """
    m, n = rank_matrix.shape
    if m < 2 or n < 2 or not np.isfinite(observed):
        return float("nan")
    count = 0
    for _ in range(n_perm):
        perm = np.empty_like(rank_matrix)
        for i in range(m):
            perm[i, :] = rng.permutation(rank_matrix[i, :])
        if _kendall_w(perm) >= observed - 1e-12:
            count += 1
    return float((count + 1) / (n_perm + 1))


def _rank_matrix(sub: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    acc = sub.groupby(["model", "variant"], as_index=False)["ok"].mean()
    models = sorted(acc["model"].unique())
    present_variants = [v for v in VARIANTS if v in set(acc["variant"])]
    mat = np.zeros((len(models), len(present_variants)))
    for i, model in enumerate(models):
        row = acc[acc["model"] == model].set_index("variant")["ok"]
        vals = row.reindex(present_variants).astype(float)
        mat[i, :] = stats.rankdata(-vals.to_numpy(), method="average")
    keep = np.all(np.isfinite(mat), axis=1)
    mat = mat[keep, :]
    models = [m for m, k in zip(models, keep) if k]
    return mat, models, present_variants


def _pairwise_spearman(mat: np.ndarray, labels: list[str], family: str) -> list[dict]:
    rows: list[dict] = []
    m = len(labels)
    for i in range(m):
        for j in range(m):
            if i == j:
                rho, p = 1.0, 0.0
            else:
                rho, p = stats.spearmanr(mat[i, :], mat[j, :])
            rows.append(
                {
                    "family": family,
                    "model_a": labels[i],
                    "model_b": labels[j],
                    "spearman_rho": round(float(rho), 4) if rho == rho else "",
                    "p_value": round(float(p), 4) if p == p else "",
                    "n_variants": mat.shape[1],
                }
            )
    return rows


def validate_synthetic() -> None:
    """Two sanity checks before real analyses."""
    rng = np.random.default_rng(0)
    n_items, m_rankers = 6, 4

    # (a) Perfectly concordant rankers → p near 0
    perfect = np.tile(np.arange(1, n_items + 1, dtype=float), (m_rankers, 1))
    w_perf = _kendall_w(perfect)
    p_perf = _perm_p_value(perfect, w_perf, rng, n_perm=2000)
    print("SYNTHETIC (a) perfect concordance:")
    print(f"  W={w_perf:.4f}  p={p_perf:.4f}  (expect W=1, p near 0)")

    # (b) Independent random rankers → p roughly uniform on [0,1]
    ps = []
    for trial in range(20):
        mat = np.vstack([rng.permutation(n_items) + 1.0 for _ in range(m_rankers)])
        w = _kendall_w(mat)
        ps.append(_perm_p_value(mat, w, rng, n_perm=2000))
    ps_arr = np.asarray(ps)
    print("SYNTHETIC (b) 20 independent random rankers:")
    print(f"  p-values: {[round(float(x), 3) for x in ps_arr]}")
    print(
        f"  mean={ps_arr.mean():.3f}  min={ps_arr.min():.3f}  "
        f"max={ps_arr.max():.3f}  frac_p<0.05={(ps_arr < 0.05).mean():.2f}  "
        f"(expect roughly uniform)"
    )
    if not (w_perf > 0.99 and p_perf < 0.01):
        raise RuntimeError(f"Synthetic (a) failed: W={w_perf}, p={p_perf}")
    if not (0.2 < ps_arr.mean() < 0.8 and ps_arr.min() < 0.3 and ps_arr.max() > 0.7):
        raise RuntimeError(f"Synthetic (b) p-values not roughly uniform: {ps_arr}")
    print("Synthetic validation PASSED.\n")


def main() -> None:
    print("=== Permutation loop (fixed: within-row) ===")
    print(
        "for i in range(m):\n"
        "    perm[i, :] = rng.permutation(rank_matrix[i, :])\n"
        "if _kendall_w(perm) >= observed - 1e-12:\n"
        "    count += 1\n"
        "return (count + 1) / (n_perm + 1)\n"
    )
    validate_synthetic()

    DER.mkdir(parents=True, exist_ok=True)
    p1 = _load_p1()
    p1 = p1[p1["variant"].isin(VARIANTS)].copy()
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    pair_rows: list[dict] = []

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
        pair_rows.extend(_pairwise_spearman(mat, models, fam))

    for model in sorted(p1["model"].unique()):
        sub_m = p1[p1["model"] == model]
        fams = sorted(sub_m["family"].unique())
        if len(fams) < 2:
            continue
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
    out.to_csv(OUT_LEGACY, index=False)
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(OUT_PAIR, index=False)
    print(f"Wrote {OUT} ({len(out)} rows)")
    print(f"Wrote {OUT_PAIR} ({len(pair)} rows)")
    print(out.to_string(index=False))
    print("\nPairwise Spearman (upper triangle, off-diagonal):")
    for fam in sorted(pair["family"].unique()):
        sub = pair[pair["family"] == fam]
        labels = sorted(sub["model_a"].unique())
        print(f"\n[{fam}]")
        # compact matrix
        grid = sub.pivot(index="model_a", columns="model_b", values="spearman_rho")
        grid = grid.reindex(index=labels, columns=labels)
        print(grid.to_string())


if __name__ == "__main__":
    main()
