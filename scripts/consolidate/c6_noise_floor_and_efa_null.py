#!/usr/bin/env python3
"""C6: Correct O10 G-theory for Bernoulli noise floor; calibrate EFA null.

(a) Residual on binary LPM is bounded below by p(1-p). Simulate null response
    matrices with the same fitted cell probabilities + pure Bernoulli noise,
    run identical ANOVA, subtract the simulated floor from each VC / recompute G.

(b) EFA "one factor" may be an artifact of near-zero item variance. Simulate
    null item×variant matrices with matched item variance + residual but no
    true factor structure; run identical parallel analysis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.consolidate.o10_generalizability import (  # noqa: E402
    FAMILIES,
    SURFACE_VARIANTS,
    VARIANTS,
    _vc_from_Y,
    anova_variance_components,
    df_to_Y,
    g_coefficients,
    load_p1_tensor,
    make_balanced,
    make_balanced_full_variants,
)

DER = REPO_ROOT / "results" / "derived"
OUT_VC = DER / "C6_variance_components_noise_corrected.csv"
OUT_G = DER / "C6_generalizability_noise_corrected.csv"
OUT_EFA = DER / "C6_efa_null_calibration.csv"

N_NULL = 1000
N_BOOT = 500
SEED = 42
VC_KEYS = ["item", "variant", "model", "item_model", "variant_model", "item_variant", "residual"]


def _round(x: float, nd: int = 6):
    if x != x:
        return ""
    return round(float(x), nd)


def primary_design_for(family: str, g_df: pd.DataFrame | None = None) -> str:
    """Match O10 primary rule: full_7 if n_items≥20 else max_cells."""
    if g_df is None:
        g_df = pd.read_csv(DER / "O10_generalizability_coefficients.csv")
    full = g_df[
        (g_df["family"] == family)
        & (g_df["design"] == "full_7_variants")
        & (g_df["outcome"] == "binary_accuracy")
    ]
    if not full.empty and int(full.iloc[0]["n_items"]) >= 20:
        return "full_7_variants"
    return "max_cells"


def balanced_for(family: str, design: str) -> pd.DataFrame:
    raw = load_p1_tensor(family)
    if design == "full_7_variants":
        return make_balanced_full_variants(raw)
    return make_balanced(raw, VARIANTS)


def fitted_cell_probs(Y: np.ndarray) -> np.ndarray:
    """LPM fitted means: projection onto all main + two-way terms (excl. three-way)."""
    mu = float(Y.mean())
    y_i = Y.mean(axis=(1, 2))
    y_v = Y.mean(axis=(0, 2))
    y_m = Y.mean(axis=(0, 1))
    y_im = Y.mean(axis=1)
    y_vm = Y.mean(axis=0)
    y_iv = Y.mean(axis=2)
    fitted = (
        y_iv[:, :, None]
        + y_im[:, None, :]
        + y_vm[None, :, :]
        - y_i[:, None, None]
        - y_v[None, :, None]
        - y_m[None, None, :]
        + mu
    )
    return np.clip(fitted, 1e-6, 1.0 - 1e-6)


def simulate_bernoulli_null_vcs(P: np.ndarray, n_rep: int, seed: int) -> dict[str, np.ndarray]:
    """Return arrays of VC estimates under Y ~ Bern(P)."""
    rng = np.random.default_rng(seed)
    out = {k: np.empty(n_rep, dtype=float) for k in VC_KEYS}
    for r in range(n_rep):
        Y = (rng.random(P.shape) < P).astype(float)
        vc = _vc_from_Y(Y)
        for k in VC_KEYS:
            out[k][r] = vc[k]
    return out


def correct_vc(raw: dict[str, float], null_mean: dict[str, float]) -> dict[str, float]:
    """Subtract Bernoulli-null mean from each component; floor at 0; recompute total."""
    corr = {k: float(max(0.0, raw[k] - null_mean[k])) for k in VC_KEYS}
    corr["total"] = float(sum(corr[k] for k in VC_KEYS))
    for meta in ("n_items", "n_variants", "n_models", "n_obs"):
        corr[meta] = raw[meta]
    return corr


def item_bootstrap_raw_and_corrected(
    Y: np.ndarray,
    null_mean: dict[str, float],
    n_boot: int,
    seed: int,
) -> tuple[dict[str, tuple[float, float, float]], dict[str, tuple[float, float, float]], list[dict]]:
    """Item-resample bootstrap for raw and floor-corrected VCs; also G samples."""
    point_raw = _vc_from_Y(Y)
    point_corr = correct_vc(point_raw, null_mean)
    n_i = Y.shape[0]
    rng = np.random.default_rng(seed)
    raw_boots = {k: [] for k in VC_KEYS}
    corr_boots = {k: [] for k in VC_KEYS}
    g_raw_boots: list[dict] = []
    g_corr_boots: list[dict] = []
    for _ in range(n_boot):
        Yb = Y[rng.choice(n_i, size=n_i, replace=True), :, :]
        vcr = _vc_from_Y(Yb)
        vcc = correct_vc(vcr, null_mean)
        for k in VC_KEYS:
            if vcr[k] == vcr[k]:
                raw_boots[k].append(vcr[k])
            if vcc[k] == vcc[k]:
                corr_boots[k].append(vcc[k])
        g_raw_boots.append(g_coefficients(vcr))
        g_corr_boots.append(g_coefficients(vcc))

    def pack(point: dict, boots: dict) -> dict[str, tuple[float, float, float]]:
        out = {}
        for k in VC_KEYS:
            arr = np.asarray(boots[k], dtype=float)
            if len(arr) < 20:
                out[k] = (point[k], float("nan"), float("nan"))
            else:
                out[k] = (
                    point[k],
                    float(np.percentile(arr, 2.5)),
                    float(np.percentile(arr, 97.5)),
                )
        return out

    return (
        pack(point_raw, raw_boots),
        pack(point_corr, corr_boots),
        [{"raw": g_raw_boots, "corr": g_corr_boots, "point_raw": g_coefficients(point_raw), "point_corr": g_coefficients(point_corr)}],
    )


def _g_ci(samples: list[dict], key: str, point: float) -> tuple[float, float, float]:
    arr = np.asarray([s[key] for s in samples if s[key] == s[key]], dtype=float)
    if len(arr) < 20:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def run_part_a() -> tuple[pd.DataFrame, pd.DataFrame]:
    g_o10 = pd.read_csv(DER / "O10_generalizability_coefficients.csv")
    vc_rows: list[dict] = []
    g_rows: list[dict] = []

    for family in FAMILIES:
        design = primary_design_for(family, g_o10)
        bal = balanced_for(family, design)
        if bal.empty:
            print(f"[{family}] no balanced design")
            continue
        Y = df_to_Y(bal)
        P = fitted_cell_probs(Y)
        bern_cell = P * (1.0 - P)
        print(
            f"[{family}/{design}] Y={Y.shape} mean_p={P.mean():.3f} "
            f"mean_bern_var={bern_cell.mean():.4f} "
            f"(midrange 0.25 reference)"
        )

        null = simulate_bernoulli_null_vcs(P, N_NULL, SEED + hash(family) % 10_000)
        null_mean = {k: float(np.nanmean(null[k])) for k in VC_KEYS}
        null_lo = {k: float(np.nanpercentile(null[k], 2.5)) for k in VC_KEYS}
        null_hi = {k: float(np.nanpercentile(null[k], 97.5)) for k in VC_KEYS}

        raw_ci, corr_ci, g_pack = item_bootstrap_raw_and_corrected(Y, null_mean, N_BOOT, SEED)
        point_raw = _vc_from_Y(Y)
        point_corr = correct_vc(point_raw, null_mean)
        g_info = g_pack[0]

        variants_used = "|".join(sorted(bal["variant"].unique()))
        for k in VC_KEYS:
            raw_est, raw_lo, raw_hi = raw_ci[k]
            c_est, c_lo, c_hi = corr_ci[k]
            vc_rows.append(
                {
                    "family": family,
                    "design": design,
                    "is_primary": True,
                    "variants_included": variants_used,
                    "component": k,
                    "variance_raw": _round(raw_est),
                    "ci_low_raw": _round(raw_lo),
                    "ci_high_raw": _round(raw_hi),
                    "proportion_raw": _round(raw_est / point_raw["total"], 4) if point_raw["total"] > 0 else "",
                    "bernoulli_null_mean": _round(null_mean[k]),
                    "bernoulli_null_ci_low": _round(null_lo[k]),
                    "bernoulli_null_ci_high": _round(null_hi[k]),
                    "variance_corrected": _round(c_est),
                    "ci_low_corrected": _round(c_lo),
                    "ci_high_corrected": _round(c_hi),
                    "proportion_corrected": (
                        _round(c_est / point_corr["total"], 4) if point_corr["total"] > 0 else ""
                    ),
                    "mean_fitted_p": _round(float(P.mean()), 4),
                    "mean_bernoulli_p1mp": _round(float(bern_cell.mean()), 4),
                    "n_items": int(point_raw["n_items"]),
                    "n_variants": int(point_raw["n_variants"]),
                    "n_models": int(point_raw["n_models"]),
                    "n_obs": int(point_raw["n_obs"]),
                    "n_null_sims": N_NULL,
                    "n_boot": N_BOOT,
                    "seed": SEED,
                    "note": (
                        "corrected = max(0, raw − E[VC | Bern(fitted_p)]); "
                        "fitted_p = LPM projection onto mains+two-ways; "
                        "CIs from item bootstrap with fixed null floor"
                    ),
                }
            )

        # G coefficients raw vs corrected
        for label, point_g, samples in [
            ("raw", g_info["point_raw"], g_info["raw"]),
            ("corrected", g_info["point_corr"], g_info["corr"]),
        ]:
            row = {
                "family": family,
                "design": design,
                "is_primary": True,
                "variants_included": variants_used,
                "estimate": label,
                "n_items": int(point_raw["n_items"]),
                "n_variants": int(point_raw["n_variants"]),
                "n_models": int(point_raw["n_models"]),
                "mean_bernoulli_p1mp": _round(float(bern_cell.mean()), 4),
                "residual_raw": _round(point_raw["residual"]),
                "residual_null_mean": _round(null_mean["residual"]),
                "residual_corrected": _round(point_corr["residual"]),
                "n_null_sims": N_NULL,
                "n_boot": N_BOOT,
                "seed": SEED,
            }
            for key in [
                "G_item_over_variant_model",
                "G_item_vs_surface_noise",
                "phi_item_absolute",
                "prop_item",
                "prop_residual",
                "ratio_item_model_over_item",
                "ratio_variant_model_over_variant",
            ]:
                est, lo, hi = _g_ci(samples, key, point_g[key])
                row[key] = _round(est, 4)
                row[f"{key}_ci_low"] = _round(lo, 4)
                row[f"{key}_ci_high"] = _round(hi, 4)
            # Paper headline G
            g_surf = point_g["G_item_vs_surface_noise"]
            row["paper_G"] = _round(g_surf, 4)
            row["paper_G_name"] = "G_item_vs_surface_noise"
            row["note"] = (
                "paper_G = G_item_vs_surface_noise (item / (item+surface_noise)). "
                "If BW corrected G rises substantially above ~0.01, headline changes."
                if label == "corrected"
                else "Uncorrected O10-style G on binary LPM."
            )
            g_rows.append(row)

        print(
            f"  residual raw={point_raw['residual']:.4f} null={null_mean['residual']:.4f} "
            f"corr={point_corr['residual']:.4f} | "
            f"G_surface raw={g_info['point_raw']['G_item_vs_surface_noise']:.4f} "
            f"corr={g_info['point_corr']['G_item_vs_surface_noise']:.4f}"
        )

    return pd.DataFrame(vc_rows), pd.DataFrame(g_rows)


def efa_matrix(df_raw: pd.DataFrame) -> pd.DataFrame:
    sub = df_raw[df_raw["variant"].isin(SURFACE_VARIANTS)].copy()
    mat = sub.groupby(["item", "variant"])["y"].mean().unstack("variant")
    mat = mat.reindex(columns=SURFACE_VARIANTS).dropna()
    return mat


def parallel_n_factors(X: np.ndarray, seed: int, n_null_pa: int = 200) -> tuple[int, np.ndarray, np.ndarray]:
    """Identical to O10 run_efa parallel analysis rule."""
    corr = np.corrcoef(X, rowvar=False)
    # Guard degenerate columns
    if np.isnan(corr).any():
        # jitter for zero-variance columns
        rng = np.random.default_rng(seed)
        X = X + rng.normal(0, 1e-8, size=X.shape)
        corr = np.corrcoef(X, rowvar=False)
    evals = np.linalg.eigvalsh(corr)[::-1]
    rng = np.random.default_rng(seed)
    null_evals = []
    for _ in range(n_null_pa):
        R = np.random.default_rng(rng.integers(1e9)).normal(size=X.shape)
        null_evals.append(np.linalg.eigvalsh(np.corrcoef(R, rowvar=False))[::-1])
    null_mean = np.mean(null_evals, axis=0)
    n_factors = int(sum(evals[i] > null_mean[i] for i in range(len(evals))))
    n_factors = max(1, min(n_factors, X.shape[1] - 1))
    return n_factors, evals, null_mean


def simulate_additive_null_matrix(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Matched item variance + residual, NO common factor beyond additive item/variant means.

    Model: X_ij = μ + α_i + β_j + ε_ij, ε iid. α,β,ε variances matched to OLS fit on X.
    """
    n_i, n_v = X.shape
    mu = float(X.mean())
    item_means = X.mean(axis=1)
    var_means = X.mean(axis=0)
    alpha = item_means - mu
    beta = var_means - mu
    resid = X - item_means[:, None] - var_means[None, :] + mu
    # Match empirical second moments
    s_alpha = float(np.var(alpha, ddof=1)) if n_i > 1 else 0.0
    s_eps = float(np.var(resid, ddof=1)) if resid.size > 1 else 0.0
    # Use empirical alpha/beta shapes (shuffle item effects) + fresh noise — no factor
    alpha_s = rng.permutation(alpha)
    eps = rng.normal(0.0, np.sqrt(max(s_eps, 1e-12)), size=X.shape)
    Xn = mu + alpha_s[:, None] + beta[None, :] + eps
    return Xn


def run_part_b() -> pd.DataFrame:
    rows: list[dict] = []
    for family in FAMILIES:
        raw = load_p1_tensor(family)
        mat = efa_matrix(raw)
        if len(mat) < 10:
            rows.append(
                {
                    "family": family,
                    "n_items": len(mat),
                    "note": "insufficient items for EFA",
                    "verdict": "skipped",
                }
            )
            continue
        X = mat.to_numpy(dtype=float)
        # Observed item / residual variance (additive decomposition)
        item_means = X.mean(axis=1)
        var_means = X.mean(axis=0)
        mu = float(X.mean())
        resid = X - item_means[:, None] - var_means[None, :] + mu
        item_var = float(np.var(item_means, ddof=1))
        resid_var = float(np.var(resid, ddof=1))
        # Total column-wise variance (for reporting)
        col_var_mean = float(np.mean(np.var(X, axis=0, ddof=1)))

        n_obs, evals, null_pa = parallel_n_factors(X, SEED)
        # Also fit 1-factor FA communality mean (diagnostic)
        fa = FactorAnalysis(n_components=1, random_state=SEED, max_iter=1000)
        fa.fit(X)
        mean_comm = float(np.mean(np.sum(fa.components_.T ** 2, axis=1)))

        rng = np.random.default_rng(SEED + 99)
        null_factors = np.empty(N_NULL, dtype=int)
        for i in range(N_NULL):
            Xn = simulate_additive_null_matrix(X, rng)
            nf, _, _ = parallel_n_factors(Xn, int(rng.integers(1e9)))
            null_factors[i] = nf

        frac_one = float(np.mean(null_factors == 1))
        frac_ge_obs = float(np.mean(null_factors >= n_obs))
        # If null usually returns 1 factor too, EFA is uninformative
        uninformative = bool(frac_one >= 0.80 or (n_obs == 1 and frac_one >= 0.5))

        rows.append(
            {
                "family": family,
                "n_items": len(mat),
                "n_variants": X.shape[1],
                "item_variance": _round(item_var, 6),
                "residual_variance_additive": _round(resid_var, 6),
                "mean_column_variance": _round(col_var_mean, 6),
                "observed_n_factors_parallel": n_obs,
                "observed_mean_communality_1factor": _round(mean_comm, 4),
                "observed_eigenvalues": ",".join(f"{e:.4f}" for e in evals),
                "o10_style_normal_null_eigenvalues": ",".join(f"{e:.4f}" for e in null_pa),
                "n_null_sims": N_NULL,
                "null_frac_n_factors_eq_1": _round(frac_one, 4),
                "null_frac_n_factors_ge_observed": _round(frac_ge_obs, 4),
                "null_mean_n_factors": _round(float(null_factors.mean()), 4),
                "null_median_n_factors": int(np.median(null_factors)),
                "verdict": (
                    "EFA_uninformative_null_also_one_factor"
                    if uninformative
                    else "EFA_informative_one_factor_exceeds_structured_null"
                ),
                "paper_statement": (
                    "Do NOT report unidimensionality of W1–W6 as evidence: "
                    "null matrices with matched item variance and residual but no factor "
                    f"structure also yield one factor in {frac_one:.0%} of replicates."
                    if uninformative
                    else (
                        "One-factor result exceeds additive null rate; "
                        "unidimensionality claim is tentatively supported."
                    )
                ),
                "note": (
                    "Null = additive item+variant means + iid residual (no common factor). "
                    "Parallel analysis rule identical to O10 (200 normal PA draws per matrix). "
                    f"O10 reported one_factor for {family}; item_var={item_var:.6f}."
                ),
                "seed": SEED,
            }
        )
        print(
            f"[EFA {family}] item_var={item_var:.5f} resid_var={resid_var:.5f} "
            f"obs_factors={n_obs} null_frac_1={frac_one:.3f} → "
            f"{'UNINFORMATIVE' if uninformative else 'informative'}"
        )

    return pd.DataFrame(rows)


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    print("=== C6(a) Bernoulli noise floor ===")
    vc_df, g_df = run_part_a()
    vc_df.to_csv(OUT_VC, index=False)
    g_df.to_csv(OUT_G, index=False)
    print(f"Wrote {OUT_VC} ({len(vc_df)} rows)")
    print(f"Wrote {OUT_G} ({len(g_df)} rows)")

    print("\n=== C6(b) EFA null calibration ===")
    efa_df = run_part_b()
    efa_df.to_csv(OUT_EFA, index=False)
    print(f"Wrote {OUT_EFA} ({len(efa_df)} rows)")

    print("\n--- Headline G (paper: G_item_vs_surface_noise) ---")
    show = g_df[["family", "estimate", "paper_G", "G_item_vs_surface_noise_ci_low", "G_item_vs_surface_noise_ci_high", "residual_raw", "residual_corrected"]]
    print(show.to_string(index=False))
    print("\n--- EFA verdicts ---")
    print(efa_df[["family", "item_variance", "observed_n_factors_parallel", "null_frac_n_factors_eq_1", "verdict"]].to_string(index=False))


if __name__ == "__main__":
    main()
