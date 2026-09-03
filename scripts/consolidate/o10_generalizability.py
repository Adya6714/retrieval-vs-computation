#!/usr/bin/env python3
"""O10: Generalizability-theory variance decomposition on Probe 1 grid.

Crossed random-effects (item × variant × model) via balanced ANOVA moments
estimators (classical G-theory; residual absorbs item:variant:model).
Binary accuracy uses a linear probability model (Gaussian moments); a full
crossed logistic GLMM is deferred as often unidentified with n=1 per cell.

Also: EFA on item×variant accuracy; parallel continuous analysis when
results/raw/O5_teacher_forced_likelihood.csv exists.

Designs:
  - max_cells: prefers idle variants W1/W2/W4/W5, then ≥3 models, then cell count
  - full_7_variants: force all 7 variants; largest model subset
Primary: full_7 if n_items≥20 else max_cells (ALGO W5 sparsity).
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.exclusions import filter_excluded  # noqa: E402
from probes.common.variants import normalize_variant  # noqa: E402

DER = REPO_ROOT / "results" / "derived"
RAW = REPO_ROOT / "results" / "raw"
FIG = REPO_ROOT / "paper" / "figures"

OUT_VC = DER / "O10_variance_components.csv"
OUT_G = DER / "O10_generalizability_coefficients.csv"
OUT_EFA = DER / "O10_factor_structure.csv"
OUT_PLOT = FIG / "fig_o10_variance_components.pdf"
OUT_HYP = DER / "O10_hypothesis_tests.csv"
O5_PATH = RAW / "O5_teacher_forced_likelihood.csv"

PAPER_MODELS = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}
VARIANTS = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]
# Surface-form facet used for idle-cell emphasis (all variants still in the full model)
SURFACE_VARIANTS = ["W1", "W2", "W3", "W4", "W5", "W6"]
FAMILIES = ["ALGO", "BW", "GSM"]
N_BOOT = 500
SEED = 42



def _is_true(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_p1_tensor(family: str) -> pd.DataFrame:
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
        df["y"] = _is_true(ok).astype(float)
        df["model"] = df["model"].map(PAPER_MODELS)
        df = df[df["model"].isin(PAPER_MODELS.values())]
        df = df[df["variant"].isin(VARIANTS)]
        parts.append(df[["problem_id", "variant", "model", "y"]])
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(["problem_id", "variant", "model"], keep="last")
    out = out.rename(columns={"problem_id": "item"})
    return out


def load_o5_tensor(family: str) -> pd.DataFrame:
    if not O5_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(O5_PATH, dtype=str).fillna("")
    # flexible column names
    fam_col = "family" if "family" in df.columns else None
    if fam_col:
        df = df[df[fam_col].astype(str).str.upper() == family]
    df["variant"] = df.get("variant", df.get("variant_type", pd.Series(dtype=str))).map(
        lambda v: normalize_variant(v) if str(v).strip() else "",
    )
    df["model"] = df["model"].map(
        lambda m: PAPER_MODELS.get(m, m.split("/")[-1] if "/" in str(m) else m),
    )
    # map HF names to short
    remap = {
        "Qwen2.5-1.5B-Instruct": "Qwen-1.5B",
        "Qwen2.5-3B-Instruct": "Qwen-3B",
        "Llama-3.1-8B-Instruct": "Llama",
        "meta-llama/Llama-3.1-8B-Instruct": "Llama",
    }
    df["model"] = df["model"].replace(remap)
    df["item"] = df["problem_id"].astype(str)
    df["y"] = pd.to_numeric(df["mean_logprob"], errors="coerce")
    df = df[df["variant"].isin(VARIANTS) & df["y"].notna()]
    return df[["item", "variant", "model", "y"]].drop_duplicates(
        ["item", "variant", "model"], keep="last",
    )


def make_balanced(df: pd.DataFrame, variants: list[str] | None = None) -> pd.DataFrame:
    """Largest complete item×variant×model subtensor (every cell present).

    Prefers designs that retain idle surface variants (W1/W2/W4/W5), then
    maximizes n_items × n_models × n_variants.
    """
    variants = variants or VARIANTS
    sub = df[df["variant"].isin(variants)].copy()
    if sub.empty:
        return sub
    models_all = sorted(sub["model"].unique())
    idle_core = {"W1", "W2", "W4", "W5"}  # O4 idle cells

    def complete_items(model_set: list[str], var_set: list[str]) -> list[str]:
        s = sub[sub["model"].isin(model_set) & sub["variant"].isin(var_set)]
        ok = []
        for item, g in s.groupby("item"):
            pairs = set(zip(g["variant"], g["model"]))
            if all((v, m) in pairs for v in var_set for m in model_set):
                ok.append(item)
        return ok

    from itertools import combinations

    # Candidate variant sets, ranked by idle-cell priority
    var_options = [
        variants,  # all 7
        [v for v in variants if v != "W6"],  # keep W5, drop sparse W6
        [v for v in variants if v not in ("W5", "W6")],  # drop both sparse
        [v for v in variants if v != "W5"],  # keep W6, drop W5
    ]
    seen_vs: set[tuple[str, ...]] = set()
    uniq_var_options: list[list[str]] = []
    for vs in var_options:
        key = tuple(vs)
        if len(vs) >= 4 and key not in seen_vs:
            seen_vs.add(key)
            uniq_var_options.append(vs)

    candidates: list[tuple[float, list[str], list[str], list[str]]] = []
    for vs in uniq_var_options:
        idle_n = len(idle_core & set(vs))
        for k in range(len(models_all), 1, -1):
            for model_set in combinations(models_all, k):
                ms = list(model_set)
                items = complete_items(ms, vs)
                if len(items) < 8:
                    continue
                n_cells = len(items) * len(ms) * len(vs)
                # Hard priority: idle coverage, then ≥3 models, then cell count
                score = idle_n * 1_000_000 + (100_000 if len(ms) >= 3 else 0) + n_cells
                candidates.append((float(score), ms, vs, items))

    if not candidates:
        return pd.DataFrame(columns=sub.columns)

    candidates.sort(key=lambda t: t[0], reverse=True)
    _, ms, vs, items = candidates[0]
    out = sub[sub["item"].isin(items) & sub["model"].isin(ms) & sub["variant"].isin(vs)]
    out = out.drop_duplicates(["item", "variant", "model"])
    return out


def make_balanced_full_variants(df: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity: force all VARIANTS; largest model subset with ≥8 complete items."""
    from itertools import combinations

    sub = df[df["variant"].isin(VARIANTS)].copy()
    if sub.empty:
        return sub
    models_all = sorted(sub["model"].unique())
    best = None
    for k in range(len(models_all), 1, -1):
        for model_set in combinations(models_all, k):
            ms = list(model_set)
            s = sub[sub["model"].isin(ms)]
            items = []
            for item, g in s.groupby("item"):
                pairs = set(zip(g["variant"], g["model"]))
                if all((v, m) in pairs for v in VARIANTS for m in ms):
                    items.append(item)
            if len(items) < 8:
                continue
            score = len(items) * len(ms) * len(VARIANTS)
            if best is None or score > best[0]:
                best = (score, ms, items)
        if best is not None:
            break
    if best is None:
        return pd.DataFrame(columns=sub.columns)
    _, ms, items = best
    return sub[sub["item"].isin(items) & sub["model"].isin(ms)].drop_duplicates(
        ["item", "variant", "model"]
    )


def df_to_Y(df: pd.DataFrame) -> np.ndarray:
    items = sorted(df["item"].unique())
    variants = sorted(df["variant"].unique())
    models = sorted(df["model"].unique())
    item_ix = {x: i for i, x in enumerate(items)}
    var_ix = {x: i for i, x in enumerate(variants)}
    mod_ix = {x: i for i, x in enumerate(models)}
    Y = np.full((len(items), len(variants), len(models)), np.nan)
    for r in df.itertuples(index=False):
        Y[item_ix[r.item], var_ix[r.variant], mod_ix[r.model]] = r.y
    return Y


def _vc_from_Y(Y: np.ndarray) -> dict[str, float]:
    """Moments estimators for balanced item×variant×model crossed design (n=1)."""
    n_i, n_v, n_m = Y.shape
    if n_i < 3 or n_v < 2 or n_m < 2 or np.isnan(Y).any():
        return {k: float("nan") for k in (
            "item", "variant", "model", "item_model", "variant_model", "item_variant",
            "residual", "total", "n_items", "n_variants", "n_models", "n_obs",
        )}

    mu = float(Y.mean())
    y_i = Y.mean(axis=(1, 2))
    y_v = Y.mean(axis=(0, 2))
    y_m = Y.mean(axis=(0, 1))
    y_im = Y.mean(axis=1)
    y_vm = Y.mean(axis=0)
    y_iv = Y.mean(axis=2)

    ss_i = n_v * n_m * float(np.sum((y_i - mu) ** 2))
    ss_v = n_i * n_m * float(np.sum((y_v - mu) ** 2))
    ss_m = n_i * n_v * float(np.sum((y_m - mu) ** 2))
    ss_im = n_v * float(np.sum((y_im - y_i[:, None] - y_m[None, :] + mu) ** 2))
    ss_vm = n_i * float(np.sum((y_vm - y_v[:, None] - y_m[None, :] + mu) ** 2))
    ss_iv = n_m * float(np.sum((y_iv - y_i[:, None] - y_v[None, :] + mu) ** 2))
    resid = (
        Y
        - y_iv[:, :, None]
        - y_im[:, None, :]
        - y_vm[None, :, :]
        + y_i[:, None, None]
        + y_v[None, :, None]
        + y_m[None, None, :]
        - mu
    )
    ss_e = float(np.sum(resid ** 2))

    df_i, df_v, df_m = n_i - 1, n_v - 1, n_m - 1
    df_im = (n_i - 1) * (n_m - 1)
    df_vm = (n_v - 1) * (n_m - 1)
    df_iv = (n_i - 1) * (n_v - 1)
    df_e = (n_i - 1) * (n_v - 1) * (n_m - 1)

    ms_i, ms_v, ms_m = ss_i / df_i, ss_v / df_v, ss_m / df_m
    ms_im, ms_vm, ms_iv = ss_im / df_im, ss_vm / df_vm, ss_iv / df_iv
    ms_e = ss_e / df_e if df_e > 0 else 0.0

    def pos(x: float) -> float:
        return float(max(0.0, x))

    s_e = pos(ms_e)
    s_im = pos((ms_im - ms_e) / n_v)
    s_vm = pos((ms_vm - ms_e) / n_i)
    s_iv = pos((ms_iv - ms_e) / n_m)
    s_i = pos((ms_i - n_v * s_im - n_m * s_iv - ms_e) / (n_v * n_m))
    s_v = pos((ms_v - n_i * s_vm - n_m * s_iv - ms_e) / (n_i * n_m))
    s_m = pos((ms_m - n_i * s_vm - n_v * s_im - ms_e) / (n_i * n_v))
    total = s_i + s_v + s_m + s_im + s_vm + s_iv + s_e
    return {
        "item": s_i,
        "variant": s_v,
        "model": s_m,
        "item_model": s_im,
        "variant_model": s_vm,
        "item_variant": s_iv,
        "residual": s_e,
        "total": total,
        "n_items": float(n_i),
        "n_variants": float(n_v),
        "n_models": float(n_m),
        "n_obs": float(n_i * n_v * n_m),
    }


def anova_variance_components(df: pd.DataFrame) -> dict[str, float]:
    return _vc_from_Y(df_to_Y(df))


def bootstrap_vcs(df: pd.DataFrame, n_boot: int = N_BOOT, seed: int = SEED) -> dict[str, tuple[float, float, float]]:
    """Item-resample bootstrap of variance components (numpy tensor)."""
    Y0 = df_to_Y(df)
    point = _vc_from_Y(Y0)
    n_i = Y0.shape[0]
    keys = ["item", "variant", "model", "item_model", "variant_model", "item_variant", "residual"]
    boots = {k: [] for k in keys}
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        vc = _vc_from_Y(Y0[rng.choice(n_i, size=n_i, replace=True), :, :])
        for k in keys:
            if vc[k] == vc[k]:
                boots[k].append(vc[k])
    out = {}
    for k in keys:
        arr = np.asarray(boots[k], dtype=float)
        if len(arr) < 20:
            out[k] = (point[k], float("nan"), float("nan"))
        else:
            out[k] = (point[k], float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))
    return out

def g_coefficients(vc: dict[str, float]) -> dict[str, float]:
    """Generalizability coefficients for the surface-form facet."""
    s_i = vc["item"]
    s_v = vc["variant"]
    s_m = vc["model"]
    s_im = vc["item_model"]
    s_vm = vc["variant_model"]
    s_iv = vc["item_variant"]
    s_e = vc["residual"]
    n_v = vc["n_variants"]
    n_m = vc["n_models"]
    total = vc["total"] or 1.0

    # Relative G for items when generalizing over variants & models (one random model draw)
    # G_rel = σ_i / (σ_i + σ_iv/n_v + σ_im/n_m + σ_e/(n_v n_m))
    g_item = s_i / (s_i + s_iv / n_v + s_im / n_m + s_e / (n_v * n_m)) if s_i + s_iv + s_im + s_e > 0 else float("nan")

    # Surface-form contamination of item signal:
    # proportion of item-relevant variance that is pure item vs surface noise
    surface_noise = s_v + s_iv + s_vm
    g_surface = s_i / (s_i + surface_noise) if (s_i + surface_noise) > 0 else float("nan")

    # Absolute phi-like: include all facets
    phi = s_i / total if total > 0 else float("nan")

    return {
        "G_item_over_variant_model": g_item,
        "G_item_vs_surface_noise": g_surface,
        "phi_item_absolute": phi,
        "prop_item": s_i / total,
        "prop_variant": s_v / total,
        "prop_model": s_m / total,
        "prop_item_model": s_im / total,
        "prop_variant_model": s_vm / total,
        "prop_item_variant": s_iv / total,
        "prop_residual": s_e / total,
        "ratio_item_model_over_item": s_im / s_i if s_i > 1e-12 else float("nan"),
        "ratio_variant_model_over_variant": s_vm / s_v if s_v > 1e-12 else float("nan"),
    }


def run_family(
    family: str,
    outcome: str,
    df_raw: pd.DataFrame,
    design: str = "max_cells",
) -> tuple[list[dict], list[dict], list[dict]]:
    vc_rows: list[dict] = []
    g_rows: list[dict] = []
    hyp_rows: list[dict] = []

    if design == "full_7_variants":
        bal = make_balanced_full_variants(df_raw)
    else:
        bal = make_balanced(df_raw, VARIANTS)
    if bal.empty or bal["item"].nunique() < 5:
        print(f"  [{family}/{outcome}/{design}] insufficient balanced cells")
        return vc_rows, g_rows, hyp_rows

    variants_used = sorted(bal["variant"].unique())
    print(
        f"  [{family}/{outcome}/{design}] balanced n_obs={len(bal)} "
        f"items={bal['item'].nunique()} variants={variants_used} "
        f"models={sorted(bal['model'].unique())}"
    )
    boots = bootstrap_vcs(bal, n_boot=N_BOOT, seed=SEED)
    point = anova_variance_components(bal)
    g = g_coefficients(point)

    for comp in ["item", "variant", "model", "item_model", "variant_model", "item_variant", "residual"]:
        est, lo, hi = boots[comp]
        vc_rows.append(
            {
                "family": family,
                "outcome": outcome,
                "design": design,
                "variants_included": "|".join(variants_used),
                "component": comp,
                "variance": round(est, 6),
                "ci_low": round(lo, 6) if lo == lo else "",
                "ci_high": round(hi, 6) if hi == hi else "",
                "proportion": round(est / point["total"], 4) if point["total"] > 0 else "",
                "n_items": int(point["n_items"]),
                "n_variants": int(point["n_variants"]),
                "n_models": int(point["n_models"]),
                "n_obs": int(point["n_obs"]),
                "n_boot": N_BOOT,
                "estimator": "balanced_anova_moments_item_bootstrap",
                "model_formula": "y ~ (1|item)+(1|variant)+(1|model)+(1|item:model)+(1|variant:model)+(1|item:variant)",
                "note": "Gaussian/LPM on binary for G-theory; residual absorbs item:variant:model",
            }
        )

    g_rows.append(
        {
            "family": family,
            "outcome": outcome,
            "design": design,
            "variants_included": "|".join(variants_used),
            **{k: (round(v, 4) if v == v else "") for k, v in g.items()},
            "n_items": int(point["n_items"]),
            "n_variants": int(point["n_variants"]),
            "n_models": int(point["n_models"]),
            "hypothesis_item_model_gt_item": bool(
                g["ratio_item_model_over_item"] == g["ratio_item_model_over_item"]
                and g["ratio_item_model_over_item"] > 1
            ),
            "hypothesis_variant_model_gt_variant": bool(
                g["ratio_variant_model_over_variant"] == g["ratio_variant_model_over_variant"]
                and g["ratio_variant_model_over_variant"] > 1
            ),
        }
    )

    Y0 = df_to_Y(bal)
    rng = np.random.default_rng(SEED)
    r_im, r_vm = [], []
    for _ in range(N_BOOT):
        vc = _vc_from_Y(Y0[rng.choice(Y0.shape[0], size=Y0.shape[0], replace=True), :, :])
        gg = g_coefficients(vc)
        if gg["ratio_item_model_over_item"] == gg["ratio_item_model_over_item"]:
            r_im.append(gg["ratio_item_model_over_item"])
        if gg["ratio_variant_model_over_variant"] == gg["ratio_variant_model_over_variant"]:
            r_vm.append(gg["ratio_variant_model_over_variant"])

    def _ratio_row(name: str, point_ratio: float, samples: list[float], claim: str) -> dict:
        arr = np.asarray(samples, dtype=float)
        lo = float(np.percentile(arr, 2.5)) if len(arr) else float("nan")
        hi = float(np.percentile(arr, 97.5)) if len(arr) else float("nan")
        p_gt1 = float(np.mean(arr > 1)) if len(arr) else float("nan")
        supported = bool(point_ratio > 1 and lo > 1) if lo == lo else bool(point_ratio > 1)
        return {
            "family": family,
            "outcome": outcome,
            "design": design,
            "hypothesis": name,
            "ratio": round(point_ratio, 4) if point_ratio == point_ratio else "",
            "ci_low": round(lo, 4) if lo == lo else "",
            "ci_high": round(hi, 4) if hi == hi else "",
            "boot_frac_ratio_gt_1": round(p_gt1, 4) if p_gt1 == p_gt1 else "",
            "supported": supported,
            "claim": claim,
        }

    hyp_rows.append(
        _ratio_row(
            "item_model_gt_item",
            g["ratio_item_model_over_item"],
            r_im,
            "If yes: fragility is not an item property (connects to shared-hard-item counts).",
        )
    )
    hyp_rows.append(
        _ratio_row(
            "variant_model_gt_variant",
            g["ratio_variant_model_over_variant"],
            r_vm,
            "If yes: fragility is not a perturbation property (connects to Kendall W discordance).",
        )
    )
    both = hyp_rows[-1]["supported"] and hyp_rows[-2]["supported"]
    hyp_rows.append(
        {
            "family": family,
            "outcome": outcome,
            "design": design,
            "hypothesis": "both_hold_scalar_robustness_not_generalizable",
            "ratio": "",
            "ci_low": "",
            "ci_high": "",
            "boot_frac_ratio_gt_1": "",
            "supported": both,
            "claim": "If both hold, a single scalar robustness score is not a generalizable measurement.",
        }
    )
    return vc_rows, g_rows, hyp_rows



def run_efa(family: str, df_raw: pd.DataFrame) -> list[dict]:
    """EFA on item × variant matrix (accuracy averaged over models)."""
    sub = df_raw[df_raw["variant"].isin(SURFACE_VARIANTS)].copy()
    # mean over models
    mat = sub.groupby(["item", "variant"])["y"].mean().unstack("variant")
    mat = mat.reindex(columns=SURFACE_VARIANTS)
    mat = mat.dropna()
    if len(mat) < 10:
        return [{
            "family": family,
            "n_items": len(mat),
            "note": "insufficient items for EFA",
        }]

    X = mat.to_numpy(dtype=float)
    # correlation eigenvalues
    corr = np.corrcoef(X, rowvar=False)
    evals = np.linalg.eigvalsh(corr)[::-1]
    # parallel analysis null: random normal same shape
    rng = np.random.default_rng(SEED)
    null_evals = []
    for _ in range(200):
        R = np.random.default_rng(rng.integers(1e9)).normal(size=X.shape)
        null_evals.append(np.linalg.eigvalsh(np.corrcoef(R, rowvar=False))[::-1])
    null_mean = np.mean(null_evals, axis=0)
    n_factors = int(sum(evals[i] > null_mean[i] for i in range(len(evals))))
    n_factors = max(1, min(n_factors, len(SURFACE_VARIANTS) - 1))

    fa = FactorAnalysis(n_components=n_factors, random_state=SEED, max_iter=1000)
    fa.fit(X)
    loadings = fa.components_.T  # variants × factors  (sklearn FA: components_ is factors × features)
    # Actually sklearn FactorAnalysis.components_ shape (n_components, n_features)
    loadings = fa.components_.T

    rows = []
    for i, var in enumerate(SURFACE_VARIANTS):
        row = {
            "family": family,
            "variant": var,
            "n_items": len(mat),
            "n_factors_parallel": n_factors,
            "eigenvalue": round(float(evals[i]), 4) if i < len(evals) else "",
            "parallel_null_eigenvalue": round(float(null_mean[i]), 4) if i < len(null_mean) else "",
            "one_factor_interpretation": "one_factor" if n_factors == 1 else "multi_factor",
        }
        for f in range(n_factors):
            row[f"loading_F{f+1}"] = round(float(loadings[i, f]), 4)
        # communalities approx from loadings
        row["communality"] = round(float(np.sum(loadings[i, :] ** 2)), 4)
        rows.append(row)

    # summary row
    rows.append(
        {
            "family": family,
            "variant": "SUMMARY",
            "n_items": len(mat),
            "n_factors_parallel": n_factors,
            "eigenvalue": "",
            "parallel_null_eigenvalue": "",
            "one_factor_interpretation": (
                "W1-W6 behave as ONE factor — interchangeable surface variants"
                if n_factors == 1
                else f"W1-W6 span {n_factors} factors — NOT interchangeable"
            ),
            "communality": "",
            "note": f"eigenvalues={np.round(evals,3).tolist()}",
        }
    )
    return rows


def plot_variance(vc_df: pd.DataFrame) -> None:
    if "is_primary" in vc_df.columns:
        binary = vc_df[(vc_df["outcome"] == "binary_accuracy") & (vc_df["is_primary"])].copy()
    else:
        binary = vc_df[(vc_df["outcome"] == "binary_accuracy") & (vc_df["design"] == "max_cells")].copy()
    if binary.empty:
        binary = vc_df[vc_df["outcome"] == "binary_accuracy"].copy()
    if binary.empty:
        return
    comps = ["item", "variant", "model", "item_model", "variant_model", "item_variant", "residual"]
    families = [f for f in FAMILIES if f in set(binary["family"])]
    fig, axes = plt.subplots(1, len(families), figsize=(4.2 * len(families), 4.5), sharey=True)
    if len(families) == 1:
        axes = [axes]
    colors = {
        "item": "#4C78A8",
        "variant": "#F58518",
        "model": "#54A24B",
        "item_model": "#E45756",
        "variant_model": "#72B7B2",
        "item_variant": "#B279A2",
        "residual": "#9D755D",
    }
    for ax, fam in zip(axes, families):
        sub = binary[binary["family"] == fam].set_index("component")
        props = [float(sub.loc[c, "proportion"]) if c in sub.index else 0 for c in comps]
        ax.bar(range(len(comps)), props, color=[colors[c] for c in comps])
        ax.set_xticks(range(len(comps)))
        ax.set_xticklabels(comps, rotation=45, ha="right", fontsize=8)
        n_i = int(sub.iloc[0]["n_items"]) if len(sub) else 0
        n_v = int(sub.iloc[0]["n_variants"]) if len(sub) else 0
        n_m = int(sub.iloc[0]["n_models"]) if len(sub) else 0
        design = sub.iloc[0]["design"] if "design" in sub.columns else ""
        ax.set_title(f"{fam}\n(I={n_i}, V={n_v}, M={n_m})")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Proportion of variance" if fam == families[0] else "")
    fig.suptitle("O10 Probe 1 G-theory variance components (binary accuracy)", y=1.02)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, bbox_inches="tight")
    fig.savefig(OUT_PLOT.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PLOT}")


def try_logistic_note(family: str, bal: pd.DataFrame) -> str:
    """Document logistic GLMM status; primary G-theory remains ANOVA moments."""
    if bal.empty:
        return "no_balanced_design"
    return (
        "logistic_full_crossed_deferred; primary_estimator=anova_moments_on_binary_LPM; "
        "crossed logistic MixedLM often unidentified with n=1 per cell"
    )


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    all_vc: list[dict] = []
    all_g: list[dict] = []
    all_hyp: list[dict] = []
    all_efa: list[dict] = []

    o5_missing = not O5_PATH.exists()
    if o5_missing:
        print(f"WARNING: {O5_PATH} not found — continuous O5 arm skipped (run Colab O5 first).")

    for family in FAMILIES:
        print(f"\n=== {family} binary ===")
        raw = load_p1_tensor(family)
        if raw.empty:
            print("  no P1 data")
            continue
        bal = make_balanced(raw)
        note = try_logistic_note(family, bal)
        for design in ("max_cells", "full_7_variants"):
            vc, g, hyp = run_family(family, "binary_accuracy", raw, design=design)
            for row in vc:
                row["logistic_note"] = note
            all_vc.extend(vc)
            all_g.extend(g)
            all_hyp.extend(hyp)
        all_efa.extend(run_efa(family, raw))

        if not o5_missing:
            print(f"\n=== {family} O5 mean_logprob ===")
            o5 = load_o5_tensor(family)
            if o5.empty:
                print("  O5 present globally but empty for family")
                continue
            for design in ("max_cells", "full_7_variants"):
                vc2, g2, hyp2 = run_family(family, "mean_logprob", o5, design=design)
                all_vc.extend(vc2)
                all_g.extend(g2)
                all_hyp.extend(hyp2)

    vc_df = pd.DataFrame(all_vc)
    g_df = pd.DataFrame(all_g)
    hyp_df = pd.DataFrame(all_hyp)
    efa_df = pd.DataFrame(all_efa)

    # Primary design per family: full_7 when ≥20 complete items (keeps all models),
    # else idle-preferring max_cells (ALGO W5 sparsity).
    primary_design: dict[str, str] = {}
    for fam in FAMILIES:
        full = g_df[(g_df["family"] == fam) & (g_df["design"] == "full_7_variants") & (g_df["outcome"] == "binary_accuracy")]
        if not full.empty and int(full.iloc[0]["n_items"]) >= 20:
            primary_design[fam] = "full_7_variants"
        else:
            primary_design[fam] = "max_cells"

    def _mark_primary(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or "family" not in df.columns:
            return df
        df = df.copy()
        df["is_primary"] = [
            primary_design.get(r.family) == getattr(r, "design", "max_cells") for r in df.itertuples()
        ]
        return df

    vc_df = _mark_primary(vc_df)
    g_df = _mark_primary(g_df)
    hyp_df = _mark_primary(hyp_df)
    print("Primary designs:", primary_design)

    # Structure agreement binary vs continuous (primary design)
    if not o5_missing and len(g_df[g_df["outcome"] == "mean_logprob"]):
        agree_rows = []
        for fam in FAMILIES:
            b = g_df[
                (g_df["family"] == fam)
                & (g_df["outcome"] == "binary_accuracy")
                & (g_df["is_primary"])
            ]
            c = g_df[
                (g_df["family"] == fam)
                & (g_df["outcome"] == "mean_logprob")
                & (g_df["is_primary"])
            ]
            if b.empty or c.empty:
                continue
            for col in [
                "hypothesis_item_model_gt_item",
                "hypothesis_variant_model_gt_variant",
                "G_item_vs_surface_noise",
            ]:
                agree_rows.append(
                    {
                        "family": fam,
                        "metric": col,
                        "binary": b.iloc[0][col],
                        "continuous": c.iloc[0][col],
                        "agree": str(b.iloc[0][col]) == str(c.iloc[0][col]),
                    }
                )
        if agree_rows:
            pd.DataFrame(agree_rows).to_csv(DER / "O10_binary_vs_continuous_agreement.csv", index=False)

    vc_df.to_csv(OUT_VC, index=False)
    g_df.to_csv(OUT_G, index=False)
    efa_df.to_csv(OUT_EFA, index=False)
    hyp_df.to_csv(OUT_HYP, index=False)
    plot_variance(vc_df)

    print(f"\nWrote {OUT_VC}")
    print(f"Wrote {OUT_G}")
    print(f"Wrote {OUT_EFA}")
    print(f"Wrote {OUT_HYP}")
    print("\n--- Variance proportions (binary, primary) ---")
    if not vc_df.empty:
        prim = vc_df[(vc_df["outcome"] == "binary_accuracy") & (vc_df["is_primary"])]
        piv = prim.pivot(index="component", columns="family", values="proportion")
        print(piv.to_string())
    print("\n--- Hypotheses (primary) ---")
    print(hyp_df[hyp_df["is_primary"]].to_string(index=False))
    print("\n--- EFA summary ---")
    print(efa_df[efa_df["variant"] == "SUMMARY"][
        ["family", "n_factors_parallel", "one_factor_interpretation"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
