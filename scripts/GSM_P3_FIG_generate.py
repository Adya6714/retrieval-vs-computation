#!/usr/bin/env python3
"""Generate GSM Probe 3 figures."""

from __future__ import annotations

import re
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.behavioral.css import compute_css
from probes.common.stats import bootstrap_ci


RESULTS_DIR = Path("results")
GSM_CONTAM_PATH = Path("results/raw/GSM_P3_contamination.csv")
GSM_MECH_PATH = Path("results/raw/GSM_P3_mechanistic.csv")
GSM_TRIANG_PATHS = {
    "anthropic/claude-sonnet-4": Path("results/derived/GSM_P3_triangulation_per_instance_claude.csv"),
    "openai/gpt-4o": Path("results/derived/GSM_P3_triangulation_per_instance_gpt4o.csv"),
}
GSM_P1_BEHAVIORAL_PATHS = [
    Path("results/raw/GSM_P1_behavioral_claude.csv"),
    Path("results/raw/GSM_P1_behavioral_gpt4o.csv"),
    Path("results/raw/GSM_P1_behavioral_llama.csv"),
]
GSM_P1_CSS_FALLBACK = Path("results/paper/GSM_P1_RES_css.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

P3_PRIMARY = "#B91C1C"
HIGH_COLOR = "#EF4444"
MED_COLOR = "#3B82F6"


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_gsm_per_problem_css(model_name: str) -> pd.DataFrame:
    """Per-problem CSS for scatter plots.

    ``results/derived/GSM_P1_metrics.csv`` is aggregate VAR/W6_Gap only (no problem_id).
    Prefer triangulation ``css``; else ``results/paper/GSM_P1_RES_css.csv`` or P1 behavioral sweeps.
    """
    metrics_path = Path("results/deprecated/GSM_P1_metrics.csv")
    if metrics_path.exists():
        m = pd.read_csv(metrics_path, dtype=str).fillna("")
        if {"problem_id", "css", "model"}.issubset(m.columns):
            css = m[m["model"].astype(str).str.strip() == model_name]
            out = css[["problem_id", "css"]].copy()
            out["css"] = pd.to_numeric(out["css"], errors="coerce")
            if "contamination_pole" in css.columns:
                out["contamination_pole"] = css["contamination_pole"]
            return out.drop_duplicates("problem_id")

    css_models = {model_name, "anthropic/claude-3.7-sonnet" if model_name == "anthropic/claude-sonnet-4" else model_name}
    if GSM_P1_CSS_FALLBACK.exists():
        css = pd.read_csv(GSM_P1_CSS_FALLBACK, dtype=str).fillna("")
        css = css[css["model"].astype(str).str.strip().isin(css_models)]
        if {"problem_id", "css"}.issubset(css.columns):
            out = css[["problem_id", "css"]].copy()
            if "contamination_pole" in css.columns:
                out["contamination_pole"] = css["contamination_pole"]
            out["css"] = pd.to_numeric(out["css"], errors="coerce")
            return out.drop_duplicates("problem_id")

    bank = pd.read_csv("data/problems/question_bank_gsm.csv", dtype=str).fillna("")
    bank = bank[bank["variant_type"].astype(str).str.strip().str.lower() == "canonical"]
    pole = bank[["problem_id", "contamination_pole"]].drop_duplicates("problem_id")

    frames = []
    for p in GSM_P1_BEHAVIORAL_PATHS:
        if not p.exists() or p.stat().st_size <= 200:
            continue
        frames.append(pd.read_csv(p, dtype=str).fillna(""))
    if not frames:
        return pd.DataFrame(columns=["problem_id", "css", "contamination_pole"])

    beh = pd.concat(frames, ignore_index=True)
    beh = beh[beh["model"].astype(str).str.strip() == model_name].copy()
    rows: list[dict] = []
    for pid, grp in beh.groupby("problem_id"):
        can = grp[grp["variant_type"].astype(str).str.strip().str.lower() == "canonical"]
        if can.empty:
            continue
        canonical_answer = str(can.iloc[0].get("correct_answer", ""))
        family = str(can.iloc[0].get("problem_family", "arithmetic_reasoning"))
        variants = []
        for _, r in grp.iterrows():
            vt = str(r.get("variant_type", "")).strip()
            if vt.lower() in ("canonical", "w5"):
                continue
            variants.append(
                {
                    "variant_type": vt,
                    "model_answer": str(r.get("raw_response", "")),
                    "correct_answer": str(r.get("correct_answer", "")),
                    "problem_text": str(r.get("problem_text", "")),
                }
            )
        try:
            css_dict = compute_css(str(pid), canonical_answer, variants, family)
        except ValueError:
            continue
        if css_dict.get("css") is None:
            continue
        rows.append({"problem_id": str(pid), "css": css_dict["css"]})
    if not rows:
        return pd.DataFrame(columns=["problem_id", "css", "contamination_pole"])
    out = pd.DataFrame(rows)
    return out.merge(pole, on="problem_id", how="left")


def _load_css_contam(model_name: str, triang_path: Path) -> pd.DataFrame:
    if not triang_path.exists():
        return pd.DataFrame()
    tri = pd.read_csv(triang_path, dtype=str).fillna("")
    tri["problem_id"] = tri["problem_id"].astype(str).str.strip()
    if "behavioral_model" in tri.columns:
        tri = tri[tri["behavioral_model"].astype(str).str.strip() == model_name].copy()
    tri = tri.drop_duplicates("problem_id")

    if "css" in tri.columns and tri["css"].astype(str).str.strip().ne("").any():
        css = tri[["problem_id", "css"]].copy()
        css["css"] = pd.to_numeric(css["css"], errors="coerce")
    else:
        css = _load_gsm_per_problem_css(model_name)

    if not GSM_CONTAM_PATH.exists():
        return pd.DataFrame()
    contam = pd.read_csv(GSM_CONTAM_PATH, dtype=str)[["problem_id", "contamination_score"]]
    contam["problem_id"] = contam["problem_id"].astype(str).str.strip()
    contam["contamination_score"] = pd.to_numeric(contam["contamination_score"], errors="coerce")

    df = tri[["problem_id"]].drop_duplicates().merge(css, on="problem_id", how="left").merge(
        contam, on="problem_id", how="left"
    )
    if "contamination_pole" not in df.columns and "contamination_pole" in css.columns:
        pass
    elif "contamination_pole" not in df.columns:
        bank = pd.read_csv("data/problems/question_bank_gsm.csv", dtype=str).fillna("")
        pole = bank[bank["variant_type"].astype(str).str.strip().str.lower() == "canonical"][
            ["problem_id", "contamination_pole"]
        ].drop_duplicates("problem_id")
        df = df.merge(pole, on="problem_id", how="left")
    return df.dropna(subset=["css", "contamination_score"])


def _bootstrap_regression_band(x: np.ndarray, y: np.ndarray, xs: np.ndarray, n_boot: int = 1000) -> tuple[np.ndarray, np.ndarray]:
    preds = []
    n = len(x)
    for _ in range(n_boot):
        idx = np.random.randint(0, n, n)
        xb = x[idx]
        yb = y[idx]
        if np.unique(xb).size < 2:
            continue
        b1, b0 = np.polyfit(xb, yb, 1)
        preds.append(b1 * xs + b0)
    if not preds:
        return np.full_like(xs, np.nan), np.full_like(xs, np.nan)
    arr = np.vstack(preds)
    return np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)


def fig_contamination_scatter() -> None:
    models = [
        ("anthropic/claude-sonnet-4", GSM_TRIANG_PATHS["anthropic/claude-sonnet-4"]),
        ("openai/gpt-4o", GSM_TRIANG_PATHS["openai/gpt-4o"]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (model, path) in zip(axes, models):
        if not path.exists():
            ax.axis("off")
            continue
        df = _load_css_contam(model, path)
        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            ax.axis("off")
            continue
        colors = df["contamination_pole"].astype(str).str.lower().map({"high": HIGH_COLOR, "medium": MED_COLOR}).fillna(P3_PRIMARY)
        x = df["contamination_score"].to_numpy(float)
        y = df["css"].to_numpy(float)
        ax.scatter(x, y, c=colors, alpha=0.85, edgecolor="black", linewidth=0.3)

        lr = stats.linregress(x, y)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = lr.slope * xs + lr.intercept
        lo, hi = _bootstrap_regression_band(x, y, xs)
        ax.plot(xs, ys, color=P3_PRIMARY, linewidth=2)
        ax.fill_between(xs, lo, hi, color=P3_PRIMARY, alpha=0.18)
        ax.set_title(model)
        ax.set_xlabel("Contamination score")
        ax.text(
            0.03,
            0.95,
            f"R²={lr.rvalue**2:.3f}\np={lr.pvalue:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            fontsize=9,
        )
    axes[0].set_ylabel("CSS")
    fig.suptitle("Contamination vs CSS (GSM)")
    _save(fig, "GSM_P3_FIG_contamination_scatter")


def fig_crystallization_layer() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not GSM_MECH_PATH.exists():
        ax.text(0.5, 0.5, "No mechanistic results found", ha="center", va="center")
        ax.axis("off")
        _save(fig, "GSM_P3_FIG_crystallization_layer")
        return

    df = pd.read_csv(GSM_MECH_PATH, dtype=str)
    if "model" in df.columns:
        df = df[df["model"].astype(str).str.lower() != "mock"].copy()
    if "contamination_pole" not in df.columns and "problem_id" in df.columns:
        bank = pd.read_csv("data/problems/question_bank_gsm.csv", dtype=str).fillna("")
        pole = bank[bank["variant_type"].astype(str).str.strip().str.lower() == "canonical"][
            ["problem_id", "contamination_pole"]
        ].drop_duplicates("problem_id")
        df = df.merge(pole, on="problem_id", how="left")
    need = {"crystallization_layer", "contamination_pole"}
    if not need.issubset(df.columns):
        ax.text(0.5, 0.5, "Missing crystallization columns", ha="center", va="center")
        ax.axis("off")
        _save(fig, "GSM_P3_FIG_crystallization_layer")
        return
    df["crystallization_layer"] = pd.to_numeric(df["crystallization_layer"], errors="coerce")
    agg = (
        df.dropna(subset=["crystallization_layer"])
        .groupby("contamination_pole", as_index=False)["crystallization_layer"]
        .mean()
    )
    if agg.empty:
        ax.text(0.5, 0.5, "No valid crystallization data", ha="center", va="center")
        ax.axis("off")
        _save(fig, "GSM_P3_FIG_crystallization_layer")
        return
    colors = [HIGH_COLOR if str(p).lower() == "high" else MED_COLOR for p in agg["contamination_pole"]]
    ax.bar(agg["contamination_pole"], agg["crystallization_layer"], color=colors)
    ax.set_ylabel("Mean crystallization layer")
    ax.set_title("Crystallization Layer by Contamination Pole")
    _save(fig, "GSM_P3_FIG_crystallization_layer")


def _parse_regression(path: Path) -> tuple[float, float] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    m = re.search(
        r"contamination_score\s+(-?\d+(?:\.\d+)?)\s+(\d+(?:\.\d+)?)",
        text,
    )
    if not m:
        return None
    coef = float(m.group(1))
    stderr = float(m.group(2))
    return coef, stderr


def fig_cross_family_comparison() -> None:
    pairs = [
        ("Claude", RESULTS_DIR / "BW_P3_RES_contamination_regression_claude37.txt", RESULTS_DIR / "GSM_P3_RES_contamination_regression_claude.txt"),
        ("GPT-4o", RESULTS_DIR / "BW_P3_RES_contamination_regression_gpt4o.txt", RESULTS_DIR / "GSM_P3_RES_contamination_regression_gpt4o.txt"),
    ]
    rows = []
    for label, bw_path, gsm_path in pairs:
        bw = _parse_regression(bw_path)
        gsm = _parse_regression(gsm_path)
        if bw:
            rows.append((label, "BW", bw[0], 1.96 * bw[1]))
        if gsm:
            rows.append((label, "GSM", gsm[0], 1.96 * gsm[1]))
    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.text(0.5, 0.5, "No regression files parsed", ha="center", va="center")
        ax.axis("off")
        _save(fig, "GSM_P3_FIG_cross_family_comparison")
        return

    df = pd.DataFrame(rows, columns=["model", "family", "slope", "ci"])
    models = df["model"].unique().tolist()
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for idx, fam in enumerate(["BW", "GSM"]):
        sub = df[df["family"] == fam].set_index("model")
        slopes = [sub.loc[m, "slope"] if m in sub.index else np.nan for m in models]
        cis = [sub.loc[m, "ci"] if m in sub.index else np.nan for m in models]
        ax.bar(
            x + (idx - 0.5) * width,
            slopes,
            width=width,
            yerr=cis,
            capsize=4,
            label=fam,
            color=P3_PRIMARY if fam == "GSM" else "#9CA3AF",
            alpha=0.9,
        )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Contamination regression slope")
    ax.set_title("BW vs GSM contamination slope comparison")
    ax.legend()
    _save(fig, "GSM_P3_FIG_cross_family_comparison")


def main() -> None:
    fig_contamination_scatter()
    fig_crystallization_layer()
    fig_cross_family_comparison()
    print("Generated GSM Probe 3 figures in figures/")


if __name__ == "__main__":
    main()
