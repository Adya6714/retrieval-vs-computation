#!/usr/bin/env python3
"""Generate ALGO Probe 3 figures (red/coral family)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

np.random.seed(42)

RESULTS_DIR = Path("results")
FIG_DIR = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

P3_PRIMARY = "#B91C1C"
P3_ACCENT = "#EF4444"
GREY = "#9CA3AF"

MODEL_ORDER = [
    "anthropic/claude-sonnet-4",
    "openai/gpt-4o",
    "meta-llama/llama-3.1-8b-instruct",
]
MODEL_LABEL = {
    "anthropic/claude-sonnet-4": "Claude 3.7",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama 3.1 8B",
}
SUBTYPE_ORDER = ["coin_change", "shortest_path", "wis"]
SUBTYPE_LABEL = {"coin_change": "CC", "shortest_path": "SP", "wis": "WIS"}


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_triangulation() -> pd.DataFrame:
    p = Path("results/derived/ALGO_P3_triangulation.csv")
    if not p.exists():
        raise FileNotFoundError(p)
    d = pd.read_csv(p, dtype=str).fillna("")
    d = d[d["model"].astype(str).str.lower() != "mock"].copy()
    d["VAR_canonical"] = pd.to_numeric(d["VAR_canonical"], errors="coerce")
    d["instance_contamination_score"] = pd.to_numeric(d["instance_contamination_score"], errors="coerce")
    d["greedy_succeeds_num"] = pd.to_numeric(d["greedy_succeeds"], errors="coerce")
    d["problem_subtype"] = d["problem_subtype"].str.strip().str.lower()
    return d


def _load_cross_family_regression() -> pd.DataFrame:
    reg_path = Path("results/paper/cross_family_regression.csv")
    if not reg_path.exists():
        raise FileNotFoundError(reg_path)
    d = pd.read_csv(reg_path, dtype=str).fillna("")
    if "model" in d.columns:
        d = d[d["model"].astype(str).str.lower() != "mock"].copy()
    for col in ("beta", "ci_lower", "ci_upper"):
        d[col] = pd.to_numeric(d[col], errors="coerce")
    expected_models = {"Claude", "GPT-4o", "Llama"}
    d = d[d["model"].isin(expected_models)].copy()
    d["family"] = d["family"].astype(str).str.upper()
    if len(d) != 9:
        raise ValueError(f"Expected 9 rows in cross_family_regression.csv, got {len(d)}")
    return d


def plot_regression_forest(tri: pd.DataFrame) -> None:
    _ = tri  # kept for backward compatibility of call signature
    df = _load_cross_family_regression()
    family_order = ["BW", "GSM", "ALGO"]
    model_order = ["Claude", "GPT-4o", "Llama"]
    df["family"] = pd.Categorical(df["family"], categories=family_order, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=model_order, ordered=True)
    df = df.sort_values(["family", "model"]).reset_index(drop=True)
    df["label"] = df["family"].astype(str) + " — " + df["model"].astype(str)

    fig, ax = plt.subplots(figsize=(12, max(5, len(df) * 0.38)))
    y = np.arange(len(df))
    colors = {"BW": GREY, "GSM": P3_ACCENT, "ALGO": P3_PRIMARY}

    for yi, (_, row) in zip(y, df.iterrows()):
        fam = str(row["family"])
        beta = row["beta"]
        lo = row["ci_lower"]
        hi = row["ci_upper"]
        color = colors.get(fam, P3_PRIMARY)
        if pd.isna(beta):
            ax.plot(0, yi, marker="D", color=GREY, markersize=5, zorder=3)
            ax.text(0.05, yi, "not estimable", fontsize=8, color=GREY, va="center")
            continue
        ax.scatter(beta, yi, color=color, s=45, zorder=3)
        if pd.notna(lo) and pd.notna(hi):
            ax.hlines(yi, lo, hi, color=color, linewidth=2, alpha=0.9)

    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"])
    ax.set_xlabel("Coefficient on instance_contamination_score")
    ax.set_title("Figure 8 — Regression coefficient forest plot (95% bootstrap CI)")
    _save(fig, "ALGO_P3_FIG_08_regression_forest")


def main() -> None:
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 9})
    tri = _load_triangulation()
    plot_regression_forest(tri)
    print("Generated ALGO Probe 3 figures in results/figures/")


if __name__ == "__main__":
    main()
