#!/usr/bin/env python3
"""Generate GSM Probe 1 figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path("results")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

P1_PRIMARY = "#1D4ED8"
P1_ACCENT = "#3B82F6"
P1_LIGHT = "#DBEAFE"


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load_behavioral() -> pd.DataFrame:
    paths = sorted(RESULTS_DIR.glob("GSM_P1_RES_behavioral_sweep_*.csv"))
    if not paths:
        raise FileNotFoundError("No GSM_P1_RES_behavioral_sweep_*.csv files found.")
    frames = [pd.read_csv(p, dtype=str) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["variant_type"] = df["variant_type"].astype(str).str.strip().str.lower()
    df["contamination_pole"] = df["contamination_pole"].astype(str).str.strip().str.lower()
    df["behavioral_correct_bool"] = (
        df["behavioral_correct"].astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0})
    )
    return df


def fig_var_heatmap(df: pd.DataFrame) -> None:
    variants = ["canonical", "w1", "w2", "w3", "w4", "w5"]
    poles = ["high", "medium"]
    models = sorted(df["model"].astype(str).unique())

    cmap = mcolors.LinearSegmentedColormap.from_list("white_blue", ["#FFFFFF", P1_ACCENT])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    for ax, pole in zip(axes, poles):
        sub = df[df["contamination_pole"] == pole]
        mat = np.full((len(models), len(variants)), np.nan)
        for i, model in enumerate(models):
            for j, variant in enumerate(variants):
                vals = sub[
                    (sub["model"] == model) & (sub["variant_type"] == variant)
                ]["behavioral_correct_bool"].dropna()
                if len(vals):
                    mat[i, j] = float(vals.mean())

        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap=cmap)
        ax.set_title(f"Contamination: {pole}")
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=45, ha="right")
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        for i in range(len(models)):
            for j in range(len(variants)):
                if np.isnan(mat[i, j]):
                    continue
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
    cbar.set_label("VAR score")
    _save(fig, "GSM_P1_FIG_var_heatmap")


def fig_w4_gap() -> None:
    path = RESULTS_DIR / "GSM_P1_RES_w4_gap.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, dtype=str)
    for c in ["prose_var", "w4_var", "gap"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    agg = df.groupby("model", as_index=False)[["prose_var", "w4_var", "gap"]].mean()

    x = np.arange(len(agg))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - width / 2, agg["prose_var"], width, label="prose CSS", color=P1_ACCENT)
    ax.bar(x + width / 2, agg["w4_var"], width, label="w4 CSS", color=P1_PRIMARY)
    for i, g in enumerate(agg["gap"]):
        ax.text(i, max(agg["prose_var"].iloc[i], agg["w4_var"].iloc[i]) + 0.02, f"gap={g:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["model"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Prose vs W4 CSS")
    ax.legend()
    _save(fig, "GSM_P1_FIG_w4_gap")


def fig_rcs_by_difficulty() -> None:
    path = RESULTS_DIR / "GSM_P1_RES_rcs_by_difficulty.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, dtype=str)
    df["rcs"] = pd.to_numeric(df["rcs"], errors="coerce")
    order = ["easy", "medium", "hard"]
    models = sorted(df["model"].astype(str).unique())
    x = np.arange(len(order))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for idx, model in enumerate(models):
        sub = df[df["model"] == model].set_index("difficulty")
        vals = [float(sub.loc[d, "rcs"]) if d in sub.index else np.nan for d in order]
        ax.bar(x + (idx - (len(models) - 1) / 2) * width, vals, width=width, label=model)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("RCS")
    ax.set_title("RCS by Difficulty")
    ax.legend(fontsize=8)
    _save(fig, "GSM_P1_FIG_rcs_by_difficulty")


def fig_step_sensitivity() -> None:
    path = RESULTS_DIR / "GSM_P1_RES_step_count_sensitivity.csv"
    if not path.exists():
        return
    df = pd.read_csv(path, dtype=str)
    df["mean_css"] = pd.to_numeric(df["mean_css"], errors="coerce")
    subtype_to_x = {"gsm_symbolic": 0, "gsm_p1p2": 1}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        xs = [subtype_to_x.get(s, np.nan) for s in sub["problem_subtype"]]
        ys = sub["mean_css"].tolist()
        ax.scatter(xs, ys, s=80, alpha=0.9, label=model, color=P1_PRIMARY if "claude" in model else (P1_ACCENT if "gpt" in model else P1_LIGHT), edgecolor="black", linewidth=0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["gsm_symbolic", "gsm_p1p2"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("CSS")
    ax.set_title("Step Count Sensitivity by Subtype")
    ax.legend(fontsize=8)
    _save(fig, "GSM_P1_FIG_step_sensitivity")


def main() -> None:
    df = _load_behavioral()
    fig_var_heatmap(df)
    fig_w4_gap()
    fig_rcs_by_difficulty()
    fig_step_sensitivity()
    print("Generated GSM Probe 1 figures in figures/")


if __name__ == "__main__":
    main()
