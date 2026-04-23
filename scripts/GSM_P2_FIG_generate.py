#!/usr/bin/env python3
"""Generate GSM Probe 2 figures."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.stats import bootstrap_ci


RESULTS_DIR = Path("results")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

HIGH_COLOR = "#EF4444"
MED_COLOR = "#3B82F6"
P2_PRIMARY = "#6D28D9"


def _save(fig: plt.Figure, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=150, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _load() -> pd.DataFrame:
    path = RESULTS_DIR / "GSM_P2_RES_cci.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype=str)
    for c in ["cci_score", "tep_score"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["contamination_pole"] = df["contamination_pole"].astype(str).str.strip().str.lower()
    return df


def fig_cci_violin(df: pd.DataFrame) -> None:
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4 * len(models), 4.5), sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        sub = df[df["model"] == model]
        high = sub[sub["contamination_pole"] == "high"]["cci_score"].dropna().tolist()
        medium = sub[sub["contamination_pole"] == "medium"]["cci_score"].dropna().tolist()
        parts = ax.violinplot([high if high else [np.nan], medium if medium else [np.nan]], positions=[1, 2], showmeans=True, showmedians=False)
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(HIGH_COLOR if i == 0 else MED_COLOR)
            body.set_alpha(0.65)
        ax.axhline(0.3, color="black", linestyle="--", linewidth=1)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["high", "medium"])
        ax.set_title(model)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("CCI")
    fig.suptitle("CCI Distribution by Contamination Pole")
    _save(fig, "GSM_P2_FIG_cci_violin")


def fig_tep_bar(df: pd.DataFrame) -> None:
    rows = []
    for model, sub in df.groupby("model"):
        vals = sub["tep_score"].dropna().astype(float).tolist()
        if not vals:
            continue
        lo, hi = bootstrap_ci(vals, ci=0.95)
        mean = float(np.mean(vals))
        rows.append((model, mean, mean - lo, hi - mean))
    if not rows:
        return
    rows.sort(key=lambda x: x[0])
    labels = [r[0] for r in rows]
    means = [r[1] for r in rows]
    yerr = np.array([[r[2] for r in rows], [r[3] for r in rows]])

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(rows)), means, color=P2_PRIMARY, alpha=0.85, yerr=yerr, capsize=4)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean TEP")
    ax.set_title("Mean TEP per Model (bootstrap 95% CI)")
    _save(fig, "GSM_P2_FIG_tep_bar")


def fig_cci_vs_contam_scatter(df: pd.DataFrame) -> None:
    triage_path = RESULTS_DIR / "GSM_P3_RES_contamination_triage.csv"
    if not triage_path.exists():
        return
    tri = pd.read_csv(triage_path, dtype=str)[["problem_id", "contamination_score"]]
    tri["contamination_score"] = pd.to_numeric(tri["contamination_score"], errors="coerce")
    merged = df.merge(tri[["problem_id", "contamination_score"]], on="problem_id", how="left")
    merged = merged.dropna(subset=["cci_score", "contamination_score"])
    colors = merged["contamination_pole"].map({"high": HIGH_COLOR, "medium": MED_COLOR}).fillna(P2_PRIMARY)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.scatter(merged["contamination_score"], merged["cci_score"], c=colors, alpha=0.8, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Contamination score")
    ax.set_ylabel("CCI score")
    ax.set_title("CCI vs Contamination Score")
    _save(fig, "GSM_P2_FIG_cci_vs_contam_scatter")


def main() -> None:
    df = _load()
    fig_cci_violin(df)
    fig_tep_bar(df)
    fig_cci_vs_contam_scatter(df)
    print("Generated GSM Probe 2 figures in figures/")


if __name__ == "__main__":
    main()
