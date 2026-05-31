"""New EMNLP-style figures that surface findings the existing 7 do not.

This module is import-friendly with the existing `gen_figures.py` (shares the
same color palette and rcParams) and is invoked from `main()` there once we
wire it in.

New figures:
  fig_landscape.pdf       — single-panel accuracy x W3-retention across 3
                             families and 5 models; the visual headline.
  fig_implaus.pdf         — per-model plausible vs implausible final-correct
                             with paired-Wilcoxon p; shows null effect.
  fig_violations.pdf      — BW Probe-2 violation profile per model.
  fig_within_model.pdf    — 3x5 small-multiples of per-problem canonical x
                             W3-kept; per-cell Spearman rho and OLS line.

Inputs:
  results/derived/master_per_problem_5model.csv
  results/derived/implausibility_detection.csv
  results/derived/bw_violation_profile.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
DER = ROOT / "results" / "derived"
OUT = Path(__file__).resolve().parents[1]   # LLM Overleaf/figures

COLOR = {
    "Claude":   "#0072B2",
    "GPT-4o":   "#D55E00",
    "Llama":    "#CC79A7",
    "Llama-8B": "#CC79A7",
    "Gemini":   "#009E73",
    "Gemini-2.5": "#009E73",
    "o4-mini":  "#E69F00",
}
FAMILY_MARKER = {"ALGO": "s", "GSM": "o", "BW": "^"}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
})


# ---------------------------------------------------------------------------
def fig_landscape() -> None:
    """Single-panel scatter of canonical accuracy x W3 retention,
    5 models x 3 families = 15 dots, per-family OLS trendline,
    Spearman rho annotated per family."""

    master = pd.read_csv(DER / "master_per_problem_5model.csv")
    # aggregate per (family, model)
    agg = (master.groupby(["family", "model_short"], as_index=False)
                  .agg(canonical=("canonical_correct", "mean"),
                       W3=("W3_kept", "mean")))
    agg["retention"] = agg.W3 / agg.canonical.where(agg.canonical > 0, np.nan)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.subplots_adjust(top=0.92, bottom=0.16, left=0.10, right=0.97)

    # per-family OLS line and Spearman annotation
    fam_pos = {"ALGO": 0.95, "GSM": 0.85, "BW": 0.75}
    for fam in ["ALGO", "GSM", "BW"]:
        sub = agg[agg.family == fam].dropna()
        if len(sub) < 3: continue
        # OLS line
        x = sub.canonical.values; y = sub.retention.values
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, slope*xs + intercept, ls=":", color="gray", alpha=0.6, lw=0.9)
        # Spearman
        rho, p = stats.spearmanr(x, y)
        ax.text(0.99, fam_pos[fam],
                f"{fam}: $\\rho={rho:+.2f}$, p={p:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.8,
                color="#444")

    # scatter
    for _, row in agg.iterrows():
        if pd.isna(row.retention): continue
        ax.scatter(row.canonical, row.retention,
                   c=COLOR.get(row.model_short, "gray"),
                   marker=FAMILY_MARKER[row.family],
                   s=130, edgecolors="white", linewidths=0.8, zorder=4)
        ax.annotate(row.model_short,
                    (row.canonical + 0.008, row.retention + 0.012),
                    fontsize=8.0, color="#222")

    ax.axhline(1.0, ls="--", color="gray", alpha=0.4, lw=1)
    ax.set_xlabel("Canonical accuracy")
    ax.set_ylabel(r"$W_3$ retention  ($\mathrm{Acc}_{W3}/\mathrm{Acc}_{\mathrm{can}}$)")
    ax.set_title("Robustness landscape: 5 models $\\times$ 3 families")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.15)
    ax.spines[["top", "right"]].set_visible(False)

    # legend
    legend_handles = [
        mpatches.Patch(color=COLOR[m], label=m) for m in ["Claude","Gemini","GPT-4o","Llama","o4-mini"]
    ] + [
        plt.scatter([], [], c="gray", marker=FAMILY_MARKER["ALGO"], s=70, label="ALGO"),
        plt.scatter([], [], c="gray", marker=FAMILY_MARKER["GSM"],  s=70, label="GSM"),
        plt.scatter([], [], c="gray", marker=FAMILY_MARKER["BW"],   s=70, label="BW"),
    ]
    ax.legend(handles=legend_handles, ncol=4, fontsize=7.6, loc="lower left",
              framealpha=0.9, handlelength=1.3, handletextpad=0.4,
              columnspacing=0.8)

    plt.savefig(OUT / "fig_landscape.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_landscape.pdf")


# ---------------------------------------------------------------------------
def fig_implaus() -> None:
    """ALGO P2B plausible vs implausible final-correct, 5 models, with
    paired-Wilcoxon p annotated."""

    df = pd.read_csv(DER / "implausibility_detection.csv")
    models = ["Claude","Gemini","GPT-4o","Llama","o4-mini"]
    df = df.set_index("model").reindex(models).reset_index()

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    fig.subplots_adjust(top=0.88, bottom=0.18, left=0.10, right=0.98)

    x = np.arange(len(models))
    w = 0.32
    ax.bar(x - w/2, df.plausible_correct, width=w, color="#4C9F70",
           edgecolor="white", linewidth=0.5, label="plausible inj.")
    ax.bar(x + w/2, df.implausible_correct, width=w, color="#B07AA1",
           edgecolor="white", linewidth=0.5, label="implausible inj.")

    for i, row in df.iterrows():
        if pd.isna(row.wilcoxon_p): continue
        # p-value annotation above the higher bar
        h = max(row.plausible_correct, row.implausible_correct)
        sig = "n.s." if row.wilcoxon_p > 0.05 else ("*" if row.wilcoxon_p > 0.01 else "**")
        ax.text(i, h + 0.025, f"p={row.wilcoxon_p:.2f} ({sig})",
                ha="center", va="bottom", fontsize=7.6, color="#444")

    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("Final-answer correct rate")
    ax.set_ylim(0, 0.75)
    ax.set_title("Plausible vs. implausible injection: no model differentiates "
                 "(ALGO Probe 2B, 5 models, paired Wilcoxon)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", ls=":", color="#bbb", alpha=0.5)

    plt.savefig(OUT / "fig_implaus.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_implaus.pdf")


# ---------------------------------------------------------------------------
def fig_violations() -> None:
    """BW Probe-2 violation profile per model: semantic-validity, repetition,
    partial-goal, first-illegal step. Stacked vertical bar chart with 4 metrics."""

    df = pd.read_csv(DER / "bw_violation_profile.csv")
    df = df.set_index("model").loc[["Claude","GPT-4o","Llama"]].reset_index()

    metrics = [
        ("mean_semantic_validity",  "Semantic validity",   "#4C9F70"),
        ("mean_repetition_rate",    "Repetition rate",     "#B07AA1"),
        ("mean_partial_goal",       "Partial goals reached", "#F39C12"),
        ("median_first_illegal",    "Median first-illegal step", "#3498DB"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.0))
    fig.subplots_adjust(top=0.83, bottom=0.18, left=0.06, right=0.98, wspace=0.40)

    for ax, (col, label, color) in zip(axes, metrics):
        ax.barh(df.model, df[col], color=color, edgecolor="white", linewidth=0.6)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("")
        ax.spines[["top","right"]].set_visible(False)
        for i, v in enumerate(df[col]):
            ax.text(v, i, f" {v:.2f}" if v < 5 else f" {v:.1f}",
                    va="center", fontsize=8.4, color="#222")
        ax.grid(axis="x", ls=":", color="#bbb", alpha=0.5)

    fig.suptitle("Blocksworld Probe-2 failure profile per model "
                 "(n=50 sessions each)", fontsize=10.5)
    plt.savefig(OUT / "fig_violations.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_violations.pdf")


# ---------------------------------------------------------------------------
def fig_within_model() -> None:
    """3x5 small multiples: per-problem canonical x W3-kept, with Spearman
    rho. Shows the reconciliation of the paradox — within-model, canonical
    and W3 are POSITIVELY correlated even though VAR is negatively correlated
    with canonical."""

    master = pd.read_csv(DER / "master_per_problem_5model.csv")
    families = ["ALGO", "GSM", "BW"]
    models = ["Claude", "Gemini", "GPT-4o", "Llama", "o4-mini"]

    fig, axes = plt.subplots(3, 5, figsize=(13.0, 6.2), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.93, bottom=0.08, left=0.06, right=0.99,
                        hspace=0.32, wspace=0.18)

    for r, fam in enumerate(families):
        for c, m in enumerate(models):
            ax = axes[r, c]
            sub = master[(master.family == fam) & (master.model_short == m)]
            x = sub.canonical_correct.dropna().values
            y = sub.W3_kept.dropna().values
            # ensure x and y aligned
            df_xy = sub[["canonical_correct","W3_kept"]].dropna()
            x = df_xy.canonical_correct.values
            y = df_xy.W3_kept.values
            if len(x) < 5:
                ax.text(0.5, 0.5, "n.a.", ha="center", va="center",
                        fontsize=10, color="gray", transform=ax.transAxes)
                ax.set_title(f"{fam} / {m}", fontsize=8.8)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            # add small jitter so binary points don't overlap
            ax.scatter(x + np.random.uniform(-0.05, 0.05, len(x)),
                       y + np.random.uniform(-0.05, 0.05, len(y)),
                       c=COLOR.get(m, "gray"), s=14, alpha=0.55,
                       edgecolors="none", zorder=3)
            # OLS line through binary data
            if x.std() > 0 and y.std() > 0:
                slope, intercept = np.polyfit(x, y, 1)
                xs = np.array([0, 1])
                ax.plot(xs, slope*xs + intercept, ls="-", color="#444",
                        alpha=0.8, lw=1.0, zorder=4)
            if x.std() == 0 or y.std() == 0:
                ax.set_title(f"{fam} / {m}    $\\rho$=n.a.\\,(constant; n={len(x)})",
                              fontsize=8.6)
            else:
                rho, p = stats.spearmanr(x, y)
                sig = "" if p > 0.05 else ("*" if p > 0.01 else ("**" if p > 0.001 else "***"))
                ax.set_title(f"{fam} / {m}    $\\rho$={rho:+.2f}{sig} (n={len(x)})",
                              fontsize=8.6)
            ax.set_xlim(-0.15, 1.15); ax.set_ylim(-0.15, 1.15)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.spines[["top","right"]].set_visible(False)
            if r == 2: ax.set_xlabel("canonical correct")
            if c == 0: ax.set_ylabel("$W_3$ kept")

    fig.suptitle("Per-problem canonical correctness vs. $W_3$ kept "
                 "(within-model). Positive $\\rho$ resolves the across-model paradox.",
                 fontsize=10.5)
    plt.savefig(OUT / "fig_within_model.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  wrote fig_within_model.pdf")


# ---------------------------------------------------------------------------
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_landscape()
    fig_implaus()
    fig_violations()
    fig_within_model()
    print("New 4 figures written.")


if __name__ == "__main__":
    main()
