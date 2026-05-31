# scripts/figures/fig1_robustness_scatter.py
# Figure 1: canonical accuracy vs W3 retention, GSM (all 5 models)
# + ALGO adversarial subtype breakdown
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results/figures"
FIG.mkdir(exist_ok=True)

# ---- data (from verified numbers) ----
gsm = pd.DataFrame(
    {
        "model": ["Claude", "GPT-4o", "Llama-8B", "Gemini-2.5", "o4-mini*"],
        "canon": [0.818, 0.818, 0.773, 0.909, 0.879],
        "W3": [0.750, 0.477, 0.409, 0.523, 0.841],
        "color": ["#0072B2", "#D55E00", "#CC79A7", "#009E73", "#E69F00"],
        "marker": ["o", "o", "o", "o", "*"],
        "msize": [90, 90, 90, 90, 130],
    }
)
gsm["retention"] = gsm.W3 / gsm.canon

# ALGO adversarial per-subtype (4 models, Gemini SP/WIS excluded due to API errors)
algo = pd.DataFrame(
    {
        "label": [
            "Claude SP",
            "GPT-4o SP",
            "Claude CC",
            "GPT-4o CC",
            "Claude WIS",
            "GPT-4o WIS",
        ],
        "canon": [0.647, 0.412, 0.700, 0.600, 0.353, 0.353],
        "W3": [0.000, 0.265, 0.600, 0.000, 0.000, 0.000],
        "color": ["#0072B2", "#D55E00", "#0072B2", "#D55E00", "#0072B2", "#D55E00"],
        "marker": ["s", "s", "D", "D", "^", "^"],
        "msize": [60] * 6,
    }
)
algo["retention"] = np.where(algo.canon > 0, algo.W3 / algo.canon, np.nan)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)
fig.suptitle("Canonical Accuracy vs. W3 Retention (entity rename)", fontsize=12, y=1.01)

# ---- LEFT: GSM ----
ax = axes[0]
ax.set_title("(a) Arithmetic (GSM)", fontsize=11)
for _, row in gsm.iterrows():
    ax.scatter(
        row.canon,
        row.retention,
        c=row.color,
        marker=row.marker,
        s=row.msize,
        zorder=3,
        edgecolors="white",
        linewidths=0.5,
    )
    offset = {
        "Claude": (0.003, 0.01),
        "GPT-4o": (0.003, -0.03),
        "Llama-8B": (0.003, -0.025),
        "Gemini-2.5": (-0.04, 0.012),
        "o4-mini*": (0.003, 0.01),
    }.get(row.model, (0.003, 0.01))
    ax.annotate(
        row.model,
        (row.canon + offset[0], row.retention + offset[1]),
        fontsize=8.5,
        color="#333333",
    )
ax.axhline(1.0, ls="--", color="gray", alpha=0.4, lw=1)
ax.set_xlabel("Canonical accuracy", fontsize=10)
ax.set_ylabel("W3 retention  (W3 / canonical)", fontsize=10)
ax.set_xlim(0.70, 0.97)
ax.set_ylim(0.45, 1.05)
ax.text(
    0.97,
    0.48,
    "* o4-mini canonical n=33/44",
    transform=ax.transAxes,
    ha="right",
    fontsize=7,
    color="gray",
)
ax.spines[["top", "right"]].set_visible(False)

# ---- RIGHT: ALGO adversarial ----
ax2 = axes[1]
ax2.set_title("(b) Adversarial ALGO by subtype", fontsize=11)
marker_legend = {"s": "Shortest-path", "D": "Coin-change", "^": "WIS (0-contam)"}
for _, row in algo.iterrows():
    if not np.isnan(row.retention):
        ax2.scatter(
            row.canon,
            row.retention,
            c=row.color,
            marker=row.marker,
            s=row.msize,
            zorder=3,
            edgecolors="white",
            linewidths=0.5,
        )
shape_handles = [
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        mfc="gray",
        ms=8,
        label="Shortest-path",
    ),
    Line2D(
        [0],
        [0],
        marker="D",
        color="w",
        mfc="gray",
        ms=8,
        label="Coin-change",
    ),
    Line2D(
        [0],
        [0],
        marker="^",
        color="w",
        mfc="gray",
        ms=8,
        label="WIS (0-contam)",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        mfc="#0072B2",
        ms=8,
        label="Claude",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        mfc="#D55E00",
        ms=8,
        label="GPT-4o",
    ),
]
ax2.legend(
    handles=shape_handles,
    fontsize=7.5,
    loc="upper left",
    framealpha=0.8,
    handlelength=1.2,
)
ax2.axhline(1.0, ls="--", color="gray", alpha=0.4, lw=1)
ax2.axhline(0.0, ls="-", color="gray", alpha=0.2, lw=0.8)
ax2.set_xlabel("Canonical accuracy", fontsize=10)
ax2.set_ylabel("W3 retention  (W3 / canonical)", fontsize=10)
ax2.set_xlim(-0.05, 0.80)
ax2.set_ylim(-0.15, 1.15)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
for ext in ["pdf", "png"]:
    plt.savefig(FIG / f"fig1_robustness_scatter.{ext}", bbox_inches="tight")
print("Figure 1 saved to", FIG)
