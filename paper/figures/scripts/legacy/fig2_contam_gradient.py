# scripts/figures/fig2_contam_gradient.py
# Figure 2: template contamination gradient vs canonical & W3 accuracy
# Shows WIS as the natural within-family control
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results/figures"
FIG.mkdir(exist_ok=True)

subtypes = ["Coin-change\n(CC)", "Shortest-path\n(SP)", "WIS\n(zero-contam)"]
x = np.arange(3)
width = 0.18

# Canonical accuracy by subtype x model (adversarial only, verified numbers)
data_canon = {
    "Claude": [0.700, 0.647, 0.353],
    "GPT-4o": [0.600, 0.412, 0.353],
    "Llama-8B": [0.200, 0.059, 0.059],
}
data_w3 = {
    "Claude": [0.600, 0.000, 0.000],
    "GPT-4o": [0.000, 0.265, 0.000],
    "Llama-8B": [0.000, 0.000, 0.059],
}
template_contam = [0.468, 0.147, 0.000]  # mean per subtype

colors = {"Claude": "#0072B2", "GPT-4o": "#D55E00", "Llama-8B": "#CC79A7"}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

# ---- LEFT: bar chart canonical vs W3 per model per subtype ----
ax = axes[0]
ax.set_title(
    "(a) Adversarial accuracy under canonical and entity rename (W3)", fontsize=10
)
offsets = {"Claude": -1.5 * width, "GPT-4o": -0.5 * width, "Llama-8B": 0.5 * width}
for m, col in colors.items():
    ax.bar(
        x + offsets[m],
        data_canon[m],
        width,
        label=f"{m} (can)",
        color=col,
        alpha=0.9,
    )
    ax.bar(
        x + offsets[m] + width * 0.05,
        data_w3[m],
        width * 0.65,
        label=f"{m} (W3)",
        color=col,
        alpha=0.45,
        hatch="////",
    )
    # zero annotation for total collapse
    for j, val in enumerate(data_w3[m]):
        if val == 0.0 and data_canon[m][j] > 0:
            ax.text(
                x[j] + offsets[m] + width * 0.35,
                0.01,
                "✕",
                ha="center",
                fontsize=8,
                color=col,
            )

ax.set_xticks(x)
ax.set_xticklabels(subtypes, fontsize=9)
ax.set_ylabel("Accuracy", fontsize=10)
ax.set_ylim(0, 0.85)

legend_handles = [
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        mfc="#0072B2",
        ms=10,
        label="Claude",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        mfc="#D55E00",
        ms=10,
        label="GPT-4o",
    ),
    Line2D(
        [0],
        [0],
        marker="s",
        color="w",
        mfc="#CC79A7",
        ms=10,
        label="Llama-8B",
    ),
    Line2D([0], [0], lw=6, color="gray", alpha=0.9, label="Canonical"),
    Line2D(
        [0],
        [0],
        lw=4,
        color="gray",
        alpha=0.45,
        label="Entity rename (W3)",
    ),
]
ax.legend(handles=legend_handles, fontsize=8, loc="upper right", framealpha=0.8)
ax.spines[["top", "right"]].set_visible(False)

# ---- RIGHT: W3 retention vs template contamination ----
ax2 = axes[1]
ax2.set_title(
    "(b) W3 retention vs template contamination\n(within-family control)", fontsize=10
)
for m, col in colors.items():
    can = np.array(data_canon[m])
    w3 = np.array(data_w3[m])
    retention = np.where(can > 0, w3 / can, 0.0)
    ax2.scatter(
        template_contam,
        retention,
        color=col,
        s=90,
        zorder=4,
        label=m,
        edgecolors="white",
        linewidths=0.5,
    )
    ax2.plot(template_contam, retention, color=col, alpha=0.4, lw=1.2)

ax2.axhline(0, ls="--", color="gray", alpha=0.3, lw=1)
ax2.set_xlabel("Mean template contamination score", fontsize=10)
ax2.set_ylabel("W3 retention (W3 / canonical)", fontsize=10)
ax2.set_xlim(-0.05, 0.55)
ax2.set_ylim(-0.05, 1.05)
for tc, st in zip(template_contam, ["CC", "SP", "WIS"]):
    ax2.annotate(st, (tc, -0.04), ha="center", fontsize=8.5, color="#555")
ax2.legend(fontsize=9, framealpha=0.8)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
for ext in ["pdf", "png"]:
    plt.savefig(FIG / f"fig2_contam_gradient.{ext}", bbox_inches="tight")
print("Figure 2 saved to", FIG)
