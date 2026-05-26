# scripts/figures/fig4_algorithm_paradox.py
# Figure 4: reasoning type vs final-answer correctness (ALGO step traces)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results/figures"
FIG.mkdir(exist_ok=True)

labels = [
    "Algorithm\nInvocation",
    "Backtracking",
    "Forward\nSimulation",
    "Local\nGreedy",
    "Unclear",
]
n_steps = [13, 10, 77, 390, 1041]
pct_correct = [0.0, 0.0, 1.3, 3.8, 13.3]
colors = ["#e41a1c", "#e41a1c", "#ff8800", "#4daf4a", "#377eb8"]

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)

bars = ax.bar(
    x,
    pct_correct,
    color=colors,
    alpha=0.85,
    width=0.55,
    edgecolor="white",
    linewidth=0.5,
)
for bar, n, pct in zip(bars, n_steps, pct_correct):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.25,
        f"n={n}",
        ha="center",
        fontsize=8.5,
        color="#444",
    )

# highlight the 0.0 bar for algo invocation
ax.annotate(
    "0 correct\nout of 13 steps\nacross 4 models",
    xy=(0, 0.15),
    xytext=(0.8, 8),
    arrowprops=dict(arrowstyle="->", color="#e41a1c", lw=1.5),
    fontsize=8.5,
    color="#e41a1c",
)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("Final-answer correctness (%)", fontsize=10)
ax.set_ylim(0, 18)
ax.set_title(
    "Algorithm-Invocation Paradox\n"
    "Step-level reasoning style vs. final-answer correctness "
    "(n=1,531 steps, 4 models)",
    fontsize=10,
)
ax.spines[["top", "right"]].set_visible(False)
ax.axhline(13.3, ls=":", color="#377eb8", alpha=0.5, lw=1)

plt.tight_layout()
for ext in ["pdf", "png"]:
    plt.savefig(FIG / f"fig4_algorithm_paradox.{ext}", bbox_inches="tight")
print("Figure 4 saved to", FIG)
