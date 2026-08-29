# scripts/figures/fig5_teaser.py
# Figure 5: Teaser diagram showing the three-probe triangulation concept
# Uses matplotlib patches only (no external assets)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results/figures"
FIG.mkdir(exist_ok=True)

fig, ax = plt.subplots(figsize=(11, 4), dpi=150)
ax.set_xlim(0, 11)
ax.set_ylim(0, 4)
ax.axis("off")


def box(ax, x, y, w, h, text, bg="#EEF4FB", ec="#2166AC", fs=8.5, bold=False):
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.08",
        facecolor=bg,
        edgecolor=ec,
        lw=1.4,
    )
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        weight=weight,
        wrap=True,
        multialignment="center",
        color="#1a1a2e",
    )


def arrow(ax, x1, y1, x2, y2, col="#555"):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
    )


# ---- Problem instance (centre) ----
box(
    ax,
    4.1,
    1.5,
    2.8,
    1.0,
    "Problem Instance\n(219 problems,\n3 families)",
    "#FFFFFF",
    "#444",
    bold=True,
)

# ---- Three probes ----
box(
    ax,
    0.2,
    2.5,
    2.5,
    1.0,
    "Probe 1\nSurface Invariance\n(6 variants, W3 rename)",
    "#FFF0E0",
    "#D55E00",
)
box(
    ax,
    0.2,
    0.5,
    2.5,
    1.0,
    "Probe 2\nPlan-Execution\nCoupling (CCI, TEP)",
    "#E8F5E9",
    "#2ca02c",
)
box(
    ax,
    8.3,
    1.5,
    2.4,
    1.0,
    "Probe 3\nTraining-Data\nProximity (Infini-gram)",
    "#F3EAF8",
    "#7B2D8B",
)

# arrows from problem to probes
arrow(ax, 4.1, 2.0, 2.7, 2.95)  # to P1
arrow(ax, 4.1, 2.0, 2.7, 1.0)  # to P2
arrow(ax, 6.9, 2.0, 8.3, 2.0)  # to P3

# ---- Per-instance signals ----
box(
    ax,
    0.2,
    1.5,
    2.5,
    0.7,
    "W3 retention\nVRI score",
    "#FFF0E0",
    "#D55E00",
    fs=8,
)
arrow(ax, 2.7, 2.5, 2.7, 2.2)
arrow(ax, 2.7, 0.5 + 1.0, 2.7, 2.2)
arrow(ax, 2.7, 1.5 + 0.35, 3.3, 1.5 + 0.5)  # P1 signal → centre

# ---- Convergence ----
box(
    ax,
    4.1,
    0.1,
    2.8,
    1.0,
    "Per-Instance\nBehavioral Diagnosis\n(retrieval / computation / ambiguous)",
    "#F5F5F5",
    "#444",
    bold=False,
    fs=8,
)
arrow(ax, 5.5, 1.5, 5.5, 1.1)  # from problem down to diagnosis
arrow(ax, 2.7, 1.5, 4.1, 0.7)
arrow(ax, 8.3, 2.0, 6.9, 0.7)

# ---- Key findings callouts ----
ax.text(10.9, 3.8, "Key findings:", fontsize=8.5, ha="right", weight="bold", color="#222")
findings = [
    "• 0% correct when algorithm name invoked (13/13 steps)",
    "• W3 retention tracks contamination, not difficulty",
    "• CCI 2.1× higher for Claude vs GPT-4o at matched accuracy",
    "• o4-mini W3 retention = 0.957 vs 0.583 (GPT-4o)",
]
for i, f in enumerate(findings):
    ax.text(10.9, 3.4 - i * 0.35, f, fontsize=7.5, ha="right", color="#333")

plt.tight_layout(pad=0.3)
for ext in ["pdf", "png"]:
    plt.savefig(FIG / f"fig5_teaser.{ext}", bbox_inches="tight")
print("Figure 5 teaser saved to", FIG)
