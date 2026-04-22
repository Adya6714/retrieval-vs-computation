from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from analysis.figures._common import (
    MODEL_LABELS,
    MODEL_ORDER,
    P1_PRIMARY_BLUE,
    VARIANT_ORDER,
    add_problem_family_from_qb,
    compute_var,
    load_behavioral,
    output_dir,
)


def build_matrix(df, family: str):
    mat = np.full((len(MODEL_ORDER), len(VARIANT_ORDER)), np.nan, dtype=float)
    for i, m in enumerate(MODEL_ORDER):
        for j, v in enumerate(VARIANT_ORDER):
            val = compute_var(df, m, v, family)
            if val is not None:
                mat[i, j] = val
    return mat


def main():
    df = load_behavioral()
    if "problem_family" not in df.columns:
        df = add_problem_family_from_qb(df)

    bw = build_matrix(df, "blocksworld")
    mbw = build_matrix(df, "mystery_blocksworld")

    cmap = LinearSegmentedColormap.from_list("var_blue", ["#FFFFFF", P1_PRIMARY_BLUE])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, mat, title in [
        (axes[0], bw, "Blocksworld (BW)"),
        (axes[1], mbw, "Mystery Blocksworld (MBW)"),
    ]:
        im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(VARIANT_ORDER)))
        ax.set_xticklabels(VARIANT_ORDER, fontsize=10)
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=10)
        ax.set_title(title, fontsize=12)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                label = "NA" if np.isnan(v) else f"{v:.2f}"
                color = "white" if (not np.isnan(v) and v >= 0.6) else "black"
                ax.text(j, i, label, ha="center", va="center", fontsize=8, color=color)

    fig.suptitle("Figure 1 — Variant Accuracy Rate (VAR) Heatmap", fontsize=14)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("VAR (0–1)")
    fig.tight_layout()
    out = output_dir()
    fig.savefig(out / "BW_FIG_P1_var_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "BW_FIG_P1_var_heatmap.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
