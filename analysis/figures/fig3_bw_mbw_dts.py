from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analysis.figures._common import (
    MODEL_LABELS,
    MODEL_ORDER,
    P1_PRIMARY_BLUE,
    P3_CORAL,
    add_problem_family_from_qb,
    compute_var,
    load_behavioral,
    output_dir,
)


def main():
    df = load_behavioral()
    if "problem_family" not in df.columns:
        df = add_problem_family_from_qb(df)

    variants = ["W2", "W3", "W4", "W5"]
    x = np.arange(len(variants))
    width = 0.34

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        bw_vals = [compute_var(df[df["problem_family"] == "blocksworld"], model, v) for v in variants]
        mbw_vals = [compute_var(df[df["problem_family"] == "mystery_blocksworld"], model, v) for v in variants]
        bw_vals = [0 if v is None else v for v in bw_vals]
        mbw_vals = [0 if v is None else v for v in mbw_vals]

        ax.bar(x - width / 2, bw_vals, width, color=P1_PRIMARY_BLUE, label="BW")
        ax.bar(x + width / 2, mbw_vals, width, color=P3_CORAL, label="MBW")
        ax.set_xticks(x)
        ax.set_xticklabels(variants)
        ax.set_ylim(0, 1.0)
        ax.set_title(MODEL_LABELS[model], fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("VAR")
    axes[-1].legend(loc="upper right", fontsize=9)
    fig.suptitle("Figure 3 — Domain Transfer Score (BW vs MBW)", fontsize=13)
    fig.tight_layout()
    out = output_dir()
    fig.savefig(out / "BW_FIG_P1_bw_mbw_dts.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "BW_FIG_P1_bw_mbw_dts.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
