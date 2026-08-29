from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_FIG = Path(__file__).resolve().parent
if str(_FIG) not in sys.path:
    sys.path.insert(0, str(_FIG))

from _common import (
    MODEL_LABELS,
    MODEL_ORDER,
    P1_ACCENT_BLUE,
    P1_LIGHT_BLUE,
    P1_PRIMARY_BLUE,
    bootstrap_ci_bool,
    bw_standard_blocksworld_mask,
    load_behavioral,
    output_dir,
    to_bool_series,
)


def metric_with_ci(df, model: str, variant: str):
    x = df[
        (df["model"].astype(str) == model)
        & (df["variant_type"].astype(str).str.upper() == variant.upper())
    ]
    vals = to_bool_series(x["behavioral_correct"]).dropna().astype(float).to_numpy()
    if len(vals) == 0:
        return (np.nan, np.nan, np.nan)
    m = float(vals.mean())
    lo, hi = bootstrap_ci_bool(vals)
    return (m, lo, hi)


def main():
    df = load_behavioral()
    df = df[df["model"].astype(str).str.lower() != "mock"].copy()
    df = df[bw_standard_blocksworld_mask(df["problem_id"])].copy()
    x = np.arange(len(MODEL_ORDER))
    width = 0.24
    variants = ["canonical", "W5", "W6"]
    colors = [P1_LIGHT_BLUE, P1_PRIMARY_BLUE, P1_ACCENT_BLUE]
    labels = ["Forward / canonical", "W5 Reversal / teardown", "W6 Procedural regeneration"]

    fig, ax = plt.subplots(figsize=(10, 5))
    by_model = {}
    for i, (v, c, lbl) in enumerate(zip(variants, colors, labels)):
        means = []
        yerr_low = []
        yerr_high = []
        for m in MODEL_ORDER:
            mean, lo, hi = metric_with_ci(df, m, v)
            means.append(mean)
            yerr_low.append(0 if np.isnan(mean) or np.isnan(lo) else max(mean - lo, 0))
            yerr_high.append(0 if np.isnan(mean) or np.isnan(hi) else max(hi - mean, 0))
            by_model.setdefault(m, {})[v] = mean
        ax.bar(
            x + (i - 1) * width,
            means,
            width,
            label=lbl,
            color=c,
            yerr=[yerr_low, yerr_high],
            capsize=3,
        )

    for i, m in enumerate(MODEL_ORDER):
        f = by_model[m].get("canonical", np.nan)
        w5 = by_model[m].get("W5", np.nan)
        pdas = np.nan if np.isnan(f) or np.isnan(w5) else (w5 - f)
        if not np.isnan(pdas):
            ax.text(i, min(1.02, max(f, w5) + 0.07), f"PDAS={pdas:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("VAR (0–1)")
    ax.set_title("Figure 2 — Planning Direction Asymmetry (PDAS)")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Standard BW instances only (BW_* numeric IDs; excludes BW_E* extended and MBW mystery).",
        fontsize=9,
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])
    fig.text(
        0.5,
        0.01,
        "Note: MBW W1 variants carry N=3 in some probes (bank limitation; defer full W1 parity). "
        "BW_E extended instances are excluded from PDAS here by design.",
        ha="center",
        fontsize=7,
    )
    out = output_dir()
    fig.savefig(out / "BW_FIG_P1_pdas_bars.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "BW_FIG_P1_pdas_bars.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
