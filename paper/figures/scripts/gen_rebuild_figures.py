#!/usr/bin/env python3
"""Generate NeurIPS paper figures from rebuild/solidify/tri_v2 analyses."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
SOLIDIFY = ROOT / "rebuild" / "solidify"
TRI = ROOT / "rebuild" / "tri_v2"

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def fig_intrusion() -> None:
    rates = pd.read_csv(SOLIDIFY / "T2_intrusion_rates.csv")
    algo = rates[rates["family"] == "ALGO"].copy()
    fisher = pd.read_csv(SOLIDIFY / "T2_intrusion_fisher_algo.csv")
    order = ["Claude", "GPT-4o", "Llama", "Gemini", "o4-mini"]
    algo["model"] = pd.Categorical(algo["model"], categories=order, ordered=True)
    algo = algo.sort_values("model")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.4, 1]})

    ax = axes[0]
    x = np.arange(len(algo))
    y = algo["intrusion_rate"].to_numpy()
    lo = algo["wilson_ci95_lo"].to_numpy()
    hi = algo["wilson_ci95_hi"].to_numpy()
    ax.bar(x, y, color="#4C78A8", edgecolor="black", linewidth=0.4)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor="black", capsize=3, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(algo["model"].astype(str), rotation=20, ha="right")
    ax.set_ylabel("Intrusion rate on W3 errors")
    ax.set_title("(a) ALGO entity-rename intrusion")
    ax.set_ylim(0, max(hi.max() * 1.15, 0.3))

    ax = axes[1]
    vs = fisher.copy()
    labels = (
        vs["comparison"]
        .astype(str)
        .str.replace(r"^o4-mini vs\s+", "", regex=True)
        .tolist()
    )
    ors = vs["odds_ratio"].replace([np.inf, -np.inf], np.nan).fillna(20.0).to_numpy()
    ps = vs["fisher_p_two_sided"].to_numpy()
    ax.barh(np.arange(len(ors)), ors, color="#F58518", edgecolor="black", linewidth=0.4)
    ax.set_yticks(np.arange(len(ors)))
    ax.set_yticklabels(labels)
    ax.axvline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Odds ratio vs other models")
    ax.set_title("(b) o4-mini Fisher OR (ALGO)")
    for i, p in enumerate(ps):
        ax.text(min(ors[i] + 0.3, max(ors) * 0.95), i, f"p={p:.3f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(OUT / "fig_intrusion.pdf")
    fig.savefig(OUT / "fig_intrusion.png")
    plt.close(fig)


def fig_triangulation_complete_case() -> None:
    counts = pd.read_csv(SOLIDIFY / "T1_complete_case_counts.csv")
    want = counts[counts["panel"].isin(["full_440", "complete_case"])].copy()
    labels = ["retrieval", "computation", "mixed", "ambiguous"]
    cols = ["n_retrieval", "n_computation", "n_mixed", "n_ambiguous"]
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    x = np.arange(len(want))
    bottom = np.zeros(len(want))
    colors = ["#E45756", "#54A24B", "#4C78A8", "#B3B3B3"]
    for col, lab, color in zip(cols, labels, colors):
        vals = want[col].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=lab, color=color, edgecolor="black", linewidth=0.3)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"Full panel\nn={int(r.n)}\nconf={r.confident_label_rate:.1%}"
            for r in want.itertuples()
        ]
    )
    ax.set_ylabel("Instances")
    ax.set_title("Triangulation labels: full 440 vs complete-case 169")
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(OUT / "fig_triangulation_complete_case.pdf")
    fig.savefig(OUT / "fig_triangulation_complete_case.png")
    plt.close(fig)


def fig_mixed_failure() -> None:
    fail = pd.read_csv(TRI / "C_executed_mixed_failed_conditions.csv")
    fail = fail.sort_values("frac_of_mixed", ascending=True)
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    y = np.arange(len(fail))
    ax.barh(y, fail["frac_of_mixed"], color="#72B7B2", edgecolor="black", linewidth=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(fail["condition"].tolist(), fontsize=7)
    ax.set_xlabel("Fraction of executed-rule mixed (n=157)")
    ax.set_title("Which conjunction conditions fail on mixed instances")
    ax.set_xlim(0, 1.05)
    for i, (n, f) in enumerate(zip(fail["n_mixed_failing"], fail["frac_of_mixed"])):
        ax.text(min(f + 0.02, 0.98), i, f"{int(n)}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mixed_failure.pdf")
    fig.savefig(OUT / "fig_mixed_failure.png")
    plt.close(fig)


def fig_rule_crosstab() -> None:
    ct = pd.read_csv(TRI / "E_label_crosstab.csv")
    panel = ct[ct["panel"] == "full_440"].copy()
    order = ["retrieval", "computation", "mixed", "ambiguous"]
    mat = np.zeros((4, 4), dtype=float)
    for row in panel.itertuples():
        i = order.index(row.label_executed)
        j = order.index(row.label_appendix)
        mat[i, j] = row.n
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_yticklabels(order)
    ax.set_xlabel("Appendix symmetric rule")
    ax.set_ylabel("Executed 5-field AND")
    ax.set_title("Label crosstab on full 440 panel")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > mat.max() * 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rule_crosstab.pdf")
    fig.savefig(OUT / "fig_rule_crosstab.png")
    plt.close(fig)


def fig_coverage_holes() -> None:
    nums = pd.read_csv(ROOT / "rebuild" / "NUMBERS.csv")
    nc = nums[nums["value"].astype(str) == "NOT_COMPUTABLE"].copy()
    if nc.empty:
        return
    # Aggregate by family × reason keyword
    def bucket(reason: str) -> str:
        r = str(reason).lower()
        if "w5/w6" in r or "variant not present" in r:
            return "ALGO W5/W6 absent"
        if "no valid rows" in r or "bank" in r:
            return "GSM W6 bank hole"
        if "phase1" in r or "phase-1" in r or "cci" in r:
            return "o4-mini ALGO Phase-1/CCI"
        if "bw p2" in r or "not in this bw" in r:
            return "BW P2 model absent"
        if "fisher" in r or "refusal" in r:
            return "Fisher undefined"
        if "empty id" in r or "degenerate" in r:
            return "Pairwise/φ degenerate"
        return "Other"

    nc["bucket"] = nc["reason"].map(bucket) if "reason" in nc.columns else "Other"
    if "reason" not in nc.columns and "note" in nc.columns:
        nc["bucket"] = nc["note"].map(bucket)
    counts = nc["bucket"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.barh(counts.index.astype(str), counts.values, color="#E45756", edgecolor="black", linewidth=0.3)
    ax.set_xlabel("NOT_COMPUTABLE cells in rebuild/NUMBERS.csv")
    ax.set_title("Coverage holes under frozen filters (credit / design / data)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_coverage_holes.pdf")
    fig.savefig(OUT / "fig_coverage_holes.png")
    plt.close(fig)


def fig_crossover() -> None:
    cross = pd.read_csv(SOLIDIFY / "T3_crossover.csv")
    if cross.empty:
        return
    row = cross.iloc[0]
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    cats = [f"{row.subtype_1}\nW3", f"{row.subtype_2}\nW3"]
    a_vals = [float(row.a_W3_1) / max(float(row.n_1), 1), float(row.a_W3_2) / max(float(row.n_2), 1)]
    b_vals = [float(row.b_W3_1) / max(float(row.n_1), 1), float(row.b_W3_2) / max(float(row.n_2), 1)]
    # a_W3_* appear to be counts of successes in T3_crossover
    x = np.arange(2)
    w = 0.35
    ax.bar(x - w / 2, a_vals, w, label=str(row.model_a), color="#4C78A8", edgecolor="black", lw=0.3)
    ax.bar(x + w / 2, b_vals, w, label=str(row.model_b), color="#F58518", edgecolor="black", lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("W3 accuracy (matched)")
    ax.set_title(f"Strict crossover: {row.model_a} vs {row.model_b} ({row.verdict})")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(OUT / "fig_crossover.pdf")
    fig.savefig(OUT / "fig_crossover.png")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # Triangulation / intrusion rebuild figures retired from NeurIPS build.
    print(
        "Skipped retired rebuild figures "
        "(fig_triangulation_complete_case, fig_rule_crosstab, fig_coverage_holes, "
        "fig_intrusion, fig_mixed_failure, fig_crossover). "
        "Functions remain for archival calls."
    )
    print("Wrote nothing to", OUT)


if __name__ == "__main__":
    main()
