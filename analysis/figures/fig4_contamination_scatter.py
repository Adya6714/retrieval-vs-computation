from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures._common import (
    MODEL_LABELS,
    MODEL_ORDER,
    P1_LIGHT_BLUE,
    P1_PRIMARY_BLUE,
    P3_CORAL,
    P3_GOLD,
    load_behavioral,
    load_csv_candidate,
    output_dir,
    pearson_r_p,
    to_bool_series,
)


def point_color(pid: str, fam: str) -> str:
    if str(pid).startswith("BW_E"):
        return P3_GOLD
    if fam == "mystery_blocksworld":
        return P3_CORAL
    return P1_PRIMARY_BLUE


def main():
    beh = load_behavioral()
    cont = load_csv_candidate(["results/BW_P3_RES_contamination_triage.csv", "results/contamination_triage.csv"])
    qb = pd.read_csv("data/problems/question_bank.csv", dtype=str)

    canon = beh[beh["variant_type"].astype(str).str.upper() == "CANONICAL"].copy()
    canon["var"] = to_bool_series(canon["behavioral_correct"]).astype(float)
    meta = qb[qb["variant_type"].astype(str).str.upper() == "CANONICAL"][
        ["problem_id", "problem_subtype"]
    ].copy()
    meta = meta.rename(columns={"problem_subtype": "problem_family"})
    canon = canon.merge(meta, on="problem_id", how="left", suffixes=("", "_qb"))
    if "problem_family" not in canon.columns and "problem_family_qb" in canon.columns:
        canon["problem_family"] = canon["problem_family_qb"]
    elif "problem_family_qb" in canon.columns:
        canon["problem_family"] = canon["problem_family"].fillna(canon["problem_family_qb"])
    canon = canon.merge(cont[["problem_id", "contamination_score"]], on="problem_id", how="left")
    canon["contamination_score"] = pd.to_numeric(canon["contamination_score"], errors="coerce")
    canon = canon.dropna(subset=["contamination_score", "var"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, model in zip(axes, MODEL_ORDER):
        d = canon[canon["model"] == model].copy()
        if d.empty:
            ax.set_title(f"{MODEL_LABELS[model]} (no data)")
            continue

        cols = [point_color(pid, fam) for pid, fam in zip(d["problem_id"], d["problem_family"])]
        ax.scatter(d["contamination_score"], d["var"], c=cols, alpha=0.9, edgecolor="k", linewidth=0.4)

        x = d["contamination_score"].to_numpy(dtype=float)
        y = d["var"].to_numpy(dtype=float)
        if len(x) >= 3:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(max(0, float(np.nanmin(x))), min(0.4, float(np.nanmax(x))), 100)
            ys = m * xs + b
            ax.plot(xs, ys, color=P1_PRIMARY_BLUE, linewidth=2)
            # lightweight CI band from residual std (visual aid)
            resid = y - (m * x + b)
            sd = float(np.std(resid))
            ax.fill_between(xs, ys - 1.96 * sd, ys + 1.96 * sd, color=P1_LIGHT_BLUE, alpha=0.4)
            r, p = pearson_r_p(x, y)
            ax.text(0.02, 0.95, f"r={r:.3f}, p={p:.3f}", transform=ax.transAxes, va="top", fontsize=9)

        ax.set_title(MODEL_LABELS[model])
        ax.set_xlim(0, 0.4)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("contamination_score")
        ax.spines[["top", "right"]].set_visible(False)

    axes[0].set_ylabel("VAR(canonical)")
    fig.suptitle("Figure 4 — Contamination vs VAR(canonical)", fontsize=13)
    fig.tight_layout()
    out = output_dir()
    fig.savefig(out / "BW_FIG_P3_contamination_scatter.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "BW_FIG_P3_contamination_scatter.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
