from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.figures._common import P3_CORAL, GREEN, GRAY, load_csv_candidate, output_dir


DIAG_COLORS = {
    "converging_retrieval": P3_CORAL,
    "converging_computation": GREEN,
    "2signal_diverging": GRAY,
    "diverging": GRAY,
}
FAMILY_MARKERS = {
    "blocksworld": "o",
    "mystery_blocksworld": "s",
    "BW_E": "^",
}


def infer_family(pid: str, fam: str) -> str:
    if str(pid).startswith("BW_E"):
        return "BW_E"
    return fam or "blocksworld"


def main():
    tri = load_csv_candidate(
        [
            "results/triangulation_per_instance.csv",
            "results/triangulation_per_instance_gpt4o.csv",
            "results/triangulation_per_instance_llama8b.csv",
        ]
    )
    if "problem_id" not in tri.columns:
        raise RuntimeError("triangulation CSV missing problem_id column")

    tri["contamination_score"] = pd.to_numeric(tri.get("contamination_score"), errors="coerce")
    tri["css"] = pd.to_numeric(tri.get("css"), errors="coerce")
    tri = tri.sort_values("contamination_score", na_position="last").reset_index(drop=True)
    tri["x"] = np.arange(len(tri))

    fig, ax = plt.subplots(figsize=(14, 5))

    # Background contamination gradient band.
    cont = tri["contamination_score"].fillna(0).to_numpy(dtype=float)
    if len(cont):
        xmin, xmax = -0.5, len(tri) - 0.5
        gradient = np.tile(cont, (2, 1))
        ax.imshow(
            gradient,
            extent=(xmin, xmax, 0, 1),
            origin="lower",
            aspect="auto",
            cmap="Blues",
            alpha=0.15,
            zorder=0,
        )

    for _, r in tri.iterrows():
        pid = str(r.get("problem_id", ""))
        fam = infer_family(pid, str(r.get("problem_family", "")))
        marker = FAMILY_MARKERS.get(fam, "o")
        diag = str(r.get("diagnosis", ""))
        color = DIAG_COLORS.get(diag, GRAY)
        y = r.get("css")
        if pd.isna(y):
            continue
        ax.scatter(r["x"], y, marker=marker, color=color, edgecolor="black", linewidth=0.4, s=45, zorder=3)

    ax.set_xlim(-0.5, len(tri) - 0.5)
    ax.set_ylim(0, 1.02)
    ax.set_xticks(tri["x"])
    ax.set_xticklabels(tri["problem_id"], rotation=90, fontsize=7)
    ax.set_ylabel("CSS")
    ax.set_xlabel("Problem instances (sorted by contamination_score)")
    ax.set_title("Figure 6 — Per-instance Triangulation Preview (CSS strip)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    out = output_dir()
    fig.savefig(out / "BW_FIG_P3_per_instance_triangulation.png", dpi=300, bbox_inches="tight")
    fig.savefig(out / "BW_FIG_P3_per_instance_triangulation.pdf", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
