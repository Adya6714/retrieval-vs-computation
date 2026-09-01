#!/usr/bin/env python3
"""ALGO triangulation threshold sensitivity (legacy 5-field labeling rule).

Canonical published sweep is:
    python scripts/consolidate/run_appendix_triangulation_sweep.py

This script retains the AND-rule grid as a named sensitivity variant.
"""

Contamination percentile is a single moving split (generalizes the paper's
median half-split): high = rank_pct > p/100, low = rank_pct <= p/100.

Outputs:
    results/derived/ALGO_P3_threshold_sensitivity.csv
    results/derived/ALGO_P3_behavioural_subtype_patterns.csv
    results/figures/ALGO_P3_FIG_threshold_sensitivity_heatmap.{png,pdf}
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DER = ROOT / "results" / "derived"
FIG = ROOT / "results" / "figures"
TRI_PATH = DER / "ALGO_P3_triangulation_v3.csv"

CCI_THRESHOLDS = np.round(np.arange(0.05, 0.90 + 1e-9, 0.05), 2)
W3_CUTOFFS = [0.0, 0.25, 0.5, 0.75, 1.0]
CONTAM_PERCENTILES = [50, 75, 90]


def load_panel() -> pd.DataFrame:
    if not TRI_PATH.exists():
        raise FileNotFoundError(
            f"Missing {TRI_PATH}. Re-run scripts/ALGO_P3_SCR_triangulation.py first."
        )
    df = pd.read_csv(TRI_PATH)
    if len(df) != 440:
        raise ValueError(f"Expected 440 ALGO instance-rows, got {len(df)}")

    for col in [
        "VAR_canonical",
        "VAR_W3",
        "VRI_gap",
        "ACI",
        "instance_contamination_score",
        "instance_rank_pct",
        "greedy_succeeds",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "instance_rank_pct" not in df.columns or df["instance_rank_pct"].isna().all():
        df["instance_rank_pct"] = df.groupby("problem_subtype")[
            "instance_contamination_score"
        ].rank(method="average", pct=True)

    for flag in ["missing_core", "missing_phase2", "parse_failure_or_missing"]:
        if flag not in df.columns:
            df[flag] = False
        else:
            df[flag] = df[flag].fillna(False).astype(bool)

    return df


def behavioural_subtype_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """SP/CC/WIS behavioural pole per (model, subtype, instance_type) from P1.

    retrieval   — mean VRI_gap >= 0.15 (canonical survives, W3 collapses)
    computation — mean VAR_W3 >= 0.25 and mean VRI_gap < 0.15 (W3 retains)
    mixed       — otherwise (low accuracy / no clear pole)
    """
    g = (
        df.groupby(["model", "problem_subtype", "instance_type"], dropna=False)
        .agg(
            n=("problem_id", "size"),
            mean_can=("VAR_canonical", "mean"),
            mean_w3=("VAR_W3", "mean"),
            mean_vri=("VRI_gap", "mean"),
        )
        .reset_index()
    )
    g["w3_retention"] = np.where(
        g["mean_can"] > 1e-9, g["mean_w3"] / g["mean_can"], np.nan
    )

    pattern = np.full(len(g), "mixed", dtype=object)
    retrieval = g["mean_vri"].fillna(0) >= 0.15
    computation = (g["mean_w3"].fillna(0) >= 0.25) & (g["mean_vri"].fillna(0) < 0.15)
    pattern[computation.to_numpy()] = "computation"
    pattern[retrieval.to_numpy()] = "retrieval"  # VRI pole wins ties
    g["behavioural_pattern"] = pattern
    return g[
        [
            "model",
            "problem_subtype",
            "instance_type",
            "behavioural_pattern",
            "w3_retention",
            "mean_vri",
            "mean_can",
            "mean_w3",
            "n",
        ]
    ]


def label_with_thresholds(
    df: pd.DataFrame,
    cci_thr: float,
    w3_cutoff: float,
    contam_pct: int,
) -> pd.Series:
    """Parameterized ALGO_P3_SCR_triangulation.compute_convergence_labels."""
    cut = contam_pct / 100.0
    rank = df["instance_rank_pct"]
    high_contam = rank > cut
    low_contam = rank <= cut

    out = pd.Series("mixed", index=df.index, dtype=object)
    ambiguous = (
        df["missing_core"].to_numpy()
        | df["parse_failure_or_missing"].to_numpy()
        | df["missing_phase2"].to_numpy()
    )
    out.loc[ambiguous] = "ambiguous"

    greed = df["greedy_succeeds"]
    greed_ok = greed.fillna(False).astype(bool) & greed.notna()

    retrieval = (
        (df["VAR_canonical"] > 0.5)
        & (df["VAR_W3"] < w3_cutoff)
        & high_contam
        & greed_ok
        & ~ambiguous
    )
    computation = (
        (df["VAR_W3"] > w3_cutoff)
        & (df["ACI"] > cci_thr)
        & low_contam
        & ~ambiguous
    )
    out.loc[retrieval] = "retrieval_signal"
    out.loc[~retrieval & computation] = "computation_signal"
    return out


def agreement_metrics(
    labels: pd.Series,
    df: pd.DataFrame,
    cell_patterns: pd.DataFrame,
) -> dict[str, float]:
    """Agreement of triangulation labels with SP/CC/WIS behavioural cell poles.

    agreement_rate: among patterned cells, fraction where the strong label
      matches the cell pole (mixed/ambiguous count as non-matches). Recovers
      how much of the behavioural subtype structure the rule recovers.
    precision_among_strong: among strong labels in patterned cells, fraction
      that match the cell pole.
    """
    merged = df[["model", "problem_subtype", "instance_type"]].copy()
    merged["label"] = labels.to_numpy()
    merged = merged.merge(
        cell_patterns[
            ["model", "problem_subtype", "instance_type", "behavioural_pattern"]
        ],
        on=["model", "problem_subtype", "instance_type"],
        how="left",
        validate="many_to_one",
    )

    patterned = merged["behavioural_pattern"].isin(["retrieval", "computation"])
    lab_dir = merged["label"].map(
        {
            "retrieval_signal": "retrieval",
            "computation_signal": "computation",
        }
    )
    match = patterned & lab_dir.notna() & (lab_dir == merged["behavioural_pattern"])
    n_patterned = int(patterned.sum())
    n_strong_patterned = int((patterned & lab_dir.notna()).sum())

    agreement = float(match.sum() / n_patterned) if n_patterned else float("nan")
    precision = (
        float(match.sum() / n_strong_patterned) if n_strong_patterned else float("nan")
    )
    return {
        "agreement_rate_subtype_pattern": agreement,
        "precision_among_strong": precision,
        "n_patterned_cells_instances": n_patterned,
        "n_strong_in_patterned": n_strong_patterned,
    }


def sweep(df: pd.DataFrame, cell_patterns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for cci in CCI_THRESHOLDS:
        for w3 in W3_CUTOFFS:
            for contam_pct in CONTAM_PERCENTILES:
                labels = label_with_thresholds(df, float(cci), float(w3), int(contam_pct))
                vc = labels.value_counts()
                agr = agreement_metrics(labels, df, cell_patterns)
                rows.append(
                    {
                        "cci_threshold": float(cci),
                        "w3_retention_cutoff": float(w3),
                        "contam_percentile": int(contam_pct),
                        "n_rows": n,
                        "retrieval_pct": 100.0 * vc.get("retrieval_signal", 0) / n,
                        "computation_pct": 100.0 * vc.get("computation_signal", 0) / n,
                        "mixed_pct": 100.0 * vc.get("mixed", 0) / n,
                        "ambiguous_pct": 100.0 * vc.get("ambiguous", 0) / n,
                        "n_retrieval": int(vc.get("retrieval_signal", 0)),
                        "n_computation": int(vc.get("computation_signal", 0)),
                        "n_mixed": int(vc.get("mixed", 0)),
                        "n_ambiguous": int(vc.get("ambiguous", 0)),
                        "strong_pct": 100.0
                        * (
                            vc.get("retrieval_signal", 0)
                            + vc.get("computation_signal", 0)
                        )
                        / n,
                        **agr,
                    }
                )
    return pd.DataFrame(rows)


def _pivot(sub: pd.DataFrame, value: str) -> pd.DataFrame:
    return (
        sub.pivot_table(
            index="w3_retention_cutoff",
            columns="cci_threshold",
            values=value,
            aggfunc="first",
        )
        .sort_index(ascending=False)
        .reindex(columns=list(CCI_THRESHOLDS))
    )


def plot_heatmap(sweep_df: pd.DataFrame, out_stem: Path) -> None:
    """3×2 heatmap: rows = contam percentile; cols = strong% and agreement."""
    metrics = [
        ("strong_pct", "Strong label % (retr + comp)", "magma", 0.0, None),
        (
            "agreement_rate_subtype_pattern",
            "Agreement with SP/CC/WIS pattern",
            "viridis",
            0.0,
            None,  # scale to observed max (rates stay << 1 under this rule)
        ),
    ]
    fig, axes = plt.subplots(
        len(CONTAM_PERCENTILES),
        len(metrics),
        figsize=(12.5, 9.0),
        sharex=True,
        sharey=True,
    )

    for r, pct in enumerate(CONTAM_PERCENTILES):
        sub = sweep_df[sweep_df["contam_percentile"] == pct]
        for c, (col, title, cmap, vmin, vmax) in enumerate(metrics):
            ax = axes[r, c]
            pivot = _pivot(sub, col)
            data = pivot.to_numpy(dtype=float)
            if vmax is None:
                finite = data[np.isfinite(data)]
                vmax_use = float(np.nanmax(finite)) if finite.size else 1.0
                vmax_use = max(vmax_use, 1.0)
            else:
                vmax_use = vmax
            im = ax.imshow(
                data,
                aspect="auto",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax_use,
                interpolation="nearest",
            )
            if r == 0:
                ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(f"contam split {pct}th\nW3-retention cutoff")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=8)
            if r == len(CONTAM_PERCENTILES) - 1:
                ax.set_xlabel("CCI threshold")
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels(
                    [
                        f"{x:.2f}" if i % 2 == 0 else ""
                        for i, x in enumerate(pivot.columns)
                    ],
                    rotation=90,
                    fontsize=7,
                )
            else:
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        "ALGO triangulation threshold sensitivity (n=440 problem×model cells)",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    # Companion: retrieval% / computation% / mixed% / ambiguous% at contam=50.
    fig2, axes2 = plt.subplots(2, 2, figsize=(11.5, 8.0), sharex=True, sharey=True)
    sub50 = sweep_df[sweep_df["contam_percentile"] == 50]
    panels = [
        ("retrieval_pct", "Retrieval %", "Reds"),
        ("computation_pct", "Computation %", "Blues"),
        ("mixed_pct", "Mixed %", "Oranges"),
        ("ambiguous_pct", "Ambiguous %", "Greys"),
    ]
    for ax, (col, title, cmap) in zip(axes2.ravel(), panels):
        pivot = _pivot(sub50, col)
        data = pivot.to_numpy(dtype=float)
        im = ax.imshow(
            data,
            aspect="auto",
            cmap=cmap,
            vmin=0,
            vmax=max(float(np.nanmax(data)), 1.0),
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_ylabel("W3-retention cutoff")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=8)
        ax.set_xlabel("CCI threshold")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(
            [f"{x:.2f}" if i % 2 == 0 else "" for i, x in enumerate(pivot.columns)],
            rotation=90,
            fontsize=7,
        )
        fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig2.suptitle(
        "Label composition vs CCI × W3 (contam split = 50th, n=440)",
        fontsize=12,
    )
    fig2.tight_layout(rect=(0, 0, 1, 0.97))
    stem2 = out_stem.parent / f"{out_stem.name}_composition"
    fig2.savefig(stem2.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig2.savefig(stem2.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig2)


def main() -> None:
    df = load_panel()
    cell_patterns = behavioural_subtype_patterns(df)
    DER.mkdir(parents=True, exist_ok=True)
    cell_path = DER / "ALGO_P3_behavioural_subtype_patterns.csv"
    cell_patterns.to_csv(cell_path, index=False)

    default_labels = label_with_thresholds(df, cci_thr=0.5, w3_cutoff=0.5, contam_pct=50)
    print("Paper-like defaults (cci=0.5, w3=0.5, contam=50) distribution:")
    print((default_labels.value_counts(normalize=True) * 100).round(1).to_string())
    print("\nBehavioural SP/CC/WIS cell poles:")
    print(cell_patterns["behavioural_pattern"].value_counts().to_string())

    sweep_df = sweep(df, cell_patterns)
    out_csv = DER / "ALGO_P3_threshold_sensitivity.csv"
    sweep_df.to_csv(out_csv, index=False)

    heat_stem = FIG / "ALGO_P3_FIG_threshold_sensitivity_heatmap"
    plot_heatmap(sweep_df, heat_stem)

    print(
        f"\nConfigs: {len(sweep_df)} "
        f"(= {len(CCI_THRESHOLDS)} CCI × {len(W3_CUTOFFS)} W3 × {len(CONTAM_PERCENTILES)} contam)"
    )
    print(f"Wrote {out_csv}")
    print(f"Wrote {cell_path}")
    print(f"Wrote {heat_stem.with_suffix('.png')} / .pdf")
    print(f"Wrote {heat_stem}_composition.png / .pdf")
    print(
        "Strong-label % range:",
        f"{sweep_df['strong_pct'].min():.1f}–{sweep_df['strong_pct'].max():.1f}",
    )
    agr = sweep_df["agreement_rate_subtype_pattern"]
    print("Agreement rate range:", f"{agr.min():.3f}–{agr.max():.3f}")


if __name__ == "__main__":
    main()
