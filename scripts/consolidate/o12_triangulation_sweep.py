#!/usr/bin/env python3
"""O12: Probe 3 triangulation 270-config threshold sweep — identification failure.

Reconstructs the published 270-grid (18 CCI × 5 W3 × 3 contam percentiles)
under the appendix three-signal structure, persists every config's counts,
and measures cross-config label stability.

Grid (rebuild/triangulation_rule.py):
  CCI_THRESHOLDS = 0.05..0.90 step 0.05  (18)
  W3_CUTOFFS     = {0.0, 0.25, 0.50, 0.75, 1.0}  (5)
  CONTAM_PERCENTILES = {50, 75, 90}  (3)
  → 18 × 5 × 3 = 270

Per cell, label_sweep_cell pairs CCI bands as
  retrieval_max = min(cci, 1-cci), computation_min = max(cci, 1-cci)
with symmetric W3 cut and contamination floor vs percentile.
Tie-breaking: unanimous −1 → retrieval; unanimous +1 → computation;
conflict → mixed; else → ambiguous. No greedy_succeeds conjunct.

Framing: the classification the framework exists to produce is not identified.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "rebuild") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "rebuild"))

from triangulation_rule import (  # noqa: E402
    CCI_THRESHOLDS,
    CONTAM_PERCENTILES,
    PAPER_COUNTS,
    W3_CUTOFFS,
    count_labels,
    label_appendix_three_signal,
    label_legacy_five_field,
    label_sweep_cell,
)

DER = REPO_ROOT / "results" / "derived"
FIG = REPO_ROOT / "paper" / "figures"
PANEL = DER / "ALGO_P3_triangulation_v3.csv"

OUT_SWEEP = DER / "O12_triangulation_sweep.csv"
OUT_STAB = DER / "O12_label_stability.csv"
OUT_JACCARD = DER / "O12_reasonable_jaccard.csv"
OUT_DRIVER = DER / "O12_param_driver.csv"
OUT_HIST = FIG / "fig_o12_retrieval_histogram.pdf"


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL)
    if len(df) != 440:
        raise ValueError(f"Expected 440 rows in {PANEL}, got {len(df)}")
    for col in ["VAR_canonical", "VAR_W3", "ACI", "instance_contamination_score", "instance_rank_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for flag in ["missing_core", "missing_phase2", "parse_failure_or_missing"]:
        if flag not in df.columns:
            df[flag] = False
        else:
            df[flag] = df[flag].fillna(False).astype(bool)
    return df


def eta_squared(y: np.ndarray, groups: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    grand = y.mean()
    ss_tot = float(np.sum((y - grand) ** 2))
    if ss_tot < 1e-15:
        return float("nan")
    ss_between = 0.0
    for g in np.unique(groups):
        mask = groups == g
        ss_between += mask.sum() * (y[mask].mean() - grand) ** 2
    return ss_between / ss_tot


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else float("nan")


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_panel()
    instance_keys = (df["model"].astype(str) + "::" + df["problem_id"].astype(str)).tolist()
    n_inst = len(instance_keys)

    # --- full 270 sweep + label matrix ---
    sweep_rows: list[dict] = []
    label_mat = np.empty((270, n_inst), dtype=object)
    config_ids: list[str] = []

    idx = 0
    for cci in CCI_THRESHOLDS:
        for w3 in W3_CUTOFFS:
            for pct in CONTAM_PERCENTILES:
                labs = label_sweep_cell(df, cci_thr=float(cci), w3_cutoff=float(w3), contam_pct=int(pct))
                c = count_labels(labs)
                cid = f"cci{cci:.2f}_w3{w3:.2f}_p{pct}"
                ret_max = min(float(cci), 1.0 - float(cci))
                comp_min = max(float(cci), 1.0 - float(cci))
                sweep_rows.append(
                    {
                        "config_id": cid,
                        "config_index": idx,
                        "cci_threshold": float(cci),
                        "cci_retrieval_max": ret_max,
                        "cci_computation_min": comp_min,
                        "w3_cutoff": float(w3),
                        "contam_percentile": int(pct),
                        "n_retrieval": c["retrieval"],
                        "n_computation": c["computation"],
                        "n_mixed": c["mixed"],
                        "n_ambiguous": c["ambiguous"],
                        "n": c["n"],
                        "matches_paper_15_1_300_124": int(
                            c["retrieval"] == PAPER_COUNTS["retrieval"]
                            and c["computation"] == PAPER_COUNTS["computation"]
                            and c["mixed"] == PAPER_COUNTS["mixed"]
                            and c["ambiguous"] == PAPER_COUNTS["ambiguous"]
                        ),
                        "rule_structure": "appendix_three_signal_signed_votes",
                        "tie_break": "unanimous_-1=retrieval; unanimous_+1=computation; conflict=mixed; else=ambiguous",
                        "probes_combined": "W3_vote|CCI_bands|contamination_floor_vs_percentile",
                    }
                )
                label_mat[idx, :] = labs.to_numpy()
                config_ids.append(cid)
                idx += 1

    assert idx == 270
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(OUT_SWEEP, index=False)

    # --- parameter driver (eta² on n_retrieval) ---
    y = sweep["n_retrieval"].to_numpy(dtype=float)
    driver_rows = []
    for col in ["cci_threshold", "w3_cutoff", "contam_percentile"]:
        eta = eta_squared(y, sweep[col].to_numpy())
        driver_rows.append(
            {
                "parameter": col,
                "eta_squared_n_retrieval": round(float(eta), 4),
                "n_levels": int(sweep[col].nunique()),
                "n_retrieval_min_at_levels": int(sweep.groupby(col)["n_retrieval"].min().min()),
                "n_retrieval_max_at_levels": int(sweep.groupby(col)["n_retrieval"].max().max()),
            }
        )
    # pairwise partial: mean within-level range
    for col in ["cci_threshold", "w3_cutoff", "contam_percentile"]:
        others = [c for c in ["cci_threshold", "w3_cutoff", "contam_percentile"] if c != col]
        ranges = sweep.groupby(others)["n_retrieval"].agg(lambda s: s.max() - s.min())
        driver_rows.append(
            {
                "parameter": f"{col}_mean_within_slice_range",
                "eta_squared_n_retrieval": round(float(ranges.mean()), 4),
                "n_levels": "",
                "n_retrieval_min_at_levels": int(ranges.min()),
                "n_retrieval_max_at_levels": int(ranges.max()),
            }
        )
    driver = pd.DataFrame(driver_rows)
    driver.to_csv(OUT_DRIVER, index=False)
    top = max(
        (r for r in driver_rows if not str(r["parameter"]).endswith("_range")),
        key=lambda r: r["eta_squared_n_retrieval"] if r["eta_squared_n_retrieval"] == r["eta_squared_n_retrieval"] else -1,
    )

    # --- reasonable configs for Jaccard ---
    # Include appendix canonical + legacy + a small grid of defensible cells
    reasonable: dict[str, pd.Series] = {
        "appendix_canonical_0.10_0.67_p75": label_appendix_three_signal(df),
        "legacy_five_field_AND": label_legacy_five_field(df),
    }
    # nearest-grid / defensible appendix-structure cells
    for cci, w3, pct, name in [
        (0.10, 0.50, 75, "grid_cci0.10_w3.50_p75"),  # bands 0.10/0.90
        (0.35, 0.50, 75, "grid_cci0.35_w3.50_p75"),  # bands 0.35/0.65 ≈ appendix spirit
        (0.50, 0.50, 75, "grid_cci0.50_w3.50_p75"),  # collapsed dead zone
        (0.35, 0.50, 50, "grid_cci0.35_w3.50_p50"),  # median contam
        (0.35, 0.50, 90, "grid_cci0.35_w3.50_p90"),  # strict contam
        (0.10, 0.50, 50, "grid_cci0.10_w3.50_p50"),
    ]:
        reasonable[name] = label_sweep_cell(df, cci_thr=cci, w3_cutoff=w3, contam_pct=pct)

    ret_sets = {
        name: set(instance_keys[i] for i, lab in enumerate(labs.to_numpy()) if lab == "retrieval")
        for name, labs in reasonable.items()
    }
    jacc_rows = []
    for a, b in combinations(sorted(ret_sets.keys()), 2):
        ja = jaccard(ret_sets[a], ret_sets[b])
        jacc_rows.append(
            {
                "config_a": a,
                "config_b": b,
                "n_retrieval_a": len(ret_sets[a]),
                "n_retrieval_b": len(ret_sets[b]),
                "n_intersection": len(ret_sets[a] & ret_sets[b]),
                "n_union": len(ret_sets[a] | ret_sets[b]),
                "jaccard_retrieval_sets": round(ja, 4),
            }
        )
    # also vs each config's self counts for reference
    for name, s in ret_sets.items():
        jacc_rows.append(
            {
                "config_a": name,
                "config_b": name,
                "n_retrieval_a": len(s),
                "n_retrieval_b": len(s),
                "n_intersection": len(s),
                "n_union": len(s),
                "jaccard_retrieval_sets": 1.0,
            }
        )
    jacc = pd.DataFrame(jacc_rows)
    jacc.to_csv(OUT_JACCARD, index=False)

    # --- per-instance label stability across 270 ---
    stab_rows = []
    n_stable_90 = 0
    n_flip = 0
    for i, key in enumerate(instance_keys):
        labs = label_mat[:, i]
        vals, counts = np.unique(labs, return_counts=True)
        order = np.argsort(-counts)
        mode_lab = vals[order[0]]
        mode_n = int(counts[order[0]])
        mode_frac = mode_n / 270.0
        n_distinct = int(len(vals))
        stable_90 = mode_frac > 0.90
        flips = n_distinct > 1
        if stable_90:
            n_stable_90 += 1
        if flips:
            n_flip += 1
        model, pid = key.split("::", 1)
        # retrieval rate across configs
        ret_frac = float(np.mean(labs == "retrieval"))
        stab_rows.append(
            {
                "instance_key": key,
                "model": model,
                "problem_id": pid,
                "n_configs": 270,
                "n_distinct_labels": n_distinct,
                "modal_label": mode_lab,
                "modal_count": mode_n,
                "modal_fraction": round(mode_frac, 4),
                "stable_gt_90pct": bool(stable_90),
                "flips": bool(flips),
                "frac_retrieval": round(ret_frac, 4),
                "frac_computation": round(float(np.mean(labs == "computation")), 4),
                "frac_mixed": round(float(np.mean(labs == "mixed")), 4),
                "frac_ambiguous": round(float(np.mean(labs == "ambiguous")), 4),
                "label_histogram": "|".join(f"{v}:{int(c)}" for v, c in zip(vals[order], counts[order])),
            }
        )
    stab = pd.DataFrame(stab_rows)
    stab.to_csv(OUT_STAB, index=False)

    # --- histogram ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    vals = sweep["n_retrieval"].to_numpy()
    bins = np.arange(vals.min() - 0.5, vals.max() + 1.5, 1) if vals.max() <= 50 else 30
    # use unique counts for discrete
    uniq, cnts = np.unique(vals, return_counts=True)
    ax.bar(uniq, cnts, width=max(1.0, (uniq.max() - uniq.min()) / 40), color="#4C78A8", edgecolor="white")
    ax.set_xlabel("n_retrieval_labels")
    ax.set_ylabel("Number of configs (of 270)")
    ax.set_title("Distribution of retrieval-label counts")
    ax.axvline(PAPER_COUNTS["retrieval"], color="#E45756", ls="--", lw=1.5, label=f"paper={PAPER_COUNTS['retrieval']}")
    ax.legend(fontsize=8)

    ax = axes[1]
    # stacked by contam percentile — the dominant driver
    for pct, color in zip([50, 75, 90], ["#F58518", "#54A24B", "#B279A2"]):
        sub = sweep[sweep["contam_percentile"] == pct]["n_retrieval"]
        ax.hist(sub, bins=20, alpha=0.55, label=f"contam p{pct}", color=color)
    ax.set_xlabel("n_retrieval_labels")
    ax.set_ylabel("Configs")
    ax.set_title("By contamination percentile (dominant driver)")
    ax.legend(fontsize=8)
    fig.suptitle(
        "O12: Probe 3 triangulation threshold is unidentified\n"
        f"n_retrieval ∈ [{int(vals.min())}, {int(vals.max())}] across 270 configs; "
        f"driver={top['parameter']} (η²={top['eta_squared_n_retrieval']:.2f})",
        fontsize=11,
        y=1.05,
    )
    fig.tight_layout()
    fig.savefig(OUT_HIST, bbox_inches="tight")
    fig.savefig(OUT_HIST.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # --- console summary ---
    print(f"Wrote {OUT_SWEEP} ({len(sweep)} configs)")
    print(f"Wrote {OUT_STAB} ({len(stab)} instances)")
    print(f"Wrote {OUT_JACCARD}")
    print(f"Wrote {OUT_DRIVER}")
    print(f"Wrote {OUT_HIST}")
    print(
        f"\nn_retrieval range: [{int(vals.min())}, {int(vals.max())}]  "
        f"unique count-tuples: {sweep.groupby(['n_retrieval','n_computation']).ngroups}"
    )
    print(f"Dominant parameter: {top['parameter']} η²={top['eta_squared_n_retrieval']}")
    print(driver.to_string(index=False))
    print(
        f"\nLabel stability: {n_stable_90}/{n_inst} ({n_stable_90/n_inst:.1%}) "
        f"modal in >90% of configs; {n_flip}/{n_inst} ({n_flip/n_inst:.1%}) flip"
    )
    print("\nReasonable-config retrieval Jaccard (off-diagonal):")
    print(
        jacc[jacc["config_a"] != jacc["config_b"]]
        .sort_values("jaccard_retrieval_sets")
        .to_string(index=False)
    )
    # note W3 collapse
    w3_identical = True
    for cci in CCI_THRESHOLDS:
        for pct in CONTAM_PERCENTILES:
            labs = [
                tuple(
                    label_sweep_cell(df, cci_thr=float(cci), w3_cutoff=float(w), contam_pct=int(pct)).tolist()
                )
                for w in (0.25, 0.5, 0.75)
            ]
            if not (labs[0] == labs[1] == labs[2]):
                w3_identical = False
                break
        if not w3_identical:
            break
    print(f"\nW3 cutoffs {{0.25,0.50,0.75}} identical label vectors: {w3_identical} (VAR_W3 is binary)")


if __name__ == "__main__":
    main()
