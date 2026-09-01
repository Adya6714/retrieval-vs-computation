#!/usr/bin/env python3
"""270-config appendix triangulation sweep and label-count distribution.

Canonical rule: appendix three-signal (symmetric W3, CCI bands 0.10/0.67,
contamination floor vs p75, no greedy_succeeds). The 270 grid varies CCI
dead-zone width, W3 cut, and contamination percentile under that structure.

Does not overwrite results/derived/ALGO_P3_triangulation_v3.csv.
Does not call any model API.
"""

from __future__ import annotations

import sys
from pathlib import Path

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

DERIVED = REPO_ROOT / "results/derived"
PANEL = DERIVED / "ALGO_P3_triangulation_v3.csv"


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL)
    if len(df) != 440:
        raise ValueError(f"Expected 440 rows in {PANEL}, got {len(df)}")
    for col in ["VAR_canonical", "VAR_W3", "ACI", "instance_contamination_score", "instance_rank_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main() -> None:
    df = load_panel()
    chosen = label_appendix_three_signal(df)
    chosen_counts = count_labels(chosen)
    legacy = count_labels(label_legacy_five_field(df))

    labeled = df.copy()
    labeled["label_appendix"] = chosen.to_numpy()
    labeled["label_legacy_five_field"] = label_legacy_five_field(df).to_numpy()
    labeled_path = DERIVED / "ALGO_P3_triangulation_appendix_labels.csv"
    labeled.to_csv(labeled_path, index=False)

    rows = []
    for cci in CCI_THRESHOLDS:
        for w3 in W3_CUTOFFS:
            for pct in CONTAM_PERCENTILES:
                labs = label_sweep_cell(df, cci_thr=float(cci), w3_cutoff=float(w3), contam_pct=int(pct))
                c = count_labels(labs)
                rows.append(
                    {
                        "cci_threshold": float(cci),
                        "w3_cutoff": float(w3),
                        "contam_percentile": int(pct),
                        "n_retrieval": c["retrieval"],
                        "n_computation": c["computation"],
                        "n_mixed": c["mixed"],
                        "n_ambiguous": c["ambiguous"],
                        "n": c["n"],
                    }
                )
    sweep = pd.DataFrame(rows)
    if len(sweep) != 270:
        raise ValueError(f"Expected 270 sweep rows, got {len(sweep)}")
    sweep_path = DERIVED / "ALGO_P3_appendix_threshold_sensitivity.csv"
    sweep.to_csv(sweep_path, index=False)

    dist = (
        sweep.groupby(["n_retrieval", "n_computation"], as_index=False)
        .agg(
            n_configs=("n", "size"),
            n_mixed_min=("n_mixed", "min"),
            n_mixed_max=("n_mixed", "max"),
            n_ambiguous_min=("n_ambiguous", "min"),
            n_ambiguous_max=("n_ambiguous", "max"),
        )
        .sort_values(["n_configs", "n_retrieval", "n_computation"], ascending=[False, True, True])
    )
    dist_path = DERIVED / "ALGO_P3_appendix_sweep_distribution.csv"
    dist.to_csv(dist_path, index=False)

    comparison = pd.DataFrame(
        [
            {
                "rule": "appendix_canonical",
                "description": "symmetric W3, CCI 0.10/0.67, contam floor vs p75, no greedy_succeeds",
                **chosen_counts,
            },
            {
                "rule": "legacy_five_field_sensitivity",
                "description": "asymmetric W3 0.2/0.5, greedy_succeeds, CCI 0.5, median split — not published",
                **legacy,
            },
        ]
    )
    comparison_path = DERIVED / "P3_triangulation_rule_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    n_unique_pairs = len(dist)
    min_r, max_r = int(sweep["n_retrieval"].min()), int(sweep["n_retrieval"].max())
    min_c, max_c = int(sweep["n_computation"].min()), int(sweep["n_computation"].max())
    summary = pd.DataFrame(
        [
            {
                "chosen_retrieval": chosen_counts["retrieval"],
                "chosen_computation": chosen_counts["computation"],
                "chosen_mixed": chosen_counts["mixed"],
                "chosen_ambiguous": chosen_counts["ambiguous"],
                "paper_counts_match": int(chosen_counts == PAPER_COUNTS),
                "n_sweep_configs": len(sweep),
                "n_unique_retrieval_computation_pairs": n_unique_pairs,
                "retrieval_min": min_r,
                "retrieval_max": max_r,
                "computation_min": min_c,
                "computation_max": max_c,
                "unstable": int(n_unique_pairs > 1),
            }
        ]
    )
    summary_path = DERIVED / "ALGO_P3_appendix_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {labeled_path}")
    print(f"Wrote {sweep_path} ({len(sweep)} configs)")
    print(f"Wrote {dist_path} ({n_unique_pairs} distinct (retrieval, computation) pairs)")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {summary_path}")
    print(f"chosen appendix: {chosen_counts}")
    print(f"legacy AND:      {legacy}")
    print(f"sweep retrieval range [{min_r}, {max_r}] computation range [{min_c}, {max_c}]")
    print(dist.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
