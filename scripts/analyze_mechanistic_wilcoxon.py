#!/usr/bin/env python3
"""Aligned-pair Wilcoxon for mechanistic final-layer ranks (Appendix F style).

H1: canonical final-layer rank > W6 final-layer rank (one-sided).
Pairs by problem_id with _W6 suffix stripped.

Usage:
  python3 scripts/analyze_mechanistic_wilcoxon.py \\
      results/raw/mechanistic_sweep_llama31_8b_instruct.csv
"""

from __future__ import annotations

import argparse
import ast
import math
from pathlib import Path

import pandas as pd
from scipy import stats


def _parse_list(x):
    if isinstance(x, list):
        return x
    if pd.isna(x) or x == "":
        return []
    return ast.literal_eval(x)


def _base_id(pid: str) -> str:
    pid = str(pid).strip()
    return pid[:-3] if pid.endswith("_W6") else pid


def _final_rank(row) -> float | None:
    ranks = _parse_list(row["target_rank_per_layer"])
    if not ranks:
        return None
    return float(ranks[-1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["base_id"] = df["problem_id"].map(_base_id)
    df["final_rank"] = df.apply(_final_rank, axis=1)

    print(f"file={args.csv}  rows={len(df)}  model={df['model'].iloc[0] if len(df) else '?'}")
    print(df.groupby(["problem_family", "variant_type"]).size().to_string())
    print()
    print("One-sided Wilcoxon (H1: canonical final_rank > W6 final_rank)")
    print("-" * 72)

    for fam in sorted(df["problem_family"].unique()):
        can = df[(df.problem_family == fam) & (df.variant_type == "canonical")].set_index("base_id")
        w6 = df[(df.problem_family == fam) & (df.variant_type == "W6")].set_index("base_id")
        ids = sorted(set(can.index) & set(w6.index))
        pairs = [(can.loc[i, "final_rank"], w6.loc[i, "final_rank"]) for i in ids]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None and not (math.isnan(a) or math.isnan(b))]
        n = len(pairs)
        if n == 0:
            print(f"{fam}: no aligned pairs")
            continue
        a = [p[0] for p in pairs]
        b = [p[1] for p in pairs]
        med_can = float(pd.Series(a).median())
        med_w6 = float(pd.Series(b).median())
        ratios = [math.log10(c / w) for c, w in pairs if w > 0 and c > 0]
        med_log = float(pd.Series(ratios).median()) if ratios else float("nan")
        # zeros in diffs: wilcoxon may warn; match paper style (alternative greater)
        try:
            w_stat, p = stats.wilcoxon(a, b, alternative="greater", zero_method="wilcox")
        except ValueError as e:
            w_stat, p = float("nan"), float("nan")
            print(f"{fam}: wilcoxon failed ({e})")
            continue
        print(
            f"{fam}: n={n}  med_rank can={med_can:.1f} W6={med_w6:.1f}  "
            f"med_log10(can/W6)={med_log:.2f}  W={w_stat:.0f}  p={p:.4g}"
        )


if __name__ == "__main__":
    main()
