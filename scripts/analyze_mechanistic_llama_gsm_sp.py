#!/usr/bin/env python3
"""Analyze Llama GSM+SP mechanistic follow-up (Appendix H).

Reads results/raw/mechanistic_llama_gsm_sp_raw.csv and writes:

  results/derived/mechanistic_llama_gsm_sp_summary.csv
  results/derived/mechanistic_llama_behavior_link.csv
  results/derived/mechanistic_llama_gold_distribution.csv

Usage:
  python3 scripts/analyze_mechanistic_llama_gsm_sp.py
  python3 scripts/analyze_mechanistic_llama_gsm_sp.py --raw results/raw/mechanistic_llama_gsm_sp_raw.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RAW = "results/raw/mechanistic_llama_gsm_sp_raw.csv"
OUT_SUMMARY = "results/derived/mechanistic_llama_gsm_sp_summary.csv"
OUT_BEHAVIOR = "results/derived/mechanistic_llama_behavior_link.csv"
OUT_GOLD = "results/derived/mechanistic_llama_gold_distribution.csv"


def _as_bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def _instance_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Collapse long (problem, variant, layer) → one row per instance at final layer."""
    n_layers = int(raw["n_layers"].iloc[0])
    final = raw[raw["layer"].astype(int) == (n_layers - 1)].copy()
    # If duplicates, keep last
    final = final.drop_duplicates(["problem_id", "variant_type"], keep="last")
    final["final_rank"] = pd.to_numeric(final["rank"], errors="coerce")
    final["final_logprob"] = pd.to_numeric(final["logprob"], errors="coerce")
    final["model_correct_bool"] = _as_bool(final["model_correct"])
    return final


def wilcoxon_summary(inst: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam in sorted(inst["family"].unique()):
        can = inst[(inst.family == fam) & (inst.variant_type == "canonical")].set_index("problem_id")
        for vt in ["W3", "W6"]:
            var = inst[(inst.family == fam) & (inst.variant_type == vt)].set_index("problem_id")
            ids = sorted(set(can.index) & set(var.index))
            a = can.loc[ids, "final_rank"].astype(float)
            b = var.loc[ids, "final_rank"].astype(float)
            mask = a.notna() & b.notna()
            a, b = a[mask], b[mask]
            n = int(len(a))
            med_can = float(a.median()) if n else float("nan")
            med_var = float(b.median()) if n else float("nan")
            w_two = p_two = w_greater = p_greater = float("nan")
            if n >= 1:
                try:
                    # Drop zero diffs for wilcox robustness when all ties
                    if np.allclose(a.to_numpy(), b.to_numpy()):
                        w_two = p_two = w_greater = p_greater = float("nan")
                    else:
                        w_two, p_two = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
                        # H1: canonical rank > variant rank (gold more accessible under rewrite)
                        w_greater, p_greater = stats.wilcoxon(
                            a, b, zero_method="wilcox", alternative="greater"
                        )
                except ValueError:
                    pass
            rows.append(
                {
                    "family": fam,
                    "variant": vt,
                    "n_paired": n,
                    "median_final_rank_canonical": med_can,
                    "median_final_rank_variant": med_var,
                    "wilcoxon_W_two_sided": w_two,
                    "wilcoxon_p_two_sided": p_two,
                    "wilcoxon_W_greater_can_gt_var": w_greater,
                    "wilcoxon_p_greater_can_gt_var": p_greater,
                }
            )
        # Canonical-only summary row (no paired test)
        can_only = inst[(inst.family == fam) & (inst.variant_type == "canonical")]
        rows.append(
            {
                "family": fam,
                "variant": "canonical",
                "n_paired": int(len(can_only)),
                "median_final_rank_canonical": float(can_only["final_rank"].median())
                if len(can_only)
                else float("nan"),
                "median_final_rank_variant": float("nan"),
                "wilcoxon_W_two_sided": float("nan"),
                "wilcoxon_p_two_sided": float("nan"),
                "wilcoxon_W_greater_can_gt_var": float("nan"),
                "wilcoxon_p_greater_can_gt_var": float("nan"),
            }
        )
    return pd.DataFrame(rows)


def behavior_link(inst: pd.DataFrame) -> pd.DataFrame:
    """Mann-Whitney U: final-layer gold rank for correct vs incorrect."""
    rows = []
    # Use all variants pooled per family (maximizes power); also report canonical-only.
    for fam in sorted(inst["family"].unique()):
        for slice_name, sub in [
            ("all_variants", inst[inst.family == fam]),
            ("canonical_only", inst[(inst.family == fam) & (inst.variant_type == "canonical")]),
        ]:
            ok = sub[sub["model_correct_bool"]]["final_rank"].dropna().astype(float)
            bad = sub[~sub["model_correct_bool"]]["final_rank"].dropna().astype(float)
            n_ok, n_bad = int(len(ok)), int(len(bad))
            u = p = float("nan")
            med_ok = float(ok.median()) if n_ok else float("nan")
            med_bad = float(bad.median()) if n_bad else float("nan")
            if n_ok >= 1 and n_bad >= 1:
                try:
                    # H1: correct instances have *lower* gold rank (more accessible)
                    u, p = stats.mannwhitneyu(ok, bad, alternative="less")
                except ValueError:
                    pass
            rows.append(
                {
                    "family": fam,
                    "slice": slice_name,
                    "n_correct": n_ok,
                    "n_incorrect": n_bad,
                    "median_final_rank_correct": med_ok,
                    "median_final_rank_incorrect": med_bad,
                    "mannwhitney_U": u,
                    "mannwhitney_p_correct_less_rank": p,
                    "alternative": "correct_final_rank < incorrect_final_rank",
                }
            )
    return pd.DataFrame(rows)


def gold_distribution(inst: pd.DataFrame) -> pd.DataFrame:
    """Frequency audit of distinct gold values (and decoded first tokens)."""
    rows = []
    for fam in sorted(inst["family"].unique()):
        # Deduplicate by problem+variant so each instance counted once
        sub = inst[inst.family == fam].copy()
        # Full gold_value distribution
        vc = sub["gold_value"].astype(str).value_counts()
        for gold, cnt in vc.items():
            rows.append(
                {
                    "family": fam,
                    "unit": "gold_value",
                    "value": gold,
                    "count": int(cnt),
                    "n_instances": int(len(sub)),
                    "fraction": float(cnt) / len(sub) if len(sub) else float("nan"),
                }
            )
        # First-token decoded distribution (what the rank probe actually targets)
        vc_tok = sub["gold_token_decoded"].astype(str).value_counts()
        for tok, cnt in vc_tok.items():
            rows.append(
                {
                    "family": fam,
                    "unit": "gold_first_token_decoded",
                    "value": tok,
                    "count": int(cnt),
                    "n_instances": int(len(sub)),
                    "fraction": float(cnt) / len(sub) if len(sub) else float("nan"),
                }
            )
        # Summary flags
        n_distinct_gold = int(sub["gold_value"].astype(str).nunique())
        n_distinct_tok = int(sub["gold_token_decoded"].astype(str).nunique())
        top_tok_frac = float(vc_tok.iloc[0] / len(sub)) if len(vc_tok) and len(sub) else float("nan")
        rows.append(
            {
                "family": fam,
                "unit": "SUMMARY",
                "value": (
                    f"n_distinct_gold_value={n_distinct_gold}; "
                    f"n_distinct_first_token={n_distinct_tok}; "
                    f"top_first_token_fraction={top_tok_frac:.3f}; "
                    f"digit0_dominated={top_tok_frac >= 0.5 and str(vc_tok.index[0]).strip() in {'0', ' 0'}}"
                ),
                "count": int(len(sub)),
                "n_instances": int(len(sub)),
                "fraction": 1.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=DEFAULT_RAW)
    ap.add_argument("--summary-out", default=OUT_SUMMARY)
    ap.add_argument("--behavior-out", default=OUT_BEHAVIOR)
    ap.add_argument("--gold-out", default=OUT_GOLD)
    args = ap.parse_args()

    raw_path = REPO_ROOT / args.raw
    if not raw_path.exists() or raw_path.stat().st_size == 0:
        raise SystemExit(f"Missing raw CSV: {raw_path} — run scripts/run_mechanistic_llama_gsm_sp.py first")

    raw = pd.read_csv(raw_path, dtype=str).fillna("")
    if raw.empty:
        raise SystemExit(f"Empty raw CSV: {raw_path}")

    inst = _instance_frame(raw)
    print(f"[load] {raw_path}  long_rows={len(raw)}  instances={len(inst)}")
    print(inst.groupby(["family", "variant_type"]).size().to_string())

    summary = wilcoxon_summary(inst)
    behavior = behavior_link(inst)
    gold = gold_distribution(inst)

    for path, df in [
        (REPO_ROOT / args.summary_out, summary),
        (REPO_ROOT / args.behavior_out, behavior),
        (REPO_ROOT / args.gold_out, gold),
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"[write] {path}  rows={len(df)}")

    print("\n=== Wilcoxon (canonical vs variant final-layer gold rank) ===")
    print(summary.to_string(index=False))
    print("\n=== Behavior-internals (Mann-Whitney: correct rank < incorrect rank) ===")
    print(behavior.to_string(index=False))
    print("\n=== Gold SUMMARY rows ===")
    print(gold[gold.unit == "SUMMARY"][["family", "value"]].to_string(index=False))


if __name__ == "__main__":
    main()
