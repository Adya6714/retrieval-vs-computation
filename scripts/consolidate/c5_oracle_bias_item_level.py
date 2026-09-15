#!/usr/bin/env python3
"""C5: Item-level selective oracle-bias test (upgrade from O13 n=3 sign test).

For each graded verifier defect, score before→after verdicts on perturbed rows
and matched canonical counterparts. Report mean Δ_pert, mean Δ_can, and excess
(Δ_pert − Δ_can) with cluster-bootstrap CIs under:
  (1) cluster_by_defect (3 clusters — low-powered; sensitivity)
  (2) cluster_by_problem_id (secondary)

Point-estimate target (O13 equal-weight defect means): +0.171 vs +0.054, excess +0.116.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

csv.field_size_limit(sys.maxsize)

from probes.common.variants import normalize_variant  # noqa: E402
from scripts.consolidate.oracle_bias_summary import (  # noqa: E402
    _bool_col,
    _load_bank,
    _load_raw_rows,
    _model_answer,
    _score_bw,
)

DER = REPO_ROOT / "results" / "derived"
OUT = DER / "C5_oracle_bias_item_level.csv"
INST_OUT = DER / "C5_oracle_bias_item_level_instances.csv"

N_BOOT = 5000
SEED = 42

MODEL_SHORT = {
    "anthropic/claude-sonnet-4": "Claude",
    "openai/gpt-4o": "GPT-4o",
    "google/gemini-2.5-flash": "Gemini",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
    "openai/o1-mini": "o4-mini",
}


def _short_model(m: str) -> str:
    m = str(m or "").strip()
    return MODEL_SHORT.get(m, m.split("/")[-1] if "/" in m else m)


def _round(x: float, nd: int = 4):
    if x != x:
        return ""
    return round(float(x), nd)


def cluster_bootstrap_mean(
    values: np.ndarray,
    cluster_ids: list[str],
    *,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Percentile CI + two-sided p for H0: mean = 0, resampling clusters."""
    xv = np.asarray(values, dtype=float)
    cids = [str(c) for c in cluster_ids]
    if len(xv) != len(cids):
        raise ValueError("length mismatch")
    estimate = float(np.mean(xv)) if len(xv) else float("nan")
    clusters = sorted(set(cids))
    grouped = {c: [i for i, cid in enumerate(cids) if cid == c] for c in clusters}
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        draw = rng.choice(clusters, size=len(clusters), replace=True)
        idx = [j for c in draw for j in grouped[c]]
        boots[i] = float(np.mean(xv[idx])) if idx else float("nan")
    finite = boots[np.isfinite(boots)]
    if len(finite) == 0:
        return {
            "estimate": estimate,
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "p_clustered": float("nan"),
            "n": len(xv),
            "n_clusters": len(clusters),
        }
    left = float(np.mean(finite <= 0.0))
    right = float(np.mean(finite >= 0.0))
    p = 2.0 * min(left, right)
    p = float(max(p, 1.0 / len(finite)))
    return {
        "estimate": estimate,
        "ci_low": float(np.percentile(finite, 2.5)),
        "ci_high": float(np.percentile(finite, 97.5)),
        "p_clustered": p,
        "n": len(xv),
        "n_clusters": len(clusters),
    }


def _delta(before: bool | None, after: bool | None) -> float | None:
    if before is None or after is None:
        return None
    return float(after) - float(before)


def build_sp_instances() -> pd.DataFrame:
    """ALGO SP W3: old_verified → rescored_correct; matched canonical."""
    bank = _load_bank()
    rows: list[dict] = []
    for path in sorted(DER.glob("ALGO_P1_*_rescored.csv")):
        df = pd.read_csv(path, dtype=str).fillna("")
        if "included" not in df.columns:
            continue
        df = df[df["included"].str.lower().eq("true")]
        for variant in ("W3", "canonical"):
            sub = df[df["variant_type"].map(normalize_variant).eq(variant)]
            for _, row in sub.iterrows():
                pid = str(row["problem_id"])
                key = ("ALGO", pid, variant)
                if key not in bank:
                    continue
                if str(bank[key].get("problem_subtype", "")).lower() != "shortest_path":
                    continue
                before = _bool_col(row.get("old_verified", row.get("verified", "")))
                after = _bool_col(row.get("rescored_correct", ""))
                if before is None:
                    before = _bool_col(row.get("behavioral_correct", ""))
                rows.append(
                    {
                        "defect": "ALGO_SP_W3_node_mapping",
                        "family": "ALGO",
                        "problem_id": pid,
                        "model": _short_model(row.get("model", "")),
                        "variant": variant,
                        "role": "perturbed" if variant == "W3" else "canonical",
                        "before": before,
                        "after": after,
                        "delta": _delta(before, after),
                        "source": path.name,
                    }
                )
    return pd.DataFrame(rows)


def build_bw_instances() -> tuple[pd.DataFrame, pd.DataFrame]:
    """BW action-mapping (W3 vs can) and BW state-parser (all variants vs can)."""
    bank = _load_bank()
    map_rows: list[dict] = []
    parser_rows: list[dict] = []
    for raw in _load_raw_rows():
        fam = raw["_family"]
        if fam != "BW":
            continue
        pid = str(raw.get("problem_id", "")).strip()
        variant = normalize_variant(raw.get("variant_type", ""))
        key = (fam, pid, variant)
        if key not in bank or not _model_answer(raw).strip():
            continue
        brow = bank[key]
        if str(brow.get("problem_subtype", "")).lower() != "blocksworld":
            continue
        if pid.upper().startswith("MBW"):
            continue
        model = _short_model(raw.get("model", ""))

        # Action mapping: W3 + canonical only
        if variant in {"W3", "canonical"}:
            before = _score_bw(raw, brow, use_action_mapping=False, parser_mode="after")
            after = _score_bw(raw, brow, use_action_mapping=True, parser_mode="after")
            map_rows.append(
                {
                    "defect": "BW_W3_action_mapping",
                    "family": "BW",
                    "problem_id": pid,
                    "model": model,
                    "variant": variant,
                    "role": "perturbed" if variant == "W3" else "canonical",
                    "before": before,
                    "after": after,
                    "delta": _delta(before, after),
                    "source": "BW_P1_behavioral raw rescore",
                }
            )

        # State parser: all variants; non-canonical = perturbed role
        before_p = _score_bw(raw, brow, use_action_mapping=True, parser_mode="before")
        after_p = _score_bw(raw, brow, use_action_mapping=True, parser_mode="after")
        parser_rows.append(
            {
                "defect": "BW_legacy_state_parser",
                "family": "BW",
                "problem_id": pid,
                "model": model,
                "variant": variant,
                "role": "canonical" if variant == "canonical" else "perturbed",
                "before": before_p,
                "after": after_p,
                "delta": _delta(before_p, after_p),
                "source": "BW_P1_behavioral raw rescore",
            }
        )
    return pd.DataFrame(map_rows), pd.DataFrame(parser_rows)


def make_pairs(inst: pd.DataFrame, *, aggregate_perturbed: bool = False) -> pd.DataFrame:
    """Match perturbed rows to canonical on (defect, model, problem_id).

    If aggregate_perturbed=True (BW_state_parser style), collapse all non-canonical
    variants for a problem×model to their mean delta so each pair is 1:1.
    """
    pert = inst[inst["role"] == "perturbed"].copy()
    can = inst[inst["role"] == "canonical"].copy()
    can = can.drop_duplicates(["defect", "model", "problem_id"], keep="last")
    can = can.rename(
        columns={
            "before": "can_before",
            "after": "can_after",
            "delta": "delta_canonical",
            "variant": "canonical_variant",
        }
    )

    if aggregate_perturbed:
        # Mean delta across perturbed variants; flip if any variant flipped
        agg = (
            pert.groupby(["defect", "family", "model", "problem_id"], as_index=False)
            .agg(
                delta_perturbed=("delta", "mean"),
                pert_before=("before", "mean"),  # fraction correct before
                pert_after=("after", "mean"),
                n_perturbed_variants=("delta", "count"),
                pert_flipped=("delta", lambda s: bool((s.fillna(0) != 0).any())),
            )
        )
        # Reconstruct boolean-ish before/after from means not needed for excess
        merged = agg.merge(
            can[
                [
                    "defect",
                    "model",
                    "problem_id",
                    "can_before",
                    "can_after",
                    "delta_canonical",
                ]
            ],
            on=["defect", "model", "problem_id"],
            how="inner",
        )
        merged["perturbed_variant"] = "all_noncanonical_mean"
        merged["pert_before"] = merged["pert_before"]  # float fraction
        merged["pert_after"] = merged["pert_after"]
    else:
        merged = pert.merge(
            can[
                [
                    "defect",
                    "model",
                    "problem_id",
                    "can_before",
                    "can_after",
                    "delta_canonical",
                ]
            ],
            on=["defect", "model", "problem_id"],
            how="inner",
        )
        merged = merged.rename(
            columns={
                "before": "pert_before",
                "after": "pert_after",
                "delta": "delta_perturbed",
                "variant": "perturbed_variant",
            }
        )
        merged["pert_flipped"] = merged["pert_before"] != merged["pert_after"]
        merged["n_perturbed_variants"] = 1

    merged = merged.dropna(subset=["delta_perturbed", "delta_canonical"])
    merged["excess"] = merged["delta_perturbed"] - merged["delta_canonical"]
    if "pert_flipped" not in merged.columns:
        merged["pert_flipped"] = merged["delta_perturbed"] != 0
    merged["can_flipped"] = merged["can_before"] != merged["can_after"]
    merged["either_flipped"] = merged["pert_flipped"] | merged["can_flipped"]
    return merged


def unpaired_defect_means_o13_style(inst: pd.DataFrame) -> pd.DataFrame:
    """Per-defect (acc_after − acc_before) with O13 denominators (Nones dropped per side)."""

    def _acc(series: pd.Series) -> tuple[float, int]:
        vals = []
        for v in series:
            if pd.isna(v):
                continue
            s = str(v).strip().lower()
            if s in {"true", "1"}:
                vals.append(True)
            elif s in {"false", "0"}:
                vals.append(False)
        if not vals:
            return float("nan"), 0
        return float(sum(vals) / len(vals)), len(vals)

    rows = []
    for defect, g in inst.groupby("defect"):
        pert = g[g["role"] == "perturbed"]
        can = g[g["role"] == "canonical"]
        pb, nb = _acc(pert["before"])
        pa, na = _acc(pert["after"])
        cb, ncb = _acc(can["before"])
        ca, nca = _acc(can["after"])
        affected = g.dropna(subset=["before", "after"])
        # before/after may be bool; require both non-null and unequal
        aff = 0
        for _, r in g.iterrows():
            if pd.isna(r["before"]) or pd.isna(r["after"]):
                continue
            if bool(r["before"]) != bool(r["after"]):
                aff += 1
        rows.append(
            {
                "defect": defect,
                "n_perturbed": len(pert),
                "n_canonical": len(can),
                "n_before_pert": nb,
                "n_after_pert": na,
                "n_affected_rows": aff,
                "mean_delta_perturbed": pa - pb if pa == pa and pb == pb else float("nan"),
                "mean_delta_canonical": ca - cb if ca == ca and cb == cb else float("nan"),
                "acc_before_perturbed": pb,
                "acc_after_perturbed": pa,
                "acc_before_canonical": cb,
                "acc_after_canonical": ca,
            }
        )
    return pd.DataFrame(rows)

def _result_row(
    *,
    analysis: str,
    clustering: str,
    subset: str,
    pairs: pd.DataFrame,
    note: str = "",
) -> dict:
    if pairs.empty:
        return {
            "analysis": analysis,
            "subset": subset,
            "clustering": clustering,
            "n_pairs": 0,
            "n_clusters": 0,
            "mean_delta_perturbed": "",
            "mean_delta_canonical": "",
            "excess_rate_difference": "",
            "ci_low": "",
            "ci_high": "",
            "p_clustered": "",
            "p_value_method": "cluster_bootstrap_H0_mean=0",
            "n_boot": N_BOOT,
            "seed": SEED,
            "ci_excludes_zero": "",
            "note": note or "empty subset",
        }
    if clustering == "defect":
        clusters = pairs["defect"].astype(str).tolist()
        boot_label = "cluster_by_defect"
    elif clustering == "problem_id":
        # family:problem_id so BW/ALGO same numeric ids do not collide
        clusters = (pairs["family"].astype(str) + ":" + pairs["problem_id"].astype(str)).tolist()
        boot_label = "cluster_by_problem_id"
    else:
        raise ValueError(clustering)

    mean_p = float(pairs["delta_perturbed"].mean())
    mean_c = float(pairs["delta_canonical"].mean())
    res = cluster_bootstrap_mean(
        pairs["excess"].to_numpy(dtype=float),
        clusters,
        n_boot=N_BOOT,
        seed=SEED,
    )
    excl = (
        bool(res["ci_low"] > 0 or res["ci_high"] < 0)
        if res["ci_low"] == res["ci_low"] and res["ci_high"] == res["ci_high"]
        else ""
    )
    low_pow = ""
    if clustering == "defect":
        low_pow = (
            "NOTE: clustering on 3 defects is itself low-powered "
            "(same limitation as O13 sign test); prefer problem_id clustering for power. "
        )
    return {
        "analysis": analysis,
        "subset": subset,
        "clustering": boot_label,
        "n_pairs": res["n"],
        "n_clusters": res["n_clusters"],
        "mean_delta_perturbed": _round(mean_p),
        "mean_delta_canonical": _round(mean_c),
        "excess_rate_difference": _round(res["estimate"]),
        "ci_low": _round(res["ci_low"]),
        "ci_high": _round(res["ci_high"]),
        "p_clustered": _round(res["p_clustered"]),
        "p_value_method": "cluster_bootstrap_H0_mean_excess=0",
        "n_boot": N_BOOT,
        "seed": SEED,
        "ci_excludes_zero": excl,
        "note": (
            low_pow
            + note
            + f" O13 defect-equal-weight reference: Δ_pert≈0.171 Δ_can≈0.054 excess≈0.116."
        ).strip(),
    }


def defect_equal_weight_reference(pairs: pd.DataFrame) -> dict:
    """Reproduce O13 equal-weight-across-defects point estimate from item pairs."""
    rows = []
    for defect, g in pairs.groupby("defect"):
        rows.append(
            {
                "defect": defect,
                "n_pairs": len(g),
                "mean_delta_perturbed": float(g["delta_perturbed"].mean()),
                "mean_delta_canonical": float(g["delta_canonical"].mean()),
                "excess": float(g["excess"].mean()),
            }
        )
    d = pd.DataFrame(rows)
    return {
        "analysis": "o13_equal_weight_defect_means",
        "subset": "all_matched_pairs",
        "clustering": "none_point_estimate",
        "n_pairs": int(len(pairs)),
        "n_clusters": int(d["defect"].nunique()),
        "mean_delta_perturbed": _round(float(d["mean_delta_perturbed"].mean())),
        "mean_delta_canonical": _round(float(d["mean_delta_canonical"].mean())),
        "excess_rate_difference": _round(float(d["excess"].mean())),
        "ci_low": "",
        "ci_high": "",
        "p_clustered": "",
        "p_value_method": "none",
        "n_boot": "",
        "seed": SEED,
        "ci_excludes_zero": "",
        "note": (
            "Equal-weight average of per-defect mean deltas (O13 TABLE A / sign-test input). "
            f"Per-defect: "
            + "; ".join(
                f"{r.defect}: Δp={r.mean_delta_perturbed:.4f} Δc={r.mean_delta_canonical:.4f} "
                f"excess={r.excess:.4f} n={r.n_pairs}"
                for r in d.itertuples()
            )
        ),
    }


def main() -> None:
    print("Scoring SP rescored instances...")
    sp = build_sp_instances()
    print(f"  SP rows: {len(sp)}")
    print("Scoring BW mapping + parser (slow: re-verify)...")
    bw_map, bw_parser = build_bw_instances()
    print(f"  BW map rows: {len(bw_map)}; BW parser rows: {len(bw_parser)}")

    inst = pd.concat([sp, bw_map, bw_parser], ignore_index=True)
    # Full instance dump for audit
    inst_path = DER / "C5_oracle_bias_scored_instances.csv"
    inst.to_csv(inst_path, index=False)
    print(f"Wrote {inst_path} ({len(inst)} scored rows)")

    unpaired = unpaired_defect_means_o13_style(inst)
    print("Unpaired O13-style defect means (should match O13):")
    print(unpaired.to_string(index=False))

    # Hybrid pairing: 1:1 for SP + BW-W3; aggregate non-canonical variants for BW parser
    pairs_sp = make_pairs(sp, aggregate_perturbed=False)
    pairs_map = make_pairs(bw_map, aggregate_perturbed=False)
    pairs_parser = make_pairs(bw_parser, aggregate_perturbed=True)
    pairs = pd.concat([pairs_sp, pairs_map, pairs_parser], ignore_index=True)
    print(f"Matched pairs: {len(pairs)}")
    print(pairs.groupby("defect").size())
    print(
        "Flips: pert",
        int(pairs["pert_flipped"].sum()),
        "can",
        int(pairs["can_flipped"].sum()),
        "either",
        int(pairs["either_flipped"].sum()),
    )
    print(
        "Affected scored rows (O13 rows_affected target 348):",
        int(unpaired["n_affected_rows"].sum()),
    )

    pairs.to_csv(INST_OUT, index=False)

    out_rows: list[dict] = []

    # Exact O13 unpaired equal-weight reference (acc_after−acc_before per side)
    out_rows.append(
        {
            "analysis": "o13_unpaired_equal_weight_defect_means",
            "subset": "all_scored_rows",
            "clustering": "none_point_estimate",
            "n_pairs": int(unpaired["n_affected_rows"].sum()),
            "n_clusters": int(len(unpaired)),
            "mean_delta_perturbed": _round(float(unpaired["mean_delta_perturbed"].mean())),
            "mean_delta_canonical": _round(float(unpaired["mean_delta_canonical"].mean())),
            "excess_rate_difference": _round(
                float(
                    unpaired["mean_delta_perturbed"].mean()
                    - unpaired["mean_delta_canonical"].mean()
                )
            ),
            "ci_low": "",
            "ci_high": "",
            "p_clustered": "",
            "p_value_method": "none",
            "n_boot": "",
            "seed": SEED,
            "ci_excludes_zero": "",
            "note": (
                "O13 definition: per-defect (acc_after−acc_before) with Nones dropped "
                "per side (BW_parser before often None when legacy parse fails), then "
                "equal-weight average across 3 defects. "
                f"n_affected_rows={int(unpaired['n_affected_rows'].sum())} "
                f"(target 64+92+192=348). "
                + "; ".join(
                    f"{r.defect}: Δp={r.mean_delta_perturbed:.4f} Δc={r.mean_delta_canonical:.4f} "
                    f"aff={r.n_affected_rows}"
                    for r in unpaired.itertuples()
                )
            ),
        }
    )

    out_rows.append(defect_equal_weight_reference(pairs))

    # Primary: all matched pairs
    for clustering in ("defect", "problem_id"):
        out_rows.append(
            _result_row(
                analysis="paired_excess_delta_pert_minus_delta_can",
                clustering=clustering,
                subset="all_matched_pairs",
                pairs=pairs,
                note=(
                    "Matched (model×problem) pairs; BW_legacy_state_parser aggregates "
                    "non-canonical variants to mean Δ before pairing."
                ),
            )
        )

    flipped = pairs[pairs["pert_flipped"]].copy()
    for clustering in ("defect", "problem_id"):
        out_rows.append(
            _result_row(
                analysis="paired_excess_delta_pert_minus_delta_can",
                clustering=clustering,
                subset="perturbed_verdict_flipped",
                pairs=flipped,
                note=(
                    f"Subset where perturbed Δ≠0 (n={len(flipped)} pairs). "
                    "Selective-bias arm among the affected scored rows."
                ),
            )
        )

    either = pairs[pairs["either_flipped"]].copy()
    for clustering in ("defect", "problem_id"):
        out_rows.append(
            _result_row(
                analysis="paired_excess_delta_pert_minus_delta_can",
                clustering=clustering,
                subset="either_verdict_flipped",
                pairs=either,
                note="Pairs where pert and/or canonical verdict flipped.",
            )
        )

    for defect, g in pairs.groupby("defect"):
        out_rows.append(
            {
                "analysis": "per_defect_descriptive",
                "subset": "all_matched_pairs",
                "clustering": f"defect={defect}",
                "n_pairs": len(g),
                "n_clusters": 1,
                "mean_delta_perturbed": _round(float(g["delta_perturbed"].mean())),
                "mean_delta_canonical": _round(float(g["delta_canonical"].mean())),
                "excess_rate_difference": _round(float(g["excess"].mean())),
                "ci_low": "",
                "ci_high": "",
                "p_clustered": "",
                "p_value_method": "none",
                "n_boot": "",
                "seed": SEED,
                "ci_excludes_zero": "",
                "note": (
                    f"n_pert_flipped={int(g['pert_flipped'].sum())}; "
                    f"n_can_flipped={int(g['can_flipped'].sum())}"
                ),
            }
        )

    sign = pd.read_csv(DER / "O13_oracle_bias_sign_test.csv", dtype=str).fillna("")
    if len(sign):
        r = sign.iloc[0]
        out_rows.append(
            {
                "analysis": "reference_o13_defect_sign_test",
                "subset": "n_graded_defects=3",
                "clustering": "defect_sign_test",
                "n_pairs": r.get("n_graded_defects", "3"),
                "n_clusters": 3,
                "mean_delta_perturbed": r.get("mean_perturbed_delta", ""),
                "mean_delta_canonical": r.get("mean_canonical_delta", ""),
                "excess_rate_difference": r.get("mean_excess_perturbed_minus_canonical", ""),
                "ci_low": "",
                "ci_high": "",
                "p_clustered": r.get("sign_test_p_two_sided", ""),
                "p_value_method": "exact_binomial_sign_test",
                "n_boot": "",
                "seed": "",
                "ci_excludes_zero": "",
                "note": (
                    "UNDERPOWERED reference (n=3, p=0.25). C5 replaces this with item-level "
                    "rate difference + cluster-bootstrap CI. Clustering on 3 defects remains "
                    "low-powered — report alongside problem_id clustering."
                ),
            }
        )

    out = pd.DataFrame(out_rows)
    out.to_csv(OUT, index=False)
    print(f"\nWrote {OUT} ({len(out)} rows)")
    print(f"Wrote {INST_OUT} ({len(pairs)} pairs)")
    show = out[
        out["analysis"].isin(
            [
                "o13_unpaired_equal_weight_defect_means",
                "o13_equal_weight_defect_means",
                "paired_excess_delta_pert_minus_delta_can",
                "reference_o13_defect_sign_test",
            ]
        )
    ]
    print(
        show[
            [
                "analysis",
                "subset",
                "clustering",
                "n_pairs",
                "n_clusters",
                "mean_delta_perturbed",
                "mean_delta_canonical",
                "excess_rate_difference",
                "ci_low",
                "ci_high",
                "p_clustered",
                "ci_excludes_zero",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
