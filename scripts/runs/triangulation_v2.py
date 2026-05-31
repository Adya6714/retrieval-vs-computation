#!/usr/bin/env python3
"""Triangulation v2: k-of-n vote labels + threshold sensitivity sweep.

Builds on the full P1/P2/P3 signal inventory (no new API runs) and compares
against the legacy 5-field AND rule in ALGO_P3_SCR_triangulation.py.

Outputs:
    results/derived/triangulation_v2_labels.csv
    results/derived/triangulation_threshold_sweep.csv
    results/derived/triangulation_v2_summary.md
"""

from __future__ import annotations

import ast
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

RAW = ROOT / "results" / "raw"
DER = ROOT / "results" / "derived"

SHORT = {
    "anthropic/claude-sonnet-4": "Claude",
    "google/gemini-2.5-flash": "Gemini",
    "openai/gpt-4o": "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct": "Llama",
    "openai/o4-mini": "o4-mini",
}


def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception:
        return None


def _to_bool(x: object) -> bool | None:
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _norm_variant(v: object) -> str:
    s = str(v).strip()
    if not s:
        return s
    if s.lower() == "canonical":
        return "canonical"
    if s.lower().startswith("w") and s[1:].isdigit():
        return f"W{s[1:]}"
    return s


def load_p1_long() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    tags = {
        "anthropic/claude-sonnet-4": "claude",
        "google/gemini-2.5-flash": "gemini",
        "openai/gpt-4o": "gpt4o",
        "meta-llama/llama-3.1-8b-instruct": "llama",
        "openai/o4-mini": "o1mini",
    }
    for fam in ["ALGO", "GSM", "BW"]:
        for model, tag in tags.items():
            path = RAW / f"{fam}_P1_behavioral_{tag}.csv"
            df = _safe_read(path)
            if df is None or "variant_type" not in df.columns:
                continue
            df = df.copy()
            df["family"] = fam
            df["model"] = df.get("model", model).replace("", model)
            df["variant_type"] = df["variant_type"].map(_norm_variant)
            if "verified" in df.columns:
                corr = df["verified"].map(_to_bool)
            elif "behavioral_correct" in df.columns:
                corr = df["behavioral_correct"].map(_to_bool)
            else:
                corr = pd.Series([None] * len(df))
            fail = df.get("raw_response", pd.Series([""] * len(df))).astype(str).str.startswith("ERROR:")
            df["correct"] = corr.where(~fail, other=False)
            parts.append(df[["family", "problem_id", "model", "variant_type", "correct"]])

    bw = _safe_read(RAW / "BW_P1_behavioral.csv")
    if bw is not None and "model" in bw.columns and "variant_type" in bw.columns:
        bw = bw.copy()
        bw["family"] = "BW"
        bw["variant_type"] = bw["variant_type"].map(_norm_variant)
        corr = bw.get("behavioral_correct", pd.Series([""] * len(bw))).map(_to_bool)
        fail = bw.get("raw_response", pd.Series([""] * len(bw))).astype(str).str.startswith("ERROR:")
        bw["correct"] = corr.where(~fail, other=False)
        parts.append(bw[["family", "problem_id", "model", "variant_type", "correct"]])

    if not parts:
        return pd.DataFrame(columns=["family", "problem_id", "model", "variant_type", "correct"])
    out = pd.concat(parts, ignore_index=True).drop_duplicates(
        ["family", "problem_id", "model", "variant_type"], keep="last"
    )
    return out


def load_algo_cci_per_instance() -> pd.DataFrame:
    df = _safe_read(DER / "ALGO_P2_per_instance_cci.csv")
    if df is None or df.empty:
        return pd.DataFrame(
            columns=["family", "problem_id", "model", "cci_composite", "match_first", "cci_crit", "cci_alg"]
        )
    df = df.copy()
    df["family"] = "ALGO"
    for col in ["cci_composite", "match_first", "cci_crit", "cci_alg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["family", "problem_id", "model", "cci_composite", "match_first", "cci_crit", "cci_alg"]]


def load_gsm_cci() -> pd.DataFrame:
    from scripts.runs.coverage_audit import load_gsm_p2_merged

    df = load_gsm_p2_merged()
    if df is None or df.empty:
        return pd.DataFrame(columns=["family", "problem_id", "model", "cci_composite"])
    df = df.copy()
    df["family"] = "GSM"
    df["cci_composite"] = pd.to_numeric(df["cci_score"], errors="coerce")
    return df[["family", "problem_id", "model", "cci_composite"]]


def load_injection_final(mode: str) -> pd.DataFrame:
    path = RAW / (
        "ALGO_P2_phase2_injected.csv" if mode == "plausible" else "ALGO_P2_phase2_injected_implausible.csv"
    )
    df = _safe_read(path)
    if df is None or df.empty:
        return pd.DataFrame(columns=["family", "problem_id", "model", "impl_post_correct"])
    df = df[df["post_injection_correct"].astype(str).str.strip() != ""].copy()
    df["family"] = "ALGO"
    df["impl_post_correct"] = df["post_injection_correct"].map(_to_bool)
    return df[["family", "problem_id", "model", "impl_post_correct"]].drop_duplicates()


def load_contamination() -> pd.DataFrame:
    parts = []
    for fam in ["ALGO", "GSM", "BW"]:
        c = _safe_read(RAW / f"{fam}_P3_contamination.csv")
        if c is None or c.empty:
            continue
        score_col = "instance_contamination_score" if "instance_contamination_score" in c.columns else "contamination_score"
        c = c[["problem_id", score_col]].copy()
        c[score_col] = pd.to_numeric(c[score_col], errors="coerce")
        c = c.groupby("problem_id", as_index=False)[score_col].mean()
        c["family"] = fam
        c = c.rename(columns={score_col: "contam"})
        parts.append(c)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["family", "problem_id", "contam"])


def load_mechanistic() -> pd.DataFrame:
    parts = []
    for fam in ["ALGO", "GSM", "BW"]:
        m = _safe_read(RAW / f"{fam}_P3_mechanistic.csv")
        if m is None or m.empty:
            continue
        m = m.copy()
        m["family"] = fam
        m["crystallization_layer_num"] = pd.to_numeric(m.get("crystallization_layer", ""), errors="coerce")
        m["n_layers_num"] = pd.to_numeric(m.get("n_layers_processed", ""), errors="coerce")
        m["crystallization_depth_norm"] = m["crystallization_layer_num"] / m["n_layers_num"].replace(0, np.nan)
        parts.append(m[["family", "problem_id", "model", "crystallization_depth_norm"]])
    if not parts:
        return pd.DataFrame(columns=["family", "problem_id", "model", "crystallization_depth_norm"])
    return pd.concat(parts, ignore_index=True).drop_duplicates(["family", "problem_id", "model"], keep="last")


def load_legacy_algo_labels() -> pd.DataFrame:
    df = _safe_read(DER / "ALGO_P3_triangulation.csv")
    if df is None or df.empty:
        return pd.DataFrame(columns=["problem_id", "model", "legacy_label"])
    out = df[["problem_id", "model", "convergence_label"]].copy()
    out = out.rename(columns={"convergence_label": "legacy_label"})
    return out.drop_duplicates(["problem_id", "model"], keep="last")


def build_signal_frame(p1: pd.DataFrame) -> pd.DataFrame:
    wide = p1.pivot_table(
        index=["family", "problem_id", "model"], columns="variant_type", values="correct", aggfunc="last"
    ).reset_index()
    if "canonical" not in wide.columns:
        return pd.DataFrame()

    for v in ["W1", "W2", "W3", "W4", "W5", "W6"]:
        if v not in wide.columns:
            wide[v] = np.nan

    wide["canonical_bool"] = wide["canonical"].map(_to_bool)
    for v in ["W1", "W2", "W3", "W4", "W5", "W6"]:
        wide[f"{v.lower()}_bool"] = wide[v].map(_to_bool)

    wide = wide.merge(load_contamination(), on=["family", "problem_id"], how="left")
    wide = wide.merge(load_algo_cci_per_instance(), on=["family", "problem_id", "model"], how="left")
    gsm = load_gsm_cci()
    if not gsm.empty:
        wide = wide.merge(
            gsm.rename(columns={"cci_composite": "cci_gsm"}),
            on=["family", "problem_id", "model"],
            how="left",
        )
    else:
        wide["cci_gsm"] = np.nan
    wide["cci_effective"] = wide["cci_composite"]
    wide.loc[wide["family"] == "GSM", "cci_effective"] = wide.loc[wide["family"] == "GSM", "cci_gsm"]

    inj = load_injection_final("implausible")
    wide = wide.merge(inj, on=["family", "problem_id", "model"], how="left")
    wide = wide.merge(load_mechanistic(), on=["family", "problem_id", "model"], how="left")

    wide["var_canonical"] = wide["canonical_bool"].map(lambda x: 1.0 if x is True else 0.0 if x is False else np.nan)
    wide["var_w3"] = wide["w3_bool"].map(lambda x: 1.0 if x is True else 0.0 if x is False else np.nan)
    wide["vri_gap"] = wide["var_canonical"] - wide["var_w3"]
    return wide


@dataclass(frozen=True)
class TriThresholds:
    w3_retrieval_max: float = 0.2
    w3_computation_min: float = 0.5
    contam_retrieval_min: float = 0.6
    contam_computation_max: float = 0.4
    cci_computation_min: float = 0.5
    cci_retrieval_max: float = 0.3
    depth_computation_min: float = 0.55
    min_votes: int = 2
    vote_margin: int = 2


def apply_votes(df: pd.DataFrame, th: TriThresholds) -> pd.DataFrame:
    out = df.copy()
    cb = out["canonical_bool"]
    w3 = out["w3_bool"]
    w1 = out["w1_bool"]
    w2 = out["w2_bool"]
    w4 = out["w4_bool"]
    w5 = out["w5_bool"]
    w6 = out["w6_bool"]
    contam = pd.to_numeric(out["contam"], errors="coerce")
    cci = pd.to_numeric(out["cci_effective"], errors="coerce")
    match_first = pd.to_numeric(out.get("match_first", np.nan), errors="coerce")
    cci_crit = pd.to_numeric(out.get("cci_crit", np.nan), errors="coerce")
    impl = out.get("impl_post_correct")
    depth = pd.to_numeric(out.get("crystallization_depth_norm", np.nan), errors="coerce")

    votes: dict[str, pd.Series] = {}

    # P1 variant votes
    votes["p1_rename_fragile"] = ((cb == True) & (out["var_w3"] < th.w3_retrieval_max)).astype(float)  # noqa: E712
    votes["p1_w3_keep"] = ((cb == True) & (out["var_w3"] > th.w3_computation_min)).astype(float)  # noqa: E712
    votes["p1_multi_variant"] = (
        (cb == True)
        & (w3 == True)
        & ((w1 == True) | (w2 == True) | (w4 == True) | (w5 == True) | (w6 == True))  # noqa: E712
    ).astype(float)
    votes["p1_vri_high"] = (out["vri_gap"] > 0.5).astype(float)

    # P2 votes
    votes["p2_cci_comp"] = (cci >= th.cci_computation_min).astype(float)
    votes["p2_cci_retr"] = (cci <= th.cci_retrieval_max).astype(float)
    votes["p2_match_first"] = (match_first >= 0.5).astype(float)
    votes["p2_crit_step"] = (cci_crit >= 0.5).astype(float)
    votes["p2_impl_recovery"] = (impl == True).astype(float)  # noqa: E712
    votes["p2_impl_fail"] = (impl == False).astype(float)  # noqa: E712

    # P3 votes
    votes["p3_contam_high"] = (contam >= th.contam_retrieval_min).astype(float)
    votes["p3_contam_low"] = (contam <= th.contam_computation_max).astype(float)
    votes["p3_depth_high"] = (depth >= th.depth_computation_min).astype(float)

    # Map to retrieval/computation tallies
    retrieval_keys = [
        "p1_rename_fragile",
        "p1_vri_high",
        "p2_cci_retr",
        "p2_impl_fail",
        "p3_contam_high",
    ]
    computation_keys = [
        "p1_w3_keep",
        "p1_multi_variant",
        "p2_cci_comp",
        "p2_match_first",
        "p2_crit_step",
        "p2_impl_recovery",
        "p3_contam_low",
        "p3_depth_high",
    ]

    vote_df = pd.DataFrame({k: votes[k].fillna(0.0) for k in votes})
    out = pd.concat([out, vote_df], axis=1)

    out["retrieval_votes"] = out[retrieval_keys].sum(axis=1)
    out["computation_votes"] = out[computation_keys].sum(axis=1)
    out["votes_total"] = out["retrieval_votes"] + out["computation_votes"]
    out["tri_score"] = out["computation_votes"] - out["retrieval_votes"]

    out["tri_v2_label"] = "mixed"
    insufficient = out["votes_total"] < th.min_votes
    out.loc[insufficient, "tri_v2_label"] = "insufficient"
    strong_comp = (~insufficient) & (out["tri_score"] >= th.vote_margin)
    strong_retr = (~insufficient) & (out["tri_score"] <= -th.vote_margin)
    weak_comp = (~insufficient) & (out["tri_score"] == th.vote_margin - 1)
    weak_retr = (~insufficient) & (out["tri_score"] == -(th.vote_margin - 1))
    out.loc[strong_comp, "tri_v2_label"] = "computation"
    out.loc[strong_retr, "tri_v2_label"] = "retrieval"
    out.loc[weak_comp, "tri_v2_label"] = "weak_computation"
    out.loc[weak_retr, "tri_v2_label"] = "weak_retrieval"

    out["tri_v2_confidence"] = (out["tri_score"].abs() / out["votes_total"].replace(0, np.nan)).fillna(0.0)
    return out


def sweep_thresholds(base: pd.DataFrame, legacy_algo: pd.DataFrame) -> pd.DataFrame:
    w3_ret = [0.2, 0.3, 0.5]
    w3_comp = [0.5, 0.7]
    contam_hi = [0.5, 0.6, 0.7]
    contam_lo = [0.3, 0.4, 0.5]
    cci_hi = [0.4, 0.5, 0.6]
    cci_lo = [0.25, 0.3, 0.4]
    min_votes = [2, 3]
    margins = [1, 2]

    rows: list[dict] = []
    for i, combo in enumerate(
        itertools.product(w3_ret, w3_comp, contam_hi, contam_lo, cci_hi, cci_lo, min_votes, margins)
    ):
        th = TriThresholds(
            w3_retrieval_max=combo[0],
            w3_computation_min=combo[1],
            contam_retrieval_min=combo[2],
            contam_computation_max=combo[3],
            cci_computation_min=combo[4],
            cci_retrieval_max=combo[5],
            min_votes=combo[6],
            vote_margin=combo[7],
        )
        labeled = apply_votes(base, th)
        n = len(labeled)
        dist = labeled["tri_v2_label"].value_counts(normalize=True)
        strong = labeled["tri_v2_label"].isin(["retrieval", "computation"]).mean()

        row = {
            "param_id": i,
            "w3_retrieval_max": th.w3_retrieval_max,
            "w3_computation_min": th.w3_computation_min,
            "contam_retrieval_min": th.contam_retrieval_min,
            "contam_computation_max": th.contam_computation_max,
            "cci_computation_min": th.cci_computation_min,
            "cci_retrieval_max": th.cci_retrieval_max,
            "min_votes": th.min_votes,
            "vote_margin": th.vote_margin,
            "n_rows": n,
            "pct_insufficient": float(dist.get("insufficient", 0.0)),
            "pct_mixed": float(dist.get("mixed", 0.0)),
            "pct_retrieval": float(dist.get("retrieval", 0.0)),
            "pct_computation": float(dist.get("computation", 0.0)),
            "pct_strong_total": float(strong),
            "pct_weak_total": float(
                labeled["tri_v2_label"].isin(["weak_retrieval", "weak_computation"]).mean()
            ),
        }

        algo = labeled[labeled["family"] == "ALGO"].copy()
        if not legacy_algo.empty and not algo.empty:
            cmp = algo.merge(legacy_algo, on=["problem_id", "model"], how="inner")
            if not cmp.empty:
                cmp["legacy_strong"] = cmp["legacy_label"].isin(["retrieval_signal", "computation_signal"])
                cmp["v2_strong"] = cmp["tri_v2_label"].isin(["retrieval", "computation"])
                row["algo_n_overlap"] = len(cmp)
                row["legacy_strong_rate"] = float(cmp["legacy_strong"].mean())
                row["v2_strong_rate_algo"] = float(cmp["v2_strong"].mean())
                row["strong_label_jaccard"] = float(
                    np.mean(cmp["legacy_strong"] & cmp["v2_strong"])
                    / max(1, np.mean(cmp["legacy_strong"] | cmp["v2_strong"]))
                )
                same_dir = (
                    ((cmp["legacy_label"] == "retrieval_signal") & (cmp["tri_v2_label"] == "retrieval"))
                    | ((cmp["legacy_label"] == "computation_signal") & (cmp["tri_v2_label"] == "computation"))
                )
                row["strong_direction_agreement"] = float(same_dir.sum() / max(1, (cmp["legacy_strong"] & cmp["v2_strong"]).sum()))
        rows.append(row)

    sweep = pd.DataFrame(rows)
    sweep = sweep.sort_values(["pct_strong_total", "pct_mixed"], ascending=[False, True])
    return sweep


def write_summary(default_labels: pd.DataFrame, sweep: pd.DataFrame, p2a_note: str, out_path: Path) -> None:
    dist = default_labels.groupby(["family", "tri_v2_label"], as_index=False).size()
    best = sweep.iloc[0] if not sweep.empty else None
    lines = [
        "# Triangulation v2 summary",
        "",
        "k-of-n vote labels from P1 variants, P2 CCI/injection, P3 contamination/mechanistic.",
        "",
        "## Default thresholds (recommended starting point)",
        "",
        "- `min_votes=2`, `vote_margin=2`",
        "- contamination high ≥0.6 / low ≤0.4",
        "- CCI computation ≥0.5 / retrieval ≤0.3",
        "",
        "## Label distribution (default params)",
        "",
        "```",
        dist.sort_values(["family", "size"], ascending=[True, False]).to_string(index=False),
        "```",
        "",
        f"## P2A decision normalization\n\n{p2a_note}",
        "",
    ]
    if best is not None:
        lines.extend(
            [
                "## Threshold sweep (best strong-label rate)",
                "",
                f"- param_id={int(best['param_id'])}: **{100*best['pct_strong_total']:.1f}%** strong labels "
                f"(retrieval {100*best['pct_retrieval']:.1f}%, computation {100*best['pct_computation']:.1f}%)",
                f"- mixed {100*best['pct_mixed']:.1f}%, insufficient {100*best['pct_insufficient']:.1f}%",
            ]
        )
        if "legacy_strong_rate" in best and pd.notna(best["legacy_strong_rate"]):
            lines.append(
                f"- ALGO vs legacy: legacy strong {100*best['legacy_strong_rate']:.1f}%, "
                f"v2 strong {100*best['v2_strong_rate_algo']:.1f}%"
            )
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)
    p1 = load_p1_long()
    base = build_signal_frame(p1)
    if base.empty:
        raise SystemExit("No P1 signal frame — check raw behavioral CSVs.")

    default_th = TriThresholds()
    labels = apply_votes(base, default_th)
    legacy = load_legacy_algo_labels()
    labels = labels.merge(legacy, on=["problem_id", "model"], how="left")
    labels["model_short"] = labels["model"].map(SHORT).fillna(labels["model"])

    sweep = sweep_thresholds(base, legacy)

    keep = [
        "family",
        "problem_id",
        "model_short",
        "contam",
        "var_canonical",
        "var_w3",
        "vri_gap",
        "cci_effective",
        "match_first",
        "impl_post_correct",
        "retrieval_votes",
        "computation_votes",
        "votes_total",
        "tri_score",
        "tri_v2_confidence",
        "tri_v2_label",
        "legacy_label",
    ]
    vote_cols = [c for c in labels.columns if c.startswith("p1_") or c.startswith("p2_") or c.startswith("p3_")]
    out = labels[keep + vote_cols].rename(columns={"model_short": "model"})

    out.to_csv(DER / "triangulation_v2_labels.csv", index=False)
    sweep.to_csv(DER / "triangulation_threshold_sweep.csv", index=False)

    p2a_path = DER / "deep_p2a_phase_link.csv"
    if p2a_path.exists():
        p2a = pd.read_csv(p2a_path)
        raw = p2a["first_decision_match_rate_raw"].mean() if "first_decision_match_rate_raw" in p2a.columns else 0.0
        norm = p2a["first_decision_match_rate"].mean()
        p2a_note = (
            f"Raw prose↔token match **{100*raw:.1f}%**; normalized match **{100*norm:.1f}%** "
            f"(see `deep_p2a_decision_schema_audit.csv`)."
        )
    else:
        p2a_note = "Run `deep_metrics_analysis.py` for P2A schema audit."

    write_summary(out, sweep, p2a_note, DER / "triangulation_v2_summary.md")

    print("Wrote triangulation v2 pack:")
    for name in [
        "triangulation_v2_labels.csv",
        "triangulation_threshold_sweep.csv",
        "triangulation_v2_summary.md",
    ]:
        print(f" - results/derived/{name}")

    default_dist = out["tri_v2_label"].value_counts(normalize=True)
    print("\nDefault label rates:")
    print((100 * default_dist).round(1).astype(str) + "%")


if __name__ == "__main__":
    main()
