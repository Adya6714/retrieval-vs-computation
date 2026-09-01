#!/usr/bin/env python3
"""Deep cross-probe metric pack from existing raw runs.

This script intentionally computes metrics that are currently under-used by
the main paper pipeline, without requiring any new API runs.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probes.algo.decision_normalize import (
    decisions_match,
    normalize_phase1_decision,
    normalize_phase2_decision,
)
from scripts.runs.coverage_audit import filter_p1_to_bank, load_gsm_p2_merged
RAW = ROOT / "results" / "raw"
DER = ROOT / "results" / "derived"

VARIANTS = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]
MODEL_TAGS = {
    "anthropic/claude-sonnet-4": "claude",
    "google/gemini-2.5-flash": "gemini",
    "openai/gpt-4o": "gpt4o",
    "meta-llama/llama-3.1-8b-instruct": "llama",
    "openai/o4-mini": "o1mini",
}
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


def _norm_variant(v: object) -> str:
    s = str(v).strip()
    if not s:
        return s
    if s.lower() == "canonical":
        return "canonical"
    if s.lower().startswith("w") and s[1:].isdigit():
        return f"W{s[1:]}"
    return s


def _to_bool(x: object) -> bool | None:
    s = str(x).strip().lower()
    if s in {"true", "1", "yes"}:
        return True
    if s in {"false", "0", "no"}:
        return False
    return None


def _parse_failed_mask(df: pd.DataFrame) -> pd.Series:
    if "raw_response" in df.columns:
        return df["raw_response"].astype(str).str.startswith("ERROR:")
    return pd.Series([False] * len(df), index=df.index)


def load_p1_behavioral() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for fam in ["ALGO", "GSM", "BW"]:
        for model, tag in MODEL_TAGS.items():
            path = RAW / f"{fam}_P1_behavioral_{tag}.csv"
            df = _safe_read(path)
            if df is None or "variant_type" not in df.columns:
                continue
            df = df.copy()
            df["family"] = fam
            df["model"] = df.get("model", model).replace("", model)
            df["variant_type"] = df["variant_type"].map(_norm_variant)
            mask_fail = _parse_failed_mask(df)
            if "verified" in df.columns:
                corr = df["verified"].map(_to_bool)
            elif "behavioral_correct" in df.columns:
                corr = df["behavioral_correct"].map(_to_bool)
            else:
                corr = pd.Series([None] * len(df), index=df.index)
            df["correct"] = corr.where(~mask_fail, other=False)
            df = filter_p1_to_bank(df, fam)
            keep_cols = ["family", "problem_id", "model", "variant_type", "correct"]
            parts.append(df[keep_cols].drop_duplicates(keep="last"))

    # BW combined file (claude/gpt4o/llama)
    bw_combined = _safe_read(RAW / "BW_P1_behavioral.csv")
    if bw_combined is not None and "variant_type" in bw_combined.columns and "model" in bw_combined.columns:
        bw_combined = bw_combined.copy()
        bw_combined["family"] = "BW"
        bw_combined["variant_type"] = bw_combined["variant_type"].map(_norm_variant)
        corr = bw_combined.get("behavioral_correct", pd.Series([""] * len(bw_combined))).map(_to_bool)
        bw_combined["correct"] = corr.where(~_parse_failed_mask(bw_combined), other=False)
        bw_combined = filter_p1_to_bank(bw_combined, "BW")
        parts.append(
            bw_combined[["family", "problem_id", "model", "variant_type", "correct"]].drop_duplicates(keep="last")
        )

    if not parts:
        return pd.DataFrame(columns=["family", "problem_id", "model", "variant_type", "correct"])
    out = pd.concat(parts, ignore_index=True)
    out["model_short"] = out["model"].map(SHORT).fillna(out["model"])
    return out.drop_duplicates(["family", "problem_id", "model", "variant_type"], keep="last")


def p1_pairwise_metrics(p1: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (family, model), grp in p1.groupby(["family", "model"]):
        wide = grp.pivot_table(index="problem_id", columns="variant_type", values="correct", aggfunc="last")
        wide = wide.loc[:, ~wide.columns.duplicated()]
        for a in VARIANTS:
            for b in VARIANTS:
                if a not in wide.columns or b not in wide.columns:
                    continue
                sub = wide[[a, b]].dropna()
                if sub.empty:
                    continue
                a_bool = sub[a].astype(bool).to_numpy()
                b_bool = sub[b].astype(bool).to_numpy()
                n = len(sub)
                p_b = float(np.mean(b_bool))
                cond = int(np.sum(a_bool))
                p_b_given_a = float(np.sum(b_bool & a_bool) / cond) if cond > 0 else float("nan")
                b_sum = int(np.sum(b_bool))
                p_a_given_b = float(np.sum(a_bool & b_bool) / b_sum) if b_sum > 0 else float("nan")
                agree = float(np.mean(a_bool == b_bool))
                # simple binary association
                if n > 1 and np.std(a_bool.astype(int)) > 0 and np.std(b_bool.astype(int)) > 0:
                    phi = float(np.corrcoef(a_bool.astype(int), b_bool.astype(int))[0, 1])
                else:
                    phi = float("nan")
                rows.append(
                    {
                        "family": family,
                        "model": SHORT.get(model, model),
                        "variant_a": a,
                        "variant_b": b,
                        "n_overlap": n,
                        "p_b": p_b,
                        "p_b_given_a": p_b_given_a,
                        "p_a_given_b": p_a_given_b,
                        "lift_b_given_a": (p_b_given_a - p_b) if np.isfinite(p_b_given_a) else float("nan"),
                        "agreement_rate": agree,
                        "phi_corr": phi,
                    }
                )
    return pd.DataFrame(rows)


def p1_transition_metrics(p1: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (family, model), grp in p1.groupby(["family", "model"]):
        wide = grp.pivot_table(index="problem_id", columns="variant_type", values="correct", aggfunc="last")
        if "canonical" not in wide.columns:
            continue
        can = wide["canonical"]
        for v in [x for x in VARIANTS if x != "canonical" and x in wide.columns]:
            sub = wide[["canonical", v]].dropna().astype(bool)
            if sub.empty:
                continue
            cc = sub["canonical"]
            vv = sub[v]
            keep = int((cc & vv).sum())
            drop = int((cc & ~vv).sum())
            rescue = int((~cc & vv).sum())
            fail_both = int((~cc & ~vv).sum())
            rows.append(
                {
                    "family": family,
                    "model": SHORT.get(model, model),
                    "variant": v,
                    "n": len(sub),
                    "keep": keep,
                    "drop": drop,
                    "rescue": rescue,
                    "fail_both": fail_both,
                    "drop_rate_given_canonical_correct": (drop / int(cc.sum())) if cc.sum() > 0 else float("nan"),
                    "rescue_rate_given_canonical_wrong": (rescue / int((~cc).sum())) if (~cc).sum() > 0 else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def _iter_phase1_files() -> Iterable[Path]:
    from probes.common.results_paths import algo_p2_phase1_files

    for p in algo_p2_phase1_files():
        if p.exists() and p.stat().st_size > 0:
            yield p


def load_algo_phase1() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for p in _iter_phase1_files():
        df = _safe_read(p)
        if df is None or df.empty:
            continue
        if "model" not in df.columns:
            continue
        needed = {
            "problem_id",
            "model",
            "subtype",
            "instance_type",
            "predicted_first_decision",
            "greedy_assessment_correct",
            "critical_point_identified",
            "phase1_parseable",
        }
        for c in needed:
            if c not in df.columns:
                df[c] = ""
        parts.append(df[list(needed)].copy())
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(["problem_id", "model", "instance_type"], keep="last")
    return out


def load_algo_phase2_normal() -> pd.DataFrame:
    df = _safe_read(RAW / "ALGO_P2_phase2_normal.csv")
    if df is None or df.empty:
        return pd.DataFrame()
    for c in ["problem_id", "model", "instance_type", "step_index", "parsed_decision", "final_answer_correct", "reasoning_type"]:
        if c not in df.columns:
            df[c] = ""
    df["_step"] = pd.to_numeric(df["step_index"], errors="coerce").fillna(0)
    first = (
        df.sort_values("_step")
        .groupby(["problem_id", "model", "instance_type"], as_index=False)
        .first()[["problem_id", "model", "instance_type", "parsed_decision"]]
        .rename(columns={"parsed_decision": "phase2_first_decision"})
    )
    last = (
        df.sort_values("_step")
        .groupby(["problem_id", "model", "instance_type"], as_index=False)
        .last()[["problem_id", "model", "instance_type", "final_answer_correct", "reasoning_type"]]
    )
    merged = first.merge(last, on=["problem_id", "model", "instance_type"], how="outer")
    return merged


def p2a_phase_link_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    p1 = load_algo_phase1()
    p2 = load_algo_phase2_normal()
    if p1.empty or p2.empty:
        return pd.DataFrame(), pd.DataFrame()
    m = p1.merge(p2, on=["problem_id", "model", "instance_type"], how="left")
    m["phase1_parseable_bool"] = m["phase1_parseable"].map(_to_bool)
    m["greedy_assessment_correct_bool"] = m["greedy_assessment_correct"].map(_to_bool)
    m["critical_point_identified_bool"] = m["critical_point_identified"].map(_to_bool)
    m["final_answer_correct_bool"] = m["final_answer_correct"].map(_to_bool)

    m["phase1_norm"] = m.apply(
        lambda r: normalize_phase1_decision(r["subtype"], r["predicted_first_decision"]),
        axis=1,
    )
    m["phase2_norm"] = m.apply(
        lambda r: normalize_phase2_decision(r["subtype"], r["phase2_first_decision"]),
        axis=1,
    )
    m["first_decision_match_raw"] = (
        m["predicted_first_decision"].astype(str).str.strip().str.lower()
        == m["phase2_first_decision"].astype(str).str.strip().str.lower()
    )
    m["first_decision_match_norm"] = m.apply(
        lambda r: decisions_match(r["subtype"], r["predicted_first_decision"], r["phase2_first_decision"]),
        axis=1,
    )

    schema_audit = m[
        [
            "problem_id",
            "model",
            "subtype",
            "instance_type",
            "predicted_first_decision",
            "phase2_first_decision",
            "phase1_norm",
            "phase2_norm",
            "first_decision_match_raw",
            "first_decision_match_norm",
        ]
    ].copy()
    schema_audit["model"] = schema_audit["model"].map(SHORT).fillna(schema_audit["model"])

    rows: list[dict] = []
    for (model, subtype), g in m.groupby(["model", "subtype"]):
        rows.append(
            {
                "model": SHORT.get(model, model),
                "subtype": subtype,
                "n_sessions": len(g),
                "phase1_parseable_rate": float(g["phase1_parseable_bool"].mean()),
                "greedy_assessment_correct_rate": float(g["greedy_assessment_correct_bool"].mean()),
                "critical_point_identified_rate": float(g["critical_point_identified_bool"].mean()),
                "first_decision_match_rate_raw": float(g["first_decision_match_raw"].mean()),
                "first_decision_match_rate": float(g["first_decision_match_norm"].mean()),
                "final_correct_rate": float(g["final_answer_correct_bool"].mean()),
                "final_correct_when_match": float(
                    g.loc[g["first_decision_match_norm"], "final_answer_correct_bool"].mean()
                )
                if g["first_decision_match_norm"].any()
                else float("nan"),
                "final_correct_when_mismatch": float(
                    g.loc[~g["first_decision_match_norm"], "final_answer_correct_bool"].mean()
                )
                if (~g["first_decision_match_norm"]).any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows), schema_audit


def load_algo_injected(mode: str) -> pd.DataFrame:
    path = RAW / (
        "ALGO_P2_phase2_injected.csv" if mode == "plausible" else "ALGO_P2_phase2_injected_implausible.csv"
    )
    df = _safe_read(path)
    if df is None or df.empty:
        return pd.DataFrame()
    for c in [
        "problem_id",
        "model",
        "subtype",
        "instance_type",
        "step_index",
        "injection_applied",
        "critical_step_index",
        "response_type",
        "post_injection_correct",
    ]:
        if c not in df.columns:
            df[c] = ""
    df["mode"] = mode
    return df


def p2b_injection_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    p = load_algo_injected("plausible")
    i = load_algo_injected("implausible")
    all_df = pd.concat([x for x in [p, i] if not x.empty], ignore_index=True)
    if all_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    inj_rows = all_df[all_df["injection_applied"].map(_to_bool).fillna(False)].copy()
    final_rows = all_df[
        all_df["post_injection_correct"].astype(str).str.strip() != ""
    ].copy()
    final_rows["post_correct_bool"] = final_rows["post_injection_correct"].map(_to_bool)

    profile = (
        inj_rows.groupby(["mode", "model", "subtype", "response_type"], as_index=False)
        .size()
        .rename(columns={"size": "n"})
    )
    totals = profile.groupby(["mode", "model", "subtype"], as_index=False)["n"].sum().rename(columns={"n": "n_mode"})
    profile = profile.merge(totals, on=["mode", "model", "subtype"], how="left")
    profile["rate"] = profile["n"] / profile["n_mode"].replace(0, np.nan)
    profile["model"] = profile["model"].map(SHORT).fillna(profile["model"])

    agg = (
        final_rows.groupby(["mode", "model", "subtype"], as_index=False)["post_correct_bool"]
        .mean()
        .rename(columns={"post_correct_bool": "post_injection_correct_rate"})
    )
    agg["model"] = agg["model"].map(SHORT).fillna(agg["model"])
    wide = agg.pivot_table(index=["model", "subtype"], columns="mode", values="post_injection_correct_rate").reset_index()
    if "plausible" not in wide.columns:
        wide["plausible"] = np.nan
    if "implausible" not in wide.columns:
        wide["implausible"] = np.nan
    wide["plausible_minus_implausible"] = wide["plausible"] - wide["implausible"]

    return profile, wide


def mechanistic_links(p1: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for fam in ["ALGO", "GSM", "BW"]:
        mech = _safe_read(RAW / f"{fam}_P3_mechanistic.csv")
        cont = _safe_read(RAW / f"{fam}_P3_contamination.csv")
        if mech is None or mech.empty:
            continue
        m = mech.copy()
        for c in ["problem_id", "model", "crystallization_layer", "n_layers_processed", "layer_cosine_similarities"]:
            if c not in m.columns:
                m[c] = ""
        m["crystallization_layer_num"] = pd.to_numeric(m["crystallization_layer"], errors="coerce")
        m["n_layers_num"] = pd.to_numeric(m["n_layers_processed"], errors="coerce")
        m["crystallization_depth_norm"] = m["crystallization_layer_num"] / m["n_layers_num"].replace(0, np.nan)

        def _tail_mean(text: str, k: int) -> float:
            try:
                arr = ast.literal_eval(str(text))
                if isinstance(arr, list) and arr:
                    arrf = [float(x) for x in arr]
                    return float(np.mean(arrf[-k:]))
            except Exception:
                pass
            return float("nan")

        m["cosine_tail5_mean"] = m["layer_cosine_similarities"].map(lambda x: _tail_mean(str(x), 5))

        if cont is not None and not cont.empty and "problem_id" in cont.columns:
            score_col = "instance_contamination_score" if "instance_contamination_score" in cont.columns else "contamination_score"
            c = cont[["problem_id", score_col]].copy()
            c[score_col] = pd.to_numeric(c[score_col], errors="coerce")
            c = c.groupby("problem_id", as_index=False)[score_col].mean()
            m = m.merge(c, on="problem_id", how="left")
        else:
            score_col = "instance_contamination_score"
            m[score_col] = np.nan

        p1f = p1[p1["family"] == fam]
        if not p1f.empty:
            wide = p1f.pivot_table(index=["problem_id", "model"], columns="variant_type", values="correct", aggfunc="last").reset_index()
            if "canonical" in wide.columns and "W3" in wide.columns:
                wide["w3_drop"] = ((wide["canonical"] == True) & (wide["W3"] == False)).astype(float)
                m = m.merge(wide[["problem_id", "model", "w3_drop"]], on=["problem_id", "model"], how="left")
            else:
                m["w3_drop"] = np.nan
        else:
            m["w3_drop"] = np.nan

        for model, g in m.groupby("model"):
            g = g.dropna(subset=["crystallization_depth_norm"])
            if len(g) < 3:
                continue
            if (
                g[score_col].notna().sum() >= 3
                and np.nanstd(g["crystallization_depth_norm"].to_numpy(dtype=float)) > 0
                and np.nanstd(g[score_col].to_numpy(dtype=float)) > 0
            ):
                r_contam = float(np.corrcoef(g["crystallization_depth_norm"], g[score_col])[0, 1])
            else:
                r_contam = float("nan")
            if (
                g["w3_drop"].notna().sum() >= 3
                and np.nanstd(g["crystallization_depth_norm"].to_numpy(dtype=float)) > 0
                and np.nanstd(g["w3_drop"].to_numpy(dtype=float)) > 0
            ):
                r_w3 = float(np.corrcoef(g["crystallization_depth_norm"], g["w3_drop"])[0, 1])
            else:
                r_w3 = float("nan")
            rows.append(
                {
                    "family": fam,
                    "model": SHORT.get(model, model),
                    "n": len(g),
                    "mean_crystallization_depth_norm": float(g["crystallization_depth_norm"].mean()),
                    "mean_cosine_tail5": float(g["cosine_tail5_mean"].mean()),
                    "corr_depth_vs_contamination": r_contam,
                    "corr_depth_vs_w3_drop": r_w3,
                }
            )
    return pd.DataFrame(rows)


def triangulation_plus(p1: pd.DataFrame) -> pd.DataFrame:
    # P1 base
    base = p1.pivot_table(index=["family", "problem_id", "model"], columns="variant_type", values="correct", aggfunc="last").reset_index()
    if "canonical" not in base.columns:
        return pd.DataFrame()

    # P2 per-instance for ALGO
    cci_algo = _safe_read(DER / "ALGO_P2_per_instance_cci.csv")
    if cci_algo is not None and not cci_algo.empty:
        cci_algo = cci_algo.copy()
        cci_algo["family"] = "ALGO"
        cci_algo["cci_composite_num"] = pd.to_numeric(cci_algo["cci_composite"], errors="coerce")
        cci_algo = cci_algo[["family", "problem_id", "model", "cci_composite_num"]].drop_duplicates()
    else:
        cci_algo = pd.DataFrame(columns=["family", "problem_id", "model", "cci_composite_num"])

    # P2 GSM CCI per-instance
    gsm_cci = load_gsm_p2_merged()
    if not gsm_cci.empty:
        gsm_cci = gsm_cci.copy()
        gsm_cci["family"] = "GSM"
        gsm_cci["cci_composite_num"] = pd.to_numeric(gsm_cci["cci_score"], errors="coerce")
        gsm_cci = gsm_cci[["family", "problem_id", "model", "cci_composite_num"]].drop_duplicates()
    else:
        gsm_cci = pd.DataFrame(columns=["family", "problem_id", "model", "cci_composite_num"])

    # P2 injection evidence (ALGO)
    inj_impl = load_algo_injected("implausible")
    if not inj_impl.empty:
        final_impl = inj_impl[inj_impl["post_injection_correct"].astype(str).str.strip() != ""].copy()
        final_impl["impl_post_correct"] = final_impl["post_injection_correct"].map(_to_bool)
        final_impl["family"] = "ALGO"
        final_impl = final_impl[["family", "problem_id", "model", "impl_post_correct"]].drop_duplicates()
    else:
        final_impl = pd.DataFrame(columns=["family", "problem_id", "model", "impl_post_correct"])

    # P3 contamination
    cont_parts = []
    for fam in ["ALGO", "GSM", "BW"]:
        c = _safe_read(RAW / f"{fam}_P3_contamination.csv")
        if c is None or c.empty or "problem_id" not in c.columns:
            continue
        score_col = "instance_contamination_score" if "instance_contamination_score" in c.columns else "contamination_score"
        c = c[["problem_id", score_col]].copy()
        c[score_col] = pd.to_numeric(c[score_col], errors="coerce")
        c = c.groupby("problem_id", as_index=False)[score_col].mean()
        c["family"] = fam
        c = c.rename(columns={score_col: "contam"})
        cont_parts.append(c)
    cont = pd.concat(cont_parts, ignore_index=True) if cont_parts else pd.DataFrame(columns=["family", "problem_id", "contam"])

    t = base.merge(cont, on=["family", "problem_id"], how="left")
    t = t.merge(pd.concat([cci_algo, gsm_cci], ignore_index=True), on=["family", "problem_id", "model"], how="left")
    t = t.merge(final_impl, on=["family", "problem_id", "model"], how="left")

    # Signal votes
    t["canonical_bool"] = t["canonical"].map(_to_bool)
    t["w3_bool"] = t["W3"].map(_to_bool) if "W3" in t.columns else None
    t["w1_bool"] = t["W1"].map(_to_bool) if "W1" in t.columns else None
    t["w2_bool"] = t["W2"].map(_to_bool) if "W2" in t.columns else None
    t["w4_bool"] = t["W4"].map(_to_bool) if "W4" in t.columns else None

    # P1 rename-specific fragility vote
    t["vote_retrieval_p1"] = (
        (t["canonical_bool"] == True) & (t["w3_bool"] == False)  # noqa: E712
    ).astype(int)
    t["vote_computation_p1"] = (
        (t["canonical_bool"] == True)
        & (t["w3_bool"] == True)
        & ((t["w1_bool"] == True) | (t["w2_bool"] == True) | (t["w4_bool"] == True))
    ).astype(int)

    # P2 CCI vote
    t["vote_computation_p2_cci"] = (pd.to_numeric(t["cci_composite_num"], errors="coerce") >= 0.5).fillna(False).astype(int)
    t["vote_retrieval_p2_cci"] = (
        (pd.to_numeric(t["cci_composite_num"], errors="coerce") < 0.3).fillna(False)
    ).astype(int)

    # P2 implausible injection vote (ALGO only where present)
    t["vote_computation_p2_inj"] = (t["impl_post_correct"] == True).fillna(False).astype(int)  # noqa: E712
    t["vote_retrieval_p2_inj"] = (t["impl_post_correct"] == False).fillna(False).astype(int)  # noqa: E712

    # P3 contamination vote
    t["vote_retrieval_p3"] = (pd.to_numeric(t["contam"], errors="coerce") >= 0.6).fillna(False).astype(int)
    t["vote_computation_p3"] = (pd.to_numeric(t["contam"], errors="coerce") <= 0.4).fillna(False).astype(int)

    retrieval_cols = [c for c in t.columns if c.startswith("vote_retrieval_")]
    computation_cols = [c for c in t.columns if c.startswith("vote_computation_")]
    t["retrieval_votes"] = t[retrieval_cols].sum(axis=1)
    t["computation_votes"] = t[computation_cols].sum(axis=1)
    t["votes_total"] = t["retrieval_votes"] + t["computation_votes"]
    t["tri_plus_score"] = t["computation_votes"] - t["retrieval_votes"]
    t["tri_plus_confidence"] = (t["tri_plus_score"].abs() / t["votes_total"].replace(0, np.nan)).fillna(0.0)

    t["tri_plus_label"] = "mixed"
    t.loc[t["votes_total"] < 2, "tri_plus_label"] = "insufficient"
    t.loc[(t["votes_total"] >= 2) & (t["tri_plus_score"] >= 2), "tri_plus_label"] = "computation_signal_plus"
    t.loc[(t["votes_total"] >= 2) & (t["tri_plus_score"] <= -2), "tri_plus_label"] = "retrieval_signal_plus"
    t.loc[(t["votes_total"] >= 2) & (t["tri_plus_score"] == 1), "tri_plus_label"] = "weak_computation"
    t.loc[(t["votes_total"] >= 2) & (t["tri_plus_score"] == -1), "tri_plus_label"] = "weak_retrieval"

    t["model_short"] = t["model"].map(SHORT).fillna(t["model"])
    keep_cols = [
        "family",
        "problem_id",
        "model_short",
        "contam",
        "cci_composite_num",
        "impl_post_correct",
        "retrieval_votes",
        "computation_votes",
        "votes_total",
        "tri_plus_score",
        "tri_plus_confidence",
        "tri_plus_label",
    ]
    return t[keep_cols].rename(columns={"model_short": "model"})


def summarize_to_markdown(
    out_path: Path,
    pairwise: pd.DataFrame,
    transitions: pd.DataFrame,
    p2a: pd.DataFrame,
    p2b_delta: pd.DataFrame,
    mech: pd.DataFrame,
    tri_plus: pd.DataFrame,
) -> None:
    def _table_text(df: pd.DataFrame) -> str:
        if df.empty:
            return "(no rows)"
        return df.to_csv(index=False).strip()

    lines: list[str] = []
    lines.append("# Deep metric analysis (new)\n")
    lines.append("Computed from existing raw runs; no new API executions.\n")

    if not pairwise.empty:
        lines.append("## P1 pairwise interrelations\n")
        ex = (
            pairwise[
                (pairwise["variant_a"] == "canonical")
                & (pairwise["variant_b"].isin(["W1", "W2", "W3", "W4", "W5", "W6"]))
            ]
            .sort_values(["family", "model", "variant_b"])
            .head(24)
        )
        lines.append(_table_text(ex))
        lines.append("")

    if not transitions.empty:
        lines.append("## P1 transition diagnostics (canonical -> variant)\n")
        lines.append(_table_text(transitions.sort_values(["family", "model", "variant"]).head(30)))
        lines.append("")

    if not p2a.empty:
        lines.append("## P2A phase-link metrics\n")
        lines.append(
            "_Raw string match was 0% (prose vs token); `first_decision_match_rate` uses normalized tokens._\n"
        )
        lines.append(_table_text(p2a.sort_values(["model", "subtype"])))
        lines.append("")

    if not p2b_delta.empty:
        lines.append("## P2B injection reactivity (plausible vs implausible)\n")
        lines.append(_table_text(p2b_delta.sort_values(["model", "subtype"])))
        lines.append("")

    if not mech.empty:
        lines.append("## P3 mechanistic link metrics\n")
        lines.append(_table_text(mech.sort_values(["family", "model"])))
        lines.append("")

    if not tri_plus.empty:
        dist = tri_plus.groupby(["family", "tri_plus_label"], as_index=False).size().rename(columns={"size": "n"})
        lines.append("## Triangulation-plus label distribution\n")
        lines.append(_table_text(dist.sort_values(["family", "n"], ascending=[True, False])))
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    DER.mkdir(parents=True, exist_ok=True)

    p1 = load_p1_behavioral()
    pairwise = p1_pairwise_metrics(p1)
    transitions = p1_transition_metrics(p1)
    p2a, p2a_schema = p2a_phase_link_metrics()
    p2b_profile, p2b_delta = p2b_injection_metrics()
    mech = mechanistic_links(p1)
    tri_plus = triangulation_plus(p1)

    pairwise.to_csv(DER / "deep_p1_pairwise.csv", index=False)
    transitions.to_csv(DER / "deep_p1_transitions.csv", index=False)
    p2a.to_csv(DER / "deep_p2a_phase_link.csv", index=False)
    p2a_schema.to_csv(DER / "deep_p2a_decision_schema_audit.csv", index=False)
    p2b_profile.to_csv(DER / "deep_p2b_response_profile.csv", index=False)
    p2b_delta.to_csv(DER / "deep_p2b_reactivity_delta.csv", index=False)
    mech.to_csv(DER / "deep_probe3_mech_links.csv", index=False)
    tri_plus.to_csv(DER / "deep_triangulation_plus.csv", index=False)

    summarize_to_markdown(
        DER / "deep_metrics_summary.md",
        pairwise,
        transitions,
        p2a,
        p2b_delta,
        mech,
        tri_plus,
    )

    print("Wrote deep metric pack:")
    for name in [
        "deep_p1_pairwise.csv",
        "deep_p1_transitions.csv",
        "deep_p2a_phase_link.csv",
        "deep_p2a_decision_schema_audit.csv",
        "deep_p2b_response_profile.csv",
        "deep_p2b_reactivity_delta.csv",
        "deep_probe3_mech_links.csv",
        "deep_triangulation_plus.csv",
        "deep_metrics_summary.md",
    ]:
        print(f" - results/derived/{name}")


if __name__ == "__main__":
    main()
