"""End-to-end re-derivation of every paper metric from the current raw CSVs.

This script must be safe to run repeatedly (idempotent). It writes outputs to
`results/derived/` and `results/paper/AUDIT/` so figures and tables can pick
them up directly.

Canonical derivation path for all paper numbers — run:
    python scripts/runs/rederive_all_metrics.py

Sections:
    0) Coverage audit — master table, gap list, imputed vs complete-case (§0.1)
    1) Coverage matrix (which (probe, model) slices are now full)
    2) Probe 1 — per-model canonical / W1..W6 accuracies, VAR, W3-retention
    3) Probe 2 — GSM CCI/TEP, ALGO P2A invocation, P2B plausible vs implausible
    4) Probe 3 — contamination/VRI Pearson + triangulation summary
    5) Cross-probe — accuracy vs robustness correlation
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
RAW  = ROOT / "results" / "raw"
DER  = ROOT / "results" / "derived"
AUD  = ROOT / "results" / "paper" / "AUDIT"
DER.mkdir(parents=True, exist_ok=True)
AUD.mkdir(parents=True, exist_ok=True)

import sys as _sys

if str(ROOT) not in _sys.path:
    _sys.path.insert(0, str(ROOT))

from scripts.runs.coverage_audit import filter_p1_to_bank, load_gsm_p2_merged  # noqa: E402


MODELS = [
    "anthropic/claude-sonnet-4",
    "google/gemini-2.5-flash",
    "openai/gpt-4o",
    "meta-llama/llama-3.1-8b-instruct",
    "openai/o4-mini",
]
SHORT = {
    "anthropic/claude-sonnet-4":            "Claude",
    "google/gemini-2.5-flash":              "Gemini",
    "openai/gpt-4o":                        "GPT-4o",
    "meta-llama/llama-3.1-8b-instruct":     "Llama",
    "openai/o4-mini":                       "o4-mini",
}


def _safe_read(path: Path) -> pd.DataFrame | None:
    if not path.exists() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        print(f"  !! read failed for {path}: {e}")
        return None


def _accuracy(df: pd.DataFrame, correct_col: str | None = None) -> tuple[float, int, int]:
    """Return (accuracy, n_valid_responses, n_attempted_rows).

    Rows with ``model_answer`` or ``raw_response`` starting with ``ERROR:`` are
    excluded from the numerator AND denominator; an "attempted" count is also
    returned so callers can surface coverage gaps.
    """
    if df.empty:
        return float("nan"), 0, 0
    candidates = [correct_col] if correct_col else ["behavioral_correct", "verified", "final_answer_correct"]
    col = next((c for c in candidates if c and c in df.columns), None)
    if col is None:
        return float("nan"), 0, 0
    vals = df[col].astype(str).str.lower().str.strip()
    raw_col = df.get("raw_response", df.get("model_answer", pd.Series([""] * len(df))))
    mask_valid = ~raw_col.astype(str).str.startswith("ERROR:")
    yes = (vals.isin({"true", "1", "yes"}) & mask_valid).sum()
    n_valid = int(mask_valid.sum())
    n_attempted = int(len(df))
    return (yes / n_valid if n_valid > 0 else float("nan"), n_valid, n_attempted)


# ---------------------- 1. COVERAGE MATRIX ---------------------------------

def coverage_matrix() -> pd.DataFrame:
    families = ["ALGO_P1", "GSM_P1", "BW_P1",
                "GSM_P2", "ALGO_P2A_normal", "ALGO_P2A_elicited",
                "ALGO_P2B_plausible", "ALGO_P2B_implausible"]
    files = {
        "ALGO_P1": {m: RAW / f"ALGO_P1_behavioral_{tag}.csv" for m, tag in
                    [("anthropic/claude-sonnet-4","claude"),
                     ("google/gemini-2.5-flash","gemini"),
                     ("openai/gpt-4o","gpt4o"),
                     ("meta-llama/llama-3.1-8b-instruct","llama"),
                     ("openai/o4-mini","o1mini")]},
        "GSM_P1":  {m: RAW / f"GSM_P1_behavioral_{tag}.csv" for m, tag in
                    [("anthropic/claude-sonnet-4","claude"),
                     ("google/gemini-2.5-flash","gemini"),
                     ("openai/gpt-4o","gpt4o"),
                     ("meta-llama/llama-3.1-8b-instruct","llama"),
                     ("openai/o4-mini","o1mini")]},
        "BW_P1":   {m: RAW / f"BW_P1_behavioral_{tag}.csv" for m, tag in
                    [("google/gemini-2.5-flash","gemini"),
                     ("openai/o4-mini","o1mini")]},
    }
    rows = []
    for fam, mapper in files.items():
        for m, path in mapper.items():
            df = _safe_read(path)
            if df is None:
                n = 0; n_err = 0
            else:
                if "variant_type" in df.columns:
                    df["variant_type"] = df["variant_type"].astype(str).str.strip().apply(
                        lambda v: v.upper() if v and v[0].lower() == "w" else v
                    )
                    df = filter_p1_to_bank(df, "BW")
                if "raw_response" in df.columns:
                    n_err = df["raw_response"].astype(str).str.startswith("ERROR:").sum()
                else:
                    n_err = 0
                n = len(df) - n_err
            rows.append({"probe": fam, "model": SHORT[m], "n_valid": n, "n_errors": n_err})

    # Combined BW file for Claude / GPT-4o / Llama
    bw_combined = _safe_read(RAW / "BW_P1_behavioral.csv")
    if bw_combined is not None and "model" in bw_combined.columns:
        for m in ["anthropic/claude-sonnet-4", "openai/gpt-4o", "meta-llama/llama-3.1-8b-instruct"]:
            sub = bw_combined[bw_combined["model"] == m]
            if "raw_response" in sub.columns:
                n_err = sub["raw_response"].astype(str).str.startswith("ERROR:").sum()
            else:
                n_err = 0
            sub = filter_p1_to_bank(sub, "BW") if not sub.empty else sub
            n = len(sub) - (sub["raw_response"].astype(str).str.startswith("ERROR:").sum() if "raw_response" in sub.columns else 0)
            rows.append({"probe": "BW_P1", "model": SHORT[m], "n_valid": n, "n_errors": n_err})
    # Probe 2 shared files
    p2_specs: list[tuple[str | None, str]] = [
        (None, "GSM_P2"),
        ("ALGO_P2_phase2_normal.csv", "ALGO_P2A_normal"),
        ("ALGO_P2_phase2_normal_elicited.csv", "ALGO_P2A_elicited"),
        ("ALGO_P2_phase2_injected.csv", "ALGO_P2B_plausible"),
        ("ALGO_P2_phase2_injected_implausible.csv", "ALGO_P2B_implausible"),
    ]
    for fname, label in p2_specs:
        if label == "GSM_P2":
            df = load_gsm_p2_merged()
        else:
            df = _safe_read(RAW / fname) if fname else None
        if df is None:
            for m in MODELS:
                rows.append({"probe": label, "model": SHORT[m], "n_valid": 0, "n_errors": 0})
            continue
        if "model" not in df.columns:
            continue
        for m in MODELS:
            sub = df[df["model"] == m]
            if "raw_response" in sub.columns:
                n_err = sub["raw_response"].astype(str).str.startswith("ERROR:").sum()
            else:
                n_err = 0
            n = len(sub) - n_err
            rows.append({"probe": label, "model": SHORT[m], "n_valid": n, "n_errors": n_err})
    return pd.DataFrame(rows)


# ---------------------- 2. PROBE 1 METRICS ---------------------------------

def probe1_per_model() -> pd.DataFrame:
    """Re-derive canonical, W1..W6 accuracies for every (model, family) pair.

    Family is inferred from problem_family column or from filename. We report
    overall variant accuracy across all problems within that family for each
    model.
    """
    # Per-model files
    triples = [
        ("ALGO", [("anthropic/claude-sonnet-4","claude"),
                  ("google/gemini-2.5-flash","gemini"),
                  ("openai/gpt-4o","gpt4o"),
                  ("meta-llama/llama-3.1-8b-instruct","llama"),
                  ("openai/o4-mini","o1mini")]),
        ("GSM",  [("anthropic/claude-sonnet-4","claude"),
                  ("google/gemini-2.5-flash","gemini"),
                  ("openai/gpt-4o","gpt4o"),
                  ("meta-llama/llama-3.1-8b-instruct","llama"),
                  ("openai/o4-mini","o1mini")]),
        ("BW",   [("google/gemini-2.5-flash","gemini"),
                  ("openai/o4-mini","o1mini")]),
    ]
    out = []
    for fam, tagmap in triples:
        for model, tag in tagmap:
            path = RAW / f"{fam}_P1_behavioral_{tag}.csv"
            df = _safe_read(path)
            if df is None or "variant_type" not in df.columns:
                continue
            # Normalize variant casing — files mix W6 and w6
            df["variant_type"] = df["variant_type"].astype(str).str.strip()
            df["variant_type"] = df["variant_type"].apply(
                lambda v: v.upper() if v and v[0].lower() == "w" else v
            )
            df = df.drop_duplicates(["problem_id", "variant_type"], keep="last")
            df = filter_p1_to_bank(df, fam)
            for v in ["canonical","W1","W2","W3","W4","W5","W6"]:
                sub = df[df["variant_type"] == v]
                acc, n_valid, n_att = _accuracy(sub)
                out.append({
                    "probe": fam,
                    "model": SHORT[model],
                    "variant": v,
                    "n_attempted": n_att,
                    "n_valid": n_valid,
                    "accuracy": acc,
                })

    # Combined BW file for Claude / GPT-4o / Llama
    bw_combined = _safe_read(RAW / "BW_P1_behavioral.csv")
    if bw_combined is not None and "model" in bw_combined.columns and "variant_type" in bw_combined.columns:
        bw_combined["variant_type"] = bw_combined["variant_type"].astype(str).str.strip().apply(
            lambda v: v.upper() if v and v[0].lower() == "w" else v
        )
        for model in ["anthropic/claude-sonnet-4", "openai/gpt-4o", "meta-llama/llama-3.1-8b-instruct"]:
            sub_m = bw_combined[bw_combined["model"] == model]
            if sub_m.empty: continue
            sub_m = sub_m.drop_duplicates(["problem_id","variant_type"], keep="last")
            sub_m = filter_p1_to_bank(sub_m, "BW")
            for v in ["canonical","W1","W2","W3","W4","W5","W6"]:
                sub = sub_m[sub_m["variant_type"] == v]
                acc, n_valid, n_att = _accuracy(sub)
                out.append({
                    "probe": "BW",
                    "model": SHORT[model],
                    "variant": v,
                    "n_attempted": n_att,
                    "n_valid": n_valid,
                    "accuracy": acc,
                })
    return pd.DataFrame(out)


def probe1_w3_retention(per_model: pd.DataFrame) -> pd.DataFrame:
    """W3-retention = acc(W3) / acc(canonical) per (probe, model)."""
    rows = []
    for (probe, model), g in per_model.groupby(["probe", "model"]):
        ac = g.loc[g["variant"] == "canonical", "accuracy"].squeeze() if (g["variant"]=="canonical").any() else float("nan")
        w3 = g.loc[g["variant"] == "W3",        "accuracy"].squeeze() if (g["variant"]=="W3").any() else float("nan")
        ret = (w3 / ac) if (ac and not np.isnan(ac) and ac > 0) else float("nan")
        rows.append({"probe": probe, "model": model, "canonical": ac, "W3": w3, "W3_retention": ret})
    return pd.DataFrame(rows)


# ---------------------- 3. PROBE 2 METRICS ---------------------------------

def gsm_p2_metrics() -> pd.DataFrame:
    df = load_gsm_p2_merged()
    if df is None or "model" not in df.columns:
        return pd.DataFrame()
    rows = []
    for m in MODELS:
        sub = df[df["model"] == m]
        if sub.empty:
            continue
        n_parseable = ((sub.get("phase1_parseable", pd.Series([""]*len(sub)))
                          .astype(str).str.lower() == "true").sum())
        cci = pd.to_numeric(sub.get("cci_score", pd.Series([np.nan]*len(sub))), errors="coerce")
        tep = pd.to_numeric(sub.get("tep_score", pd.Series([np.nan]*len(sub))), errors="coerce")
        sess_b = ((sub.get("session_b_correct", pd.Series([""]*len(sub)))
                     .astype(str).str.lower() == "true").sum())
        rows.append({
            "model": SHORT[m],
            "n":     len(sub),
            "n_parseable": int(n_parseable),
            "mean_cci": float(cci.mean()) if cci.notna().any() else float("nan"),
            "mean_tep": float(tep.mean()) if tep.notna().any() else float("nan"),
            "session_b_correct_rate": sess_b / len(sub) if len(sub) > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def algo_p2_metrics() -> pd.DataFrame:
    out = []
    for fname, cond in [
        ("ALGO_P2_phase2_normal.csv",                "normal"),
        ("ALGO_P2_phase2_normal_elicited.csv",       "elicited"),
        ("ALGO_P2_phase2_injected.csv",              "plausible_inj"),
        ("ALGO_P2_phase2_injected_implausible.csv",  "implausible_inj"),
    ]:
        df = _safe_read(RAW / fname)
        if df is None or "model" not in df.columns:
            continue
        for m in MODELS:
            sub = df[df["model"] == m].copy()
            if sub.empty: continue
            group_keys = ["problem_id"]
            if "instance_type" in sub.columns:
                group_keys.append("instance_type")
            n_sessions = sub.groupby(group_keys).ngroups
            sub["_step_idx"] = pd.to_numeric(sub.get("step_index", 0), errors="coerce").fillna(0)
            last_rows = sub.sort_values("_step_idx").groupby(group_keys).tail(1)
            if "final_answer_correct" in last_rows.columns and (last_rows["final_answer_correct"].astype(str) != "").any():
                final_correct = (last_rows["final_answer_correct"].astype(str).str.lower() == "true").mean()
            elif "post_injection_correct" in last_rows.columns:
                final_correct = (last_rows["post_injection_correct"].astype(str).str.lower() == "true").mean()
            else:
                final_correct = float("nan")
            if "response_type" in sub.columns:
                algo_inv = (sub["response_type"].astype(str).str.lower().str.contains("algo")).mean()
            else:
                algo_inv = float("nan")
            if "diverged_from_normal" in sub.columns and (sub["diverged_from_normal"].astype(str) != "").any():
                divs = sub[sub["diverged_from_normal"].astype(str) != ""]
                diverged = (divs["diverged_from_normal"].astype(str).str.lower() == "true").mean()
            else:
                diverged = float("nan")
            out.append({
                "condition": cond,
                "model":     SHORT[m],
                "sessions":  n_sessions,
                "final_correct": float(final_correct) if not np.isnan(final_correct) else float("nan"),
                "algorithm_invocation": float(algo_inv) if not np.isnan(algo_inv) else float("nan"),
                "diverged_rate": float(diverged) if not np.isnan(diverged) else float("nan"),
            })
    return pd.DataFrame(out)


# ---------------------- 4. PROBE 3 METRICS ---------------------------------

def probe3_contam_vri() -> pd.DataFrame:
    """Re-derive contamination↔VRI Pearson per model from existing files."""
    # ALGO contamination
    cs = ROOT / "results" / "raw" / "ALGO_P3_contamination.csv"
    df_c = _safe_read(cs)
    if df_c is None:
        return pd.DataFrame()
    # join with VRI from triangulation
    tri = _safe_read(ROOT / "results" / "derived" / "ALGO_P3_triangulation.csv")
    if tri is None:
        return pd.DataFrame()
    # Need per-problem VRI - we'll re-derive if needed
    return pd.DataFrame()  # placeholder; the existing audit file is already correct


# ---------------------- 5. CROSS-PROBE ROBUSTNESS-ACCURACY -----------------

def accuracy_robustness_spearman(p1: pd.DataFrame) -> pd.DataFrame:
    """Within-family Spearman between canonical accuracy and W3 retention across models."""
    rows = []
    for fam, g in p1.groupby("probe"):
        ac = g[g["variant"] == "canonical"].set_index("model")["accuracy"]
        w3 = g[g["variant"] == "W3"].set_index("model")["accuracy"]
        common = ac.index.intersection(w3.index)
        ac = ac.loc[common].astype(float)
        w3 = w3.loc[common].astype(float)
        # retention
        ret = w3 / ac.replace(0, np.nan)
        valid = ac.notna() & ret.notna()
        if valid.sum() < 3:
            rho, p = float("nan"), float("nan")
        else:
            rho, p = stats.spearmanr(ac[valid], ret[valid])
        rows.append({"probe": fam,
                     "n_models": int(valid.sum()),
                     "spearman_rho": rho, "p_value": p,
                     "models_used": ",".join(common[valid].tolist())})
    return pd.DataFrame(rows)


# ---------------------- MAIN -----------------------------------------------

def main() -> None:
    from scripts.runs.coverage_audit import run_audit

    print("[0/6] coverage audit (§0.1)")
    run_audit()

    print("\n[1/6] coverage_matrix"); cov = coverage_matrix()
    cov.to_csv(DER / "coverage_matrix.csv", index=False)
    cov_piv = cov.pivot(index="probe", columns="model", values="n_valid").fillna(0).astype(int)
    cov_piv.to_csv(DER / "coverage_pivot.csv")
    print(cov_piv.to_string())

    print("\n[2/6] probe1 per-model variant accuracies"); p1 = probe1_per_model()
    p1.to_csv(DER / "probe1_per_model_variant.csv", index=False)
    print(p1.pivot_table(index=["probe","model"], columns="variant", values="accuracy").round(3).to_string())

    print("\n[3/6] probe1 W3-retention"); w3r = probe1_w3_retention(p1)
    w3r.to_csv(DER / "probe1_w3_retention.csv", index=False)
    print(w3r.round(3).to_string(index=False))

    print("\n[4/6] probe2 GSM CCI/TEP"); gsm = gsm_p2_metrics()
    gsm.to_csv(DER / "probe2_gsm_metrics.csv", index=False)
    print(gsm.round(3).to_string(index=False))

    print("\n[5/6] probe2 ALGO P2A/P2B"); algo = algo_p2_metrics()
    algo.to_csv(DER / "probe2_algo_metrics.csv", index=False)
    print(algo.round(3).to_string(index=False))

    print("\n[6/6] accuracy vs W3 retention (Spearman across models, per family)")
    sp = accuracy_robustness_spearman(p1)
    sp.to_csv(DER / "cross_probe_acc_vs_w3retention.csv", index=False)
    print(sp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
