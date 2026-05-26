"""Cross-probe pattern analysis.

For each (model, problem_id), join behavioral (P1), self-monitor (P2),
contamination/triangulation (P3) and mechanistic (P3) features. Then
look for hidden correlations between probes.

Outputs (all written to results/derived/):
  - master_per_problem_5model.csv    # rich feature table
  - cross_probe_corr_within_model.csv
  - cross_probe_corr_pooled.csv
  - implausibility_detection.csv
  - algorithm_invocation_clean.csv
  - bw_violation_profile.csv
  - cross_family_universally_fragile.csv

Designed to surface findings I can write into the paper directly.
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
DER.mkdir(parents=True, exist_ok=True)

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


def _norm_variant(v: str) -> str:
    v = str(v).strip()
    return v.upper() if v and v[0].lower() == "w" else v


def _load_p1(family: str) -> pd.DataFrame:
    """Return a per-(problem_id, model, variant) frame for a family."""
    tag = {"ALGO":[("anthropic/claude-sonnet-4","claude"),
                   ("google/gemini-2.5-flash","gemini"),
                   ("openai/gpt-4o","gpt4o"),
                   ("meta-llama/llama-3.1-8b-instruct","llama"),
                   ("openai/o4-mini","o1mini")],
           "GSM": [("anthropic/claude-sonnet-4","claude"),
                   ("google/gemini-2.5-flash","gemini"),
                   ("openai/gpt-4o","gpt4o"),
                   ("meta-llama/llama-3.1-8b-instruct","llama"),
                   ("openai/o4-mini","o1mini")],
           "BW":  [("google/gemini-2.5-flash","gemini"),
                   ("openai/o4-mini","o1mini")]}[family]
    parts = []
    for model, t in tag:
        p = RAW / f"{family}_P1_behavioral_{t}.csv"
        if not p.exists(): continue
        df = pd.read_csv(p, dtype=str).fillna("")
        if "variant_type" not in df.columns: continue
        df["variant_type"] = df["variant_type"].apply(_norm_variant)
        # accuracy column
        if "behavioral_correct" in df.columns:
            df["_correct"] = df["behavioral_correct"].str.lower().isin(["true","1","yes"])
        elif "verified" in df.columns:
            df["_correct"] = df["verified"].str.lower().isin(["true","1","yes"])
        else:
            df["_correct"] = False
        df["model"] = model
        df = df[["problem_id","model","variant_type","_correct"]]
        df = df.drop_duplicates(["problem_id","model","variant_type"], keep="last")
        parts.append(df)

    # BW combined file
    if family == "BW":
        p = RAW / "BW_P1_behavioral.csv"
        if p.exists():
            df = pd.read_csv(p, dtype=str).fillna("")
            df["variant_type"] = df["variant_type"].apply(_norm_variant)
            df["_correct"] = df["behavioral_correct"].str.lower().isin(["true","1","yes"])
            df = df[df["model"].isin(["anthropic/claude-sonnet-4","openai/gpt-4o","meta-llama/llama-3.1-8b-instruct"])]
            df = df[["problem_id","model","variant_type","_correct"]]
            df = df.drop_duplicates(["problem_id","model","variant_type"], keep="last")
            parts.append(df)

    if not parts: return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def per_problem_table() -> pd.DataFrame:
    """Build master (problem_id, model) feature table across families."""
    rows = []
    for fam in ["ALGO","GSM","BW"]:
        df = _load_p1(fam)
        if df.empty: continue
        wide = df.pivot_table(index=["problem_id","model"], columns="variant_type",
                              values="_correct", aggfunc="last").reset_index()
        wide["family"] = fam
        # per-problem VAR: 1 - mean(W1..W6)
        var_cols = [c for c in ["W1","W2","W3","W4","W5","W6"] if c in wide.columns]
        wide["mean_variant"] = wide[var_cols].astype(float).mean(axis=1)
        wide["VAR"] = 1 - wide["mean_variant"]
        # W3-retention per problem (binary)
        if "W3" in wide.columns and "canonical" in wide.columns:
            wide["W3_kept"] = wide["W3"].astype(float)
            wide["canonical_correct"] = wide["canonical"].astype(float)
            # only defined where canonical=correct
            wide["W3_retention_individual"] = np.where(
                wide["canonical_correct"] == 1, wide["W3_kept"], np.nan
            )
        rows.append(wide)
    if not rows: return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["model_short"] = out["model"].map(SHORT)
    return out


# -------- new metric: per-model implausibility detection -------------------

def implausibility_detection() -> pd.DataFrame:
    """Per model: final_correct in plausible vs implausible (on the SAME
    61 adversarial problems), and a paired Wilcoxon test."""
    plaus = pd.read_csv(RAW / "ALGO_P2_phase2_injected.csv", dtype=str).fillna("")
    impl  = pd.read_csv(RAW / "ALGO_P2_phase2_injected_implausible.csv", dtype=str).fillna("")
    rows = []
    for m in MODELS:
        pp = plaus[(plaus["model"]==m) & (plaus["post_injection_correct"] != "")]
        ii = impl[(impl["model"]==m)  & (impl["post_injection_correct"] != "")]
        if pp.empty or ii.empty:
            continue
        pp = pp.set_index("problem_id")["post_injection_correct"].astype(str).str.lower()
        ii = ii.set_index("problem_id")["post_injection_correct"].astype(str).str.lower()
        common = pp.index.intersection(ii.index)
        p_arr = (pp.loc[common] == "true").astype(int).values
        i_arr = (ii.loc[common] == "true").astype(int).values
        if len(p_arr) < 5:
            continue
        try:
            stat = stats.wilcoxon(i_arr, p_arr, zero_method="zsplit")
            pval = stat.pvalue
        except ValueError:
            pval = float("nan")
        rows.append({
            "model": SHORT[m],
            "n_paired_problems": len(common),
            "plausible_correct":  float(p_arr.mean()),
            "implausible_correct": float(i_arr.mean()),
            "delta_implaus_minus_plaus": float(i_arr.mean() - p_arr.mean()),
            "wilcoxon_p": pval,
        })
    df = pd.DataFrame(rows)
    df.to_csv(DER / "implausibility_detection.csv", index=False)
    return df


# -------- new metric: stated-algorithm rate per model (ALGO Phase 1) --------

def stated_algorithm_rate() -> pd.DataFrame:
    parts = []
    for tag in ["claude_new","gemini","gpt4o","gpt4o_new","llama","llama_new"]:
        p = RAW / f"ALGO_P2_phase1_{tag}.csv"
        if not p.exists(): continue
        df = pd.read_csv(p, dtype=str).fillna("")
        parts.append(df)
    if not parts: return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df = df.drop_duplicates(["problem_id","model","instance_type"], keep="last")
    rows = []
    for m in df["model"].unique():
        sub = df[df["model"] == m]
        n = len(sub)
        sa = sub["stated_algorithm"].astype(str).str.strip().str.lower()
        # cleanly-stated correct algorithm names
        correct_names = {
            "dynamic programming","dp","greedy","dijkstra","bfs","dfs",
            "shortest path","coin change","weighted interval scheduling",
            "wis","knapsack","bellman-ford","floyd-warshall",
            "interval scheduling",
        }
        named = (sa != "").sum()
        named_correct = sa.apply(lambda x: any(n in x for n in correct_names)).sum()
        rows.append({
            "model": SHORT.get(m, m), "n": n,
            "named_rate": named/n if n else float("nan"),
            "named_correct_rate": named_correct/n if n else float("nan"),
        })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(DER / "algorithm_invocation_clean.csv", index=False)
    return df_out


# -------- new metric: BW violation profile per model -----------------------

def bw_violation_profile() -> pd.DataFrame:
    p = RAW / "BW_P2_cci.csv"
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(p, dtype=str).fillna("")
    violation_cols = [c for c in df.columns if c.startswith("violation_")
                                              and c != "violation_profile_json"]
    rows = []
    for m in df["model"].unique():
        sub = df[df["model"] == m]
        cci = pd.to_numeric(sub["cci"], errors="coerce")
        first_illegal = pd.to_numeric(sub["first_illegal_step"], errors="coerce")
        sem_val = pd.to_numeric(sub["semantic_validity_rate"], errors="coerce")
        rep_rate = pd.to_numeric(sub["repetition_rate"], errors="coerce")
        partial_goal = pd.to_numeric(sub["partial_goal_achievement"], errors="coerce")
        rec = {
            "model": SHORT.get(m, m),
            "n_sessions": len(sub),
            "mean_cci":              float(cci.mean()),
            "median_first_illegal":  float(first_illegal.median()) if first_illegal.notna().any() else float("nan"),
            "mean_semantic_validity": float(sem_val.mean()),
            "mean_repetition_rate":  float(rep_rate.mean()),
            "mean_partial_goal":     float(partial_goal.mean()),
        }
        # dominant violation type
        viol_means = {c.replace("violation_",""): pd.to_numeric(sub[c], errors="coerce").mean() for c in violation_cols}
        viol_means = {k: v for k,v in viol_means.items() if not np.isnan(v) and v > 0}
        if viol_means:
            top = max(viol_means.items(), key=lambda x: x[1])
            rec["top_violation"] = top[0]
            rec["top_violation_rate"] = top[1]
        rows.append(rec)
    df_out = pd.DataFrame(rows)
    df_out.to_csv(DER / "bw_violation_profile.csv", index=False)
    return df_out


# -------- new metric: cross-family universally fragile problems ------------

def universally_fragile_problems(master: pd.DataFrame) -> pd.DataFrame:
    """Problems where canonical=correct for >= 4 models but W3=0 for >= 4 models."""
    rows = []
    for fam in master["family"].unique():
        sub = master[master["family"] == fam]
        piv_canon = sub.pivot_table(index="problem_id", columns="model_short",
                                    values="canonical_correct", aggfunc="last")
        piv_w3 = sub.pivot_table(index="problem_id", columns="model_short",
                                 values="W3_kept", aggfunc="last")
        if piv_canon.empty or piv_w3.empty: continue
        # for each problem, count models with canon=1 and w3=0
        n_canon_correct = (piv_canon == 1).sum(axis=1)
        n_w3_collapse  = ((piv_canon == 1) & (piv_w3 == 0)).sum(axis=1)
        for pid in piv_canon.index:
            if int(n_canon_correct.get(pid, 0)) >= 4 and int(n_w3_collapse.get(pid, 0)) >= 4:
                rows.append({
                    "family": fam, "problem_id": pid,
                    "n_models_canon_correct": int(n_canon_correct.get(pid, 0)),
                    "n_models_w3_collapse": int(n_w3_collapse.get(pid, 0)),
                })
    df = pd.DataFrame(rows)
    df.to_csv(DER / "cross_family_universally_fragile.csv", index=False)
    return df


# -------- cross-probe correlations within model ----------------------------

def cross_probe_correlations(master: pd.DataFrame) -> pd.DataFrame:
    """Within each (family, model), compute correlations between behavioral
    metrics (canonical, VAR, W3-retention) and other available signals."""
    rows = []
    for (fam, model), g in master.groupby(["family", "model_short"]):
        if len(g) < 10: continue
        v_canon = pd.to_numeric(g["canonical_correct"], errors="coerce")
        v_var   = pd.to_numeric(g["VAR"], errors="coerce")
        v_w3    = pd.to_numeric(g["W3_kept"], errors="coerce")
        # canonical vs VAR
        valid = v_canon.notna() & v_var.notna()
        if valid.sum() >= 10:
            rho, p = stats.spearmanr(v_canon[valid], v_var[valid])
            rows.append({"family":fam, "model":model, "x":"canonical","y":"VAR",
                         "n":int(valid.sum()),"rho":rho,"p":p})
        # canonical vs W3-retention
        valid = v_canon.notna() & v_w3.notna()
        if valid.sum() >= 10:
            rho, p = stats.spearmanr(v_canon[valid], v_w3[valid])
            rows.append({"family":fam,"model":model,"x":"canonical","y":"W3_kept",
                         "n":int(valid.sum()),"rho":rho,"p":p})
    df = pd.DataFrame(rows)
    df.to_csv(DER / "cross_probe_corr_within_model.csv", index=False)
    return df


def main():
    print("=== building per-problem master ===")
    master = per_problem_table()
    master.to_csv(DER / "master_per_problem_5model.csv", index=False)
    print(f"  master: {len(master)} (problem, model) rows; families: {master['family'].unique().tolist()}")

    print("\n=== implausibility detection ===")
    impl = implausibility_detection()
    print(impl.round(3).to_string(index=False))

    print("\n=== stated-algorithm rate (ALGO P2 Phase 1) ===")
    sar = stated_algorithm_rate()
    print(sar.round(3).to_string(index=False))

    print("\n=== BW violation profile ===")
    bw = bw_violation_profile()
    print(bw.round(3).to_string(index=False))

    print("\n=== universally fragile cross-model ===")
    uf = universally_fragile_problems(master)
    print(f"  universally fragile: {len(uf)} problems")
    if len(uf):
        print(uf.head(10).to_string(index=False))

    print("\n=== cross-probe correlations within model ===")
    cp = cross_probe_correlations(master)
    print(cp.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
