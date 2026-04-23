"""
Final triangulation analysis. Run after all three probes
are complete. Safe to run with partial data — missing probes produce None signals
rather than crashing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probes.common.io import load_results, QUESTION_BANK_PATH
from probes.behavioral.css import (
    compute_css,
    compute_var,
    compute_pdas,
    compute_pdas_reversal,
    compute_dts,
    compute_vri,
    compute_cfs,
)
from probes.triangulation.per_instance import align_instance


def _question_bank_meta_lookup(path: str) -> dict[tuple[str, str], dict[str, str]]:
    """(problem_id, variant_type lower) -> correct_answer, problem_text from bank."""
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p, dtype=str)
    out: dict[tuple[str, str], dict[str, str]] = {}
    for _, row in df.iterrows():
        pid = str(row.get("problem_id", "")).strip()
        if not pid:
            continue
        vt = str(row.get("variant_type", "")).strip().lower()
        ca = row.get("correct_answer", "")
        ca = "" if pd.isna(ca) else str(ca)
        pt = row.get("problem_text", "")
        pt = "" if pd.isna(pt) else str(pt)
        out[(pid, vt)] = {"correct_answer": ca, "problem_text": pt}
    return out


def _behavioral_slice_for_css(df: pd.DataFrame, model: str | None) -> tuple[pd.DataFrame, str]:
    """Return rows for one model so CSS is not mixed across models.

    If ``model`` is set, keep only that id. If unset, use the sole non-mock
    model if exactly one exists; otherwise exit with an error listing choices.
    """
    if df.empty or "model" not in df.columns:
        return df, model or ""
    mcol = df["model"].astype(str).str.strip()
    if model:
        sel = mcol == str(model).strip()
        out = df.loc[sel].copy()
        if out.empty:
            print(f"ERROR: no behavioral rows for --behavioral-model {model!r}")
            sys.exit(2)
        return out, str(model).strip()
    non_mock = df.loc[mcol.str.lower() != "mock"].copy()
    if non_mock.empty:
        return df.copy(), ""
    u = sorted(non_mock["model"].astype(str).str.strip().unique())
    if len(u) == 1:
        return non_mock, u[0]
    print(
        "ERROR: behavioral_sweep.csv has multiple non-mock models; "
        "CSS must use one model at a time. Pass --behavioral-model with one of:\n  "
        + "\n  ".join(u)
    )
    sys.exit(2)


def main():
    parser = argparse.ArgumentParser(description="Triangulation Analysis")
    parser.add_argument(
        "--behavioral",
        "--behavioral-results",
        dest="behavioral",
        type=str,
        default="results/BW_P1_RES_behavioral_sweep.csv",
    )
    parser.add_argument(
        "--behavioral-model",
        type=str,
        default=None,
        help=(
            "OpenRouter model id: restrict behavioral rows to this model for CSS. "
            "Required when the CSV contains more than one non-mock model."
        ),
    )
    parser.add_argument("--mechanistic", type=str, default="results/BW_RES_P3_probe1_mechanistic.csv")
    parser.add_argument(
        "--contamination",
        "--contamination-results",
        dest="contamination",
        type=str,
        default="results/BW_P3_RES_contamination_triage.csv",
    )
    parser.add_argument(
        "--probe2-results",
        type=str,
        default="results/GSM_P2_RES_cci.csv",
        help="Probe 2 results containing cci_score by problem/model",
    )
    parser.add_argument(
        "--family",
        type=str,
        default=None,
        help="Optional problem_family filter (e.g., arithmetic_reasoning)",
    )
    parser.add_argument(
        "--question-bank",
        type=str,
        default=QUESTION_BANK_PATH,
        help="Used to fill correct_answer / problem_text for CSS when missing from behavioral CSV",
    )
    # Available output name choices: per-model files are written by the caller via --output
    parser.add_argument("--output", type=str, default="results/BW_P3_RES_triangulation_per_instance_claude37.csv")
    parser.add_argument(
        "--regression-output",
        type=str,
        default="results/BW_P3_RES_contamination_regression_claude37.txt",
        help="Where to write the OLS summary (default: results/BW_P3_RES_contamination_regression_claude37.txt)",
    )
    args = parser.parse_args()

    # 1. Load data safely
    behavioral_path = args.behavioral
    if not Path(behavioral_path).exists() and behavioral_path == "results/BW_P1_RES_behavioral_sweep.csv":
        behavioral_path = "results/behavioral_sweep.csv"
    mechanistic_path = args.mechanistic
    if not Path(mechanistic_path).exists() and mechanistic_path == "results/BW_RES_P3_probe1_mechanistic.csv":
        mechanistic_path = "results/probe1_mechanistic.csv"
    contamination_path = args.contamination
    if not Path(contamination_path).exists() and contamination_path == "results/BW_P3_RES_contamination_triage.csv":
        contamination_path = "results/contamination_triage.csv"

    df_beh = load_results(behavioral_path)
    df_mech = load_results(mechanistic_path)
    df_cont = load_results(contamination_path)
    df_probe2 = load_results(args.probe2_results)

    # Section-1 style cleanup for analysis consistency.
    if not df_beh.empty:
        if "problem_id" in df_beh.columns:
            df_beh = df_beh[~df_beh["problem_id"].astype(str).str.contains("_W5_TEMP", na=False)].copy()
        if "model" in df_beh.columns:
            df_beh = df_beh[df_beh["model"].astype(str) != "meta-llama/llama-3-8b-instruct"].copy()
            df_beh = df_beh[df_beh["model"].astype(str).str.lower() != "mock"].copy()

    behavioral_model_resolved = ""
    if not df_beh.empty and "variant_type" in df_beh.columns:
        df_beh, behavioral_model_resolved = _behavioral_slice_for_css(
            df_beh, args.behavioral_model
        )
        if behavioral_model_resolved:
            print(f"Behavioral CSS slice: model={behavioral_model_resolved!r}")

    if args.family:
        fam = args.family.strip().lower()
        for name, df in [("behavioral", df_beh), ("mechanistic", df_mech), ("contamination", df_cont)]:
            if not df.empty and "problem_family" in df.columns:
                mask = df["problem_family"].astype(str).str.strip().str.lower() == fam
                if name == "behavioral":
                    df_beh = df.loc[mask].copy()
                elif name == "mechanistic":
                    df_mech = df.loc[mask].copy()
                else:
                    df_cont = df.loc[mask].copy()

    if not df_beh.empty and "problem_id" in df_beh.columns:
        # Normalize problem_id to base canonical ID.
        # Old W5 rows were stored as BW_001_W5_TEMP; W6 rows as BW_001_W6 or BW_E002_W6.
        # Strip all these suffixes so they group under their canonical base problem.
        def _base_id(pid: str) -> str:
            import re
            pid = pid.strip()
            pid = re.sub(r"_W5_TEMP$", "", pid)
            pid = re.sub(r"_W[0-9]+$", "", pid)
            return pid

        def _infer_vtype(pid: str, existing_vtype: str) -> str:
            """If problem_id encodes the variant, use that; otherwise keep existing."""
            import re
            if existing_vtype and existing_vtype.strip().upper() not in ("", "NAN", "NONE"):
                match = re.search(r"_(W[0-9]+)(?:_TEMP)?$", pid.strip())
                if match:
                    # problem_id has a variant suffix — use it as override
                    return match.group(1)
                return existing_vtype
            match = re.search(r"_(W[0-9]+)(?:_TEMP)?$", pid.strip())
            return match.group(1) if match else existing_vtype

        df_beh["base_id"] = df_beh["problem_id"].astype(str).apply(_base_id)
        df_beh["variant_type"] = df_beh.apply(
            lambda r: _infer_vtype(str(r["problem_id"]), str(r.get("variant_type", ""))),
            axis=1,
        )

    available = []
    if not df_beh.empty: available.append("Behavioral")
    if not df_mech.empty: available.append("Mechanistic")
    if not df_cont.empty: available.append("Contamination")

    print(f"Probes available: {', '.join(available) if available else 'None'}")

    # Extract base problem_family metadata across all files to assure coverage
    families = {}
    for df in [df_beh, df_mech, df_cont]:
        if not df.empty and "problem_id" in df.columns and "problem_family" in df.columns:
            for _, row in df.iterrows():
                families[str(row["problem_id"]).strip()] = str(row["problem_family"])

    # 2. Compute VAR (primary) + CSS (secondary) from behavioral CSV
    beh_data = []
    if not df_beh.empty and "problem_id" in df_beh.columns:
        if "variant_type" not in df_beh.columns:
            print("WARNING: behavioral_sweep.csv has no variant_type column. "
                  "CSS will be None for all problems. "
                  "Re-run behavioral sweep with variant support before triangulation.")
        else:
            bank_meta = _question_bank_meta_lookup(args.question_bank)
            df_css = df_beh.copy()
            for pid, group in df_css.groupby("base_id"):
                pid_str = str(pid).strip()
                # Prefer behavioral row family (blocksworld / mystery_blocksworld). Contamination
                # CSV may use umbrella labels like planning_suite that verify_answer rejects.
                if "problem_family" in group.columns:
                    gf = (
                        group["problem_family"].dropna().astype(str).str.strip().str.lower()
                    )
                    gf = gf[gf.isin(
                        {"blocksworld", "mystery_blocksworld", "logistics", "gsm",
                         "shortest_path", "weighted_interval_scheduling", "coin_change",
                         "knapsack"}
                    )]
                    fam = gf.iloc[0] if len(gf) else ""
                else:
                    fam = ""
                if not fam:
                    fam = families.get(pid_str, "blocksworld")

                canonical_correct = ""
                canonical_bool = None
                w5_bool = None
                variants = []
                for _, row in group.iterrows():
                    vtype = str(row.get("variant_type", "")).strip()
                    vkey = vtype.lower()
                    row_pid = str(row.get("problem_id", "")).strip()
                    # Look up meta using base_id (canonical pid) when row_pid is a variant-suffixed id
                    meta = bank_meta.get((pid_str, vkey), bank_meta.get((row_pid, vkey), {}))
                    row_ca = row.get("correct_answer", "")
                    row_ca = "" if pd.isna(row_ca) else str(row_ca)
                    corr = row_ca or meta.get("correct_answer", "")
                    ptext = meta.get("problem_text", "")
                    model_ans = str(row.get("model_answer", "") or "").strip()
                    if not model_ans:
                        raw = row.get("raw_response", "")
                        model_ans = "" if pd.isna(raw) else str(raw)

                    if (
                        pd.isna(row.get("variant_type"))
                        or vtype == ""
                        or vkey in ("nan", "none", "canonical")
                    ):
                        canonical_correct = corr
                        bc = row.get("behavioral_correct", None)
                        if pd.notna(bc):
                            sval = str(bc).strip().lower()
                            if sval in {"true", "false"}:
                                canonical_bool = 1.0 if sval == "true" else 0.0
                    elif vtype.startswith("W") and vtype not in ("W5", "W6"):
                        variants.append(
                            {
                                "variant_type": vtype,
                                "model_answer": model_ans,
                                "correct_answer": corr,
                                "problem_text": ptext,
                            }
                        )

                    elif vtype == "W5":
                        bc = row.get("behavioral_correct", None)
                        if pd.notna(bc):
                            sval = str(bc).strip().lower()
                            if sval in {"true", "false"}:
                                w5_bool = 1.0 if sval == "true" else 0.0

                css_dict = compute_css(pid_str, canonical_correct, variants, fam)
                beh_data.append({
                    "problem_id": pid_str,
                    "var_canonical": canonical_bool,
                    "var_w5": w5_bool,
                    "css": css_dict.get("css"),
                    "behavioral_model": behavioral_model_resolved,
                })

    df_css_agg = pd.DataFrame(beh_data)
    if df_css_agg.empty:
        df_css_agg = pd.DataFrame(
            columns=["problem_id", "var_canonical", "var_w5", "css", "behavioral_model"]
        )

    # 3 & 4. Extract Contamination & Mechanistic data
    if not df_cont.empty and "problem_id" in df_cont.columns:
        df_cont_filt = df_cont[["problem_id", "contamination_score"]].copy()
        df_cont_filt["problem_id"] = df_cont_filt["problem_id"].astype(str).str.strip()
        df_cont_filt = df_cont_filt.drop_duplicates("problem_id")
    else:
        df_cont_filt = pd.DataFrame(columns=["problem_id", "contamination_score"])

    if not df_mech.empty and "problem_id" in df_mech.columns:
        df_mech_filt = df_mech[["problem_id", "crystallization_layer"]].copy()
        df_mech_filt["problem_id"] = df_mech_filt["problem_id"].astype(str).str.strip()
        df_mech_filt = df_mech_filt.drop_duplicates("problem_id")
    else:
        df_mech_filt = pd.DataFrame(columns=["problem_id", "crystallization_layer"])

    if not df_probe2.empty and "problem_id" in df_probe2.columns and "cci_score" in df_probe2.columns:
        p2 = df_probe2.copy()
        if "valid_divergence" in p2.columns:
            vd = (
                p2["valid_divergence"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map({"true": True, "false": False})
                .fillna(False)
            )
            p2 = p2.loc[~vd].copy()
        if "model" in p2.columns and behavioral_model_resolved:
            p2 = p2[p2["model"].astype(str).str.strip() == behavioral_model_resolved].copy()
        p2["problem_id"] = p2["problem_id"].astype(str).str.strip()
        p2["cci_score"] = pd.to_numeric(p2["cci_score"], errors="coerce")
        p2 = p2.dropna(subset=["cci_score"])
        df_p2_filt = (
            p2.groupby("problem_id", as_index=False)["cci_score"]
            .mean()
            .rename(columns={"cci_score": "cci"})
        )
    else:
        df_p2_filt = pd.DataFrame(columns=["problem_id", "cci"])

    # 5. Outer join all datasets 
    all_pids = set()
    for df in [df_css_agg, df_cont_filt, df_mech_filt, df_p2_filt]:
        if "problem_id" in df.columns:
            all_pids.update(df["problem_id"].tolist())
    
    df_merged = pd.DataFrame({"problem_id": list(all_pids)})
    
    if not df_css_agg.empty:
        df_merged = df_merged.merge(df_css_agg, on="problem_id", how="left")
    if not df_cont_filt.empty:
        df_merged = df_merged.merge(df_cont_filt, on="problem_id", how="left")
    if not df_mech_filt.empty:
        df_merged = df_merged.merge(df_mech_filt, on="problem_id", how="left")
    if not df_p2_filt.empty:
        df_merged = df_merged.merge(df_p2_filt, on="problem_id", how="left")

    df_merged["problem_family"] = df_merged["problem_id"].map(families)

    # 6. Apply per-instance alignment via properties
    final_rows = []
    for _, row in df_merged.iterrows():
        pid = row["problem_id"]
        
        css_val = float(row["css"]) if "css" in row and pd.notna(row["css"]) else None
        var_val = (
            float(row["var_canonical"])
            if "var_canonical" in row and pd.notna(row["var_canonical"])
            else None
        )
        cont_val = float(row["contamination_score"]) if "contamination_score" in row and pd.notna(row["contamination_score"]) else None
        
        cci_val = float(row["cci"]) if "cci" in row and pd.notna(row["cci"]) else None
        alignment_dict = align_instance(
            problem_id=str(pid),
            var=var_val,
            css=css_val,
            contamination_score=cont_val,
            cci=cci_val
        )
        
        d = row.to_dict()
        d.update(alignment_dict)
        final_rows.append(d)

    df_final = pd.DataFrame(final_rows)

    # 7. Contamination Regression via statsmodels
    import statsmodels.formula.api as smf
    reg_sections = []

    # Regression A: VAR(canonical) ~ contamination_score + family FE
    req_a = ["var_canonical", "contamination_score", "problem_family"]
    if all(c in df_final.columns for c in req_a):
        df_reg_a = df_final.dropna(subset=req_a)
        if len(df_reg_a) >= 10 and df_reg_a["var_canonical"].nunique() >= 2:
            try:
                model_a = smf.ols(
                    "var_canonical ~ contamination_score + C(problem_family)", data=df_reg_a
                ).fit()
                print("\n--- Contamination Regression (VAR canonical) ---")
                print(model_a.summary())
                reg_sections.append("=== VAR(canonical) regression ===\n" + model_a.summary().as_text() + "\n")
            except Exception as e:
                print(f"\nFailed VAR(canonical) regression: {e}")
        else:
            print("\nSkipping VAR(canonical) regression — insufficient rows or zero variance.")
    else:
        print("\nSkipping VAR(canonical) regression — missing columns.")

    # Regression B: VAR(W5) ~ contamination_score + family FE
    req_b = ["var_w5", "contamination_score", "problem_family"]
    if all(c in df_final.columns for c in req_b):
        df_reg_b = df_final.dropna(subset=req_b)
        if len(df_reg_b) >= 10 and df_reg_b["var_w5"].nunique() >= 2:
            try:
                model_b = smf.ols(
                    "var_w5 ~ contamination_score + C(problem_family)", data=df_reg_b
                ).fit()
                print("\n--- Contamination Regression (VAR W5) ---")
                print(model_b.summary())
                reg_sections.append("=== VAR(W5) regression ===\n" + model_b.summary().as_text() + "\n")
            except Exception as e:
                print(f"\nFailed VAR(W5) regression: {e}")
        else:
            print("\nSkipping VAR(W5) regression — insufficient rows or zero variance.")
    else:
        print("\nSkipping VAR(W5) regression — missing columns.")

    # Supplemental model-level matrices and derived metrics
    if not df_beh.empty and all(c in df_beh.columns for c in ["variant_type", "model", "behavioral_correct"]):
        model_ids = sorted(df_beh["model"].astype(str).str.strip().unique())
        variant_order = ["canonical", "W1", "W2", "W3", "W4", "W5", "W6"]
        print("\n--- VAR Matrix (model x variant) ---")
        for m in model_ids:
            vals = []
            for v in variant_order:
                vv = compute_var(df_beh, v, m)
                vals.append(f"{v}:{vv if vv is not None else 'NA'}")
            print(f"{m} | " + " | ".join(vals))

        print("\n--- PDAS / PDAS_reversal ---")
        for m in model_ids:
            print(
                f"{m} | PDAS={compute_pdas(df_beh, m)} | "
                f"PDAS_reversal={compute_pdas_reversal(df_beh, m)}"
            )

        print("\n--- DTS (W2/W3/W4/W5) ---")
        for m in model_ids:
            print(
                f"{m} | W2={compute_dts(df_beh, 'W2', m)} | "
                f"W3={compute_dts(df_beh, 'W3', m)} | "
                f"W4={compute_dts(df_beh, 'W4', m)} | "
                f"W5={compute_dts(df_beh, 'W5', m)}"
            )

        print("\n--- VRI ---")
        for m in model_ids:
            vri = compute_vri(df_beh, m)
            print(
                f"{m} | structural={vri['structural']} | "
                f"vocabulary={vri['vocabulary']} | gap={vri['gap']}"
            )

        print("\n--- CFS (canonical/W2/W3/W4 first-action consistency) ---")
        base_ids = sorted(df_beh["problem_id"].astype(str).str.replace(r"_W[0-9]+$", "", regex=True).unique())
        for m in model_ids:
            vals = [compute_cfs(df_beh, pid, m) for pid in base_ids]
            vals = [v for v in vals if v is not None]
            cfs_mean = round(sum(vals) / len(vals), 4) if vals else None
            print(f"{m} | cfs_mean={cfs_mean}")

    if reg_sections:
        Path("results").mkdir(parents=True, exist_ok=True)
        reg_path = Path(args.regression_output)
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        with reg_path.open("w", encoding="utf-8") as f:
            f.write("\n\n".join(reg_sections))
        print(f"Wrote regression summary to {reg_path}")

    # 8. Set to output paths natively
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(args.output, index=False)

    # 9. Triangulation Metrics Summaries Display
    print("\n--- Final Triangulation Summary ---")
    print(f"Total problems triangulated: {len(df_final)}")
    
    if not df_final.empty:
        valid_mask = df_final["agreement"] != "insufficient"
        valid_count = valid_mask.sum()
        converging_count = (df_final["agreement"] == "converging").sum()
        
        if valid_count > 0:
            rate = converging_count / valid_count
            print(f"Convergence rate (on valid instances): {rate:.2%}")
        else:
            print("Convergence rate: N/A (no valid instances with >=2 signals)")
            
        print("\nBreakdown:")
        print(df_final["diagnosis"].value_counts().to_string())


if __name__ == "__main__":
    main()
