"""
Final triangulation analysis. Run after all three probes
are complete. Safe to run with partial data — missing probes produce None signals
rather than crashing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from probes.common.io import load_results
from probes.behavioral.css import compute_css
from probes.triangulation.per_instance import align_instance


def main():
    parser = argparse.ArgumentParser(description="Triangulation Analysis")
    parser.add_argument("--behavioral", type=str, default="results/behavioral_sweep.csv")
    parser.add_argument("--mechanistic", type=str, default="results/probe1_mechanistic.csv")
    parser.add_argument("--contamination", type=str, default="results/contamination_triage.csv")
    parser.add_argument("--output", type=str, default="results/triangulation_per_instance.csv")
    args = parser.parse_args()

    # 1. Load data safely
    df_beh = load_results(args.behavioral)
    df_mech = load_results(args.mechanistic)
    df_cont = load_results(args.contamination)

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

    # 2. Compute CSS from behavioral CSV
    beh_data = []
    if not df_beh.empty and "problem_id" in df_beh.columns:
        if "variant_type" not in df_beh.columns:
            print("WARNING: behavioral_sweep.csv has no variant_type column. "
                  "CSS will be None for all problems. "
                  "Re-run behavioral sweep with variant support before triangulation.")
        else:
            for pid, group in df_beh.groupby("problem_id"):
                pid_str = str(pid).strip()
                fam = families.get(pid_str, "blocksworld")
                
                canonical_correct = ""
                variants = []
                for _, row in group.iterrows():
                    vtype = str(row.get("variant_type", "")).strip()
                    if pd.isna(row.get("variant_type")) or vtype == "" or vtype.lower() in ["nan", "none", "canonical"]:
                        canonical_correct = str(row.get("correct_answer", ""))
                    elif vtype.startswith("W") and vtype != "W6":
                        variants.append({
                            "variant_type": vtype,
                            "model_answer": str(row.get("model_answer", "")),
                            "correct_answer": str(row.get("correct_answer", ""))
                        })
                
                css_dict = compute_css(pid_str, canonical_correct, variants, fam)
                beh_data.append({
                    "problem_id": pid_str,
                    "css": css_dict.get("css")
                })

    df_css_agg = pd.DataFrame(beh_data)
    if df_css_agg.empty:
        df_css_agg = pd.DataFrame(columns=["problem_id", "css"])

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

    # 5. Outer join all datasets 
    all_pids = set()
    for df in [df_css_agg, df_cont_filt, df_mech_filt]:
        if "problem_id" in df.columns:
            all_pids.update(df["problem_id"].tolist())
    
    df_merged = pd.DataFrame({"problem_id": list(all_pids)})
    
    if not df_css_agg.empty:
        df_merged = df_merged.merge(df_css_agg, on="problem_id", how="left")
    if not df_cont_filt.empty:
        df_merged = df_merged.merge(df_cont_filt, on="problem_id", how="left")
    if not df_mech_filt.empty:
        df_merged = df_merged.merge(df_mech_filt, on="problem_id", how="left")

    df_merged["problem_family"] = df_merged["problem_id"].map(families)

    # 6. Apply per-instance alignment via properties
    final_rows = []
    for _, row in df_merged.iterrows():
        pid = row["problem_id"]
        
        css_val = float(row["css"]) if "css" in row and pd.notna(row["css"]) else None
        cont_val = float(row["contamination_score"]) if "contamination_score" in row and pd.notna(row["contamination_score"]) else None
        
        # TODO: wire in Probe 2 results when available (cci=None for now)
        alignment_dict = align_instance(
            problem_id=str(pid),
            css=css_val,
            contamination_score=cont_val,
            cci=None
        )
        
        d = row.to_dict()
        d.update(alignment_dict)
        final_rows.append(d)

    df_final = pd.DataFrame(final_rows)

    # 7. Contamination Regression via statsmodels
    df_reg = df_final.dropna(subset=["css", "contamination_score", "problem_family"])
    if len(df_reg) >= 10:
        import statsmodels.formula.api as smf
        try:
            model = smf.ols("css ~ contamination_score + C(problem_family)", data=df_reg).fit()
            print("\n--- Contamination Regression Summary ---")
            print(model.summary())
            
            Path("results").mkdir(parents=True, exist_ok=True)
            with open("results/contamination_regression.txt", "w", encoding="utf-8") as f:
                f.write(model.summary().as_text())
        except Exception as e:
            print(f"\nFailed to run regression: {e}")
    else:
        print("\nInsufficient data for regression (need >= 10 problems with both scores).")

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
