"""
Triangulation module to align Probe 1, Probe 2, and Probe 3 signals on a per-instance basis.
Per-instance convergence (whether all applicable probes agree on the same diagnosis
for a single problem instance) is the core analytical contribution of the paper.
This allows us to move beyond aggregate metrics and robustly measure genuine reasoning.
"""

from __future__ import annotations

import pandas as pd


def align_instance(
    problem_id: str,
    css: float | None,
    contamination_score: float | None,
    cci: float | None = None,
) -> dict:
    
    css_signal = None
    if css is not None:
        css_signal = "computation" if css >= 0.5 else "retrieval"

    contamination_signal = None
    if contamination_score is not None:
        if contamination_score > 0.6:
            contamination_signal = "retrieval"
        elif contamination_score <= 0.4:
            contamination_signal = "computation"
        else:
            contamination_signal = "ambiguous"

    cci_signal = None
    if cci is not None:
        cci_signal = "computation" if cci >= 0.4 else "retrieval"

    signals = [s for s in [css_signal, contamination_signal, cci_signal] if s is not None]
    n_signals = len(signals)

    if n_signals < 2:
        agreement = "insufficient"
    elif "ambiguous" in signals:
        agreement = "ambiguous"
    elif "retrieval" in signals and "computation" in signals:
        agreement = "diverging"
    else:
        agreement = "converging"

    if agreement == "converging":
        # Safe to assume all elements are identical and standard
        diagnosis = f"converging_{signals[0]}"
    else:
        diagnosis = agreement

    return {
        "problem_id": problem_id,
        "css_signal": css_signal,
        "contamination_signal": contamination_signal,
        "cci_signal": cci_signal,
        "agreement": agreement,
        "n_signals": n_signals,
        "diagnosis": diagnosis,
    }


def align_all(results: list[dict]) -> pd.DataFrame:
    aligned_results = []
    
    for row in results:
        aligned_results.append(align_instance(
            problem_id=row.get("problem_id", ""),
            css=row.get("css"),
            contamination_score=row.get("contamination_score"),
            cci=row.get("cci")
        ))
        
    df = pd.DataFrame(aligned_results)
    
    print("--- Triangulation Summary ---")
    
    if df.empty:
        print("No results to display.")
        return df
        
    print(df["diagnosis"].value_counts().to_string())
    
    valid_instances = df[df["agreement"] != "insufficient"]
    total_valid = len(valid_instances)
    converging_count = len(df[df["agreement"] == "converging"])
    
    if total_valid > 0:
        rate = converging_count / total_valid
        print(f"\nConvergence Rate (across {total_valid} valid instances): {rate:.2%}")
    else:
        print("\nConvergence Rate: N/A (no valid instances with >=2 signals)")
        
    return df
