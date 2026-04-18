import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot Phase 1 Contamination Triage outcomes")
    parser.add_argument(
        "--results", 
        type=str, 
        default="results/contamination_triage.csv", 
        help="Path to contamination triage CSV"
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: {results_path} does not exist.")
        return

    df = pd.read_csv(results_path)
    
    required_cols = {"problem_id", "contamination_score"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Error: Required columns are missing from the CSV: {missing}")
        return

    if "problem_family" in df.columns:
        families = df["problem_family"].dropna().unique()
    else:
        families = ["unknown"]
        df["problem_family"] = "unknown"

    # Print summary statistics
    print("Summary statistics (mean, min, max) for Contamination Score:")
    df["contamination_score"] = pd.to_numeric(df["contamination_score"], errors="coerce")
    summary = df.groupby("problem_family")["contamination_score"].agg(["mean", "min", "max"])
    print(summary.to_string())
    print()

    # Create figures directory if it doesn't exist
    Path("figures").mkdir(parents=True, exist_ok=True)
    out_path = Path("figures/probe1_triage.png")

    fig, ax = plt.subplots(figsize=(8, 6))

    if "pilot_correct" in df.columns:
        # We have pilot accuracy data, so do a scatter plot
        df["pilot_correct"] = pd.to_numeric(df["pilot_correct"], errors="coerce")
        
        for family in families:
            subset = df[df["problem_family"] == family].dropna(subset=["contamination_score", "pilot_correct"])
            if subset.empty:
                continue
                
            x = subset["contamination_score"].values
            y = subset["pilot_correct"].values
            
            # Jitter y slightly for visibility
            y_jittered = y + np.random.uniform(-0.05, 0.05, size=len(y))
            
            ax.scatter(x, y_jittered, label=family, alpha=0.7)
            
        # Add a linear trend line using numpy polyfit across the entire dataset
        valid_df = df.dropna(subset=["contamination_score", "pilot_correct"])
        if len(valid_df) > 1:
            x_vals = valid_df["contamination_score"].values
            y_vals = valid_df["pilot_correct"].values
            
            m, b = np.polyfit(x_vals, y_vals, 1)
            
            x_trend = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_trend = m * x_trend + b
            
            ax.plot(x_trend, y_trend, color="black", linestyle="--", linewidth=2, label="Trend Line")

        ax.set_title("Contamination Score vs Pilot Accuracy (Phase 1 Triage)")
        ax.set_ylabel("Pilot Correct (Jittered)")
        ax.set_xlabel("Contamination Score")
        ax.set_yticks([0, 1])
        
    else:
        # No pilot accuracy data, so plot a stacked histogram
        hist_data = []
        labels = []
        for family in families:
            subset = df[df["problem_family"] == family]["contamination_score"].dropna().values
            if len(subset) > 0:
                hist_data.append(subset)
                labels.append(family)
                
        if hist_data:
            ax.hist(hist_data, bins=20, stacked=True, label=labels, alpha=0.8)
            
        ax.set_title("Contamination Score Distribution (Phase 1 Triage)")
        ax.set_ylabel("Count")
        ax.set_xlabel("Contamination Score")

    # Common layout and legend configurations
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
