###
## cluster_maker: demo for cluster analysis
## James Foadi - University of Bath
## November 2025
##
## This script produces clustering for a group of points in 2D,
## using k-means for k = 2, 3, 4, 5. The input file is the csv
## file 'demo_data.csv' in folder 'data/'.
###

from __future__ import annotations

import os
import sys
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path of this script: demo/cluster_plot.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Path of parent directory: EXAMMA52109/
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Add PROJECT_ROOT to Python path
sys.path.insert(0, PROJECT_ROOT)

from cluster_maker import run_clustering

OUTPUT_DIR = "demo_output"

# if there aren't the correct number of arguments (2)
def main(args: List[str]) -> None:
    if len(args) != 1:
        print("Usage: python clustering_demo.py <input_csv>")
        sys.exit(1)

# Handles FileNotFoundError if the input file does not exist
    input_path = args[0]
    if not os.path.exists(input_path):
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    print("=== cluster_maker demo: clustering with k = 2, 3, 4, 5 ===\n")
    print("This demo loads a CSV dataset, selects the first two numeric features,")
    print("and runs k-means clustering for k = 2, 3, 4, and 5.")
    print("For each value of k, it:")
    print("  • fits a clustering model,")
    print("  • computes inertia and silhouette scores, and")
    print("  • saves a 2D cluster plot and the metrics to the demo_output folder.\n")
    # added description of demo fucntionality to improve user understanding.
    
    # The CSV is assumed to have two or more data columns
    df = pd.read_csv(input_path)
    
    if df.empty:
        print("Error: The input CSV contains no data.")
        sys.exit(1) # added check for empty dataframe
    
    # checks the number of numeric columns as at least 2 are needed for 2D plotting    
    numeric_cols = [
        col for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_cols) < 2:
        print("Error: The input CSV must have at least two numeric columns.")
        sys.exit(1)
    feature_cols = numeric_cols[:2]  # Use the first two numeric columns

    # For naming outputs
    base = os.path.splitext(os.path.basename(input_path))[0]

    # Main job: run clustering for k = 2, 3, 4, 5
    metrics_summary = []

    for k in (2, 3, 4, 5):
        print(f"\n=== Running k-means with k = {k} ===")

        result = run_clustering(  # the main clustering function in interface
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="kmeans",
            k = k , 
            # Changed to k; the demo incorrectly used min(k,3), so k=4 and k=5 were both run
            # as k=3. Using k ensures that each iterations runs k-means for that k.
            standardise=True,
            output_path=os.path.join(OUTPUT_DIR, f"{base}_clustered_k{k}.csv"),
            random_state=42,
            compute_elbow=False,  # no elbow diagram
        )

        # Save cluster plot
        plot_path = os.path.join(OUTPUT_DIR, f"{base}_k{k}.png")
        result["fig_cluster"].savefig(plot_path, dpi=150)
        plt.close(result["fig_cluster"])

        # Collect metrics
        metrics = {"k": k}
        metrics.update(result.get("metrics", {}))
        metrics_summary.append(metrics)

        print("Metrics:")
        for key, value in result.get("metrics", {}).items():
            print(f"  {key}: {value}")

    # Summarise metrics across k
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_csv = os.path.join(OUTPUT_DIR, f"{base}_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    # Plot some statistics (no elbow: avoid inertia vs k)
    if "silhouette" in metrics_df.columns: 
        # changed to silhouette as the silhouette score is saved as "silhouette" in metrics
        plt.figure()
        plt.bar(metrics_df["k"], metrics_df["silhouette"]) # changed to silhouette
        plt.xlabel("k")
        plt.ylabel("Silhouette score")
        plt.title("Silhouette score for different k")
        stats_path = os.path.join(OUTPUT_DIR, f"{base}_silhouette.png")
        plt.savefig(stats_path, dpi=150)
        plt.close()

    print("\nDemo completed.")
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])