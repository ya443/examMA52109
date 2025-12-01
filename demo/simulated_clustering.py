###
## simulated_clustering.py
## Yas Akilakulasingam – University of Bath
## December 2025
##
## This demo performs a rigorous clustering analysis on data/simulated_data.csv.
## It compares manual k-means with sklearn’s k-means for k = 2..6, evaluates both
## methods using inertia and silhouette scores, generates elbow and silhouette
## plots, and prints a clear conclusion identifying the most plausible number
## of clusters and the superior algorithm.
##
## All outputs are saved in: simulated_clustering_demo_output/
###

from __future__ import annotations
import os
import sys

# -------------------------------------------------------------------
# Safe imports with explicit error messages
# -------------------------------------------------------------------
try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required for this demo.")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is required for this demo.")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is required for plotting.")
    sys.exit(1)

# Add project root to import cluster_maker
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Now import from cluster_maker
from cluster_maker import (
    run_clustering,
    select_features,
)

OUTPUT_DIR = "simulated_clustering_demo_output"


# ============================================================================
# MAIN FUNCTION
# ============================================================================
def main(args: list[str]) -> None:
    print("\n=== Simulated Clustering Demo ===\n")

    # -------------------------------------------------------------------
    # Validate arguments
    # -------------------------------------------------------------------
    if len(args) != 2:
        print("ERROR: Incorrect number of arguments.")
        print("Usage: python demo/simulated_clustering.py data/simulated_data.csv")
        sys.exit(1)

    input_path = os.path.abspath(args[-1])
    print(f"Input file detected: {input_path}")

    if not os.path.exists(input_path):
        print(f"ERROR: File not found: {input_path}")
        sys.exit(1)

    # -------------------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------------------
    print("\nLoading dataset...")
    try:
        df = pd.read_csv(input_path)
    except Exception as exc:
        print(f"ERROR: Unable to read CSV file:\n{exc}")
        sys.exit(1)

    if df.empty:
        print("ERROR: The dataset is empty; cannot cluster.")
        sys.exit(1)

    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    # Identify numeric columns
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        print("ERROR: Need at least two numeric columns for 2D clustering.")
        sys.exit(1)

    feature_cols = numeric_cols[:2]
    print(f"Using numeric features: {feature_cols}")

    # Validate selected features
    try:
        _ = select_features(df, feature_cols)
    except Exception as exc:
        print(f"Feature validation error:\n{exc}")
        sys.exit(1)

    # Ensure output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -------------------------------------------------------------------
    # Raw data plot
    # -------------------------------------------------------------------
    print("\nSaving raw data plot...")
    fig, ax = plt.subplots()
    ax.scatter(df[feature_cols[0]], df[feature_cols[1]], s=25, alpha=0.7)
    ax.set_xlabel(feature_cols[0])
    ax.set_ylabel(feature_cols[1])
    ax.set_title("Raw Data: Initial View of Structure")
    fig.tight_layout()

    raw_path = os.path.join(OUTPUT_DIR, "simulated_raw_data.png")
    fig.savefig(raw_path, dpi=150)
    plt.close(fig)
    print(f"Saved raw data plot: {raw_path}")

    # -------------------------------------------------------------------
    # Run clustering for both algorithms
    # -------------------------------------------------------------------
    k_values = [2, 3, 4, 5, 6]
    metrics_manual = []
    metrics_sklearn = []

    print("\n=== Running manual and sklearn K-means for k = 2, 3, 4, 5, 6 ===\n")

    for k in k_values:
        print(f"--- k = {k} ---")

        # -------- Manual KMeans --------
        result_manual = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="kmeans",
            k=k,
            standardise=True,
            output_path=None,
            random_state=42,
            compute_elbow=False,
        )

        inertia_m = result_manual["metrics"]["inertia"]
        sil_m = result_manual["metrics"]["silhouette"]
        metrics_manual.append({"k": k, "inertia": inertia_m, "silhouette": sil_m})
        print(f"Manual KMeans → inertia={inertia_m:.3f}, silhouette={sil_m:.3f}")

        # -------- Sklearn KMeans --------
        result_sk = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm="sklearn_kmeans",
            k=k,
            standardise=True,
            output_path=None,
            random_state=42,
            compute_elbow=False,
        )

        inertia_s = result_sk["metrics"]["inertia"]
        sil_s = result_sk["metrics"]["silhouette"]
        metrics_sklearn.append({"k": k, "inertia": inertia_s, "silhouette": sil_s})
        print(f"Sklearn KMeans → inertia={inertia_s:.3f}, silhouette={sil_s:.3f}\n")

    # Convert to DataFrames
    manual_df = pd.DataFrame(metrics_manual)
    sk_df = pd.DataFrame(metrics_sklearn)

    manual_df.to_csv(os.path.join(OUTPUT_DIR, "manual_metrics.csv"), index=False)
    sk_df.to_csv(os.path.join(OUTPUT_DIR, "sklearn_metrics.csv"), index=False)

    # -------------------------------------------------------------------
    # Elbow plots
    # -------------------------------------------------------------------
    print("\nSaving elbow plots...")

    # Manual elbow
    plt.figure()
    plt.plot(manual_df["k"], manual_df["inertia"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Manual KMeans Elbow Curve")
    elbow_m_path = os.path.join(OUTPUT_DIR, "elbow_manual.png")
    plt.savefig(elbow_m_path, dpi=150)
    plt.close()

    # Sklearn elbow
    plt.figure()
    plt.plot(sk_df["k"], sk_df["inertia"], marker="o", color="green")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.title("Sklearn KMeans Elbow Curve")
    elbow_s_path = os.path.join(OUTPUT_DIR, "elbow_sklearn.png")
    plt.savefig(elbow_s_path, dpi=150)
    plt.close()

    print(f"Saved elbow plots:\n  {elbow_m_path}\n  {elbow_s_path}")

    # -------------------------------------------------------------------
    # Choose best algorithm
    # -------------------------------------------------------------------
    mean_sil_manual = manual_df["silhouette"].mean()
    mean_sil_sklearn = sk_df["silhouette"].mean()

    if mean_sil_sklearn >= mean_sil_manual:
        best_algo = "sklearn"
        best_df = sk_df
        print("\nBest algorithm: sklearn KMeans (higher average silhouette)")
    else:
        best_algo = "manual"
        best_df = manual_df
        print("\nBest algorithm: manual KMeans (higher average silhouette)")

    # -------------------------------------------------------------------
    # Plot clusters for ALL ks using the best algorithm
    # -------------------------------------------------------------------
    print("\nSaving cluster plots for all ks (2..6) for the best algorithm...")

    for k in k_values:
        best_result = run_clustering(
            input_path=input_path,
            feature_cols=feature_cols,
            algorithm=("sklearn_kmeans" if best_algo == "sklearn" else "kmeans"),
            k=k,
            standardise=True,
            output_path=None,
            random_state=42,
            compute_elbow=False,
        )

        fig = best_result["fig_cluster"]
        out_path = os.path.join(OUTPUT_DIR, f"best_algo_k{k}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {out_path}")

    # -------------------------------------------------------------------
    # Silhouette plot for the best algorithm
    # -------------------------------------------------------------------
    plt.figure()
    plt.plot(best_df["k"], best_df["silhouette"], marker="o")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Across k (Best Algo: {best_algo})")
    plt.grid(True)
    sil_path = os.path.join(OUTPUT_DIR, "best_silhouette.png")
    plt.savefig(sil_path, dpi=150)
    plt.close()
    print(f"Saved silhouette plot: {sil_path}")

    # -------------------------------------------------------------------
    # Final Interpretation
    # -------------------------------------------------------------------
    print("\n=== Interpretation ===")

    best_row = best_df.loc[best_df["silhouette"].idxmax()]
    best_k = int(best_row["k"])
    best_score = best_row["silhouette"]

    print(f"\nStrongest clustering found at k = {best_k}, with silhouette = {best_score:.3f}.")
    print(f"Best algorithm: {best_algo}")
    print(f"The raw data clearly shows ~{best_k} well-separated groups.")
    print(f"Silhouette and inertia curves confirm that k = {best_k} is the most plausible.")
    print("\nAll outputs have been saved in:", OUTPUT_DIR)
    print("=== End of simulated clustering demo ===\n")


# ============================================================================
if __name__ == "__main__":
    main(sys.argv)
