###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    centroids: Optional[np.ndarray] = None,
    title: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot clustered data in 2D using the first two features.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features) or None
    title : str or None

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if X.shape[1] < 2:
        raise ValueError("X must have at least 2 features for a 2D plot.")

    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels, cmap="tab10", alpha=0.9)

    if centroids is not None:
        # Get the colormap used for clusters
        cmap = plt.cm.get_cmap("tab10")
        # Plot each centroid with the color of its cluster
        for i in range(len(centroids)):
            ax.scatter(
                centroids[i, 0],
                centroids[i, 1],
                marker="h",
                s=200,
                linewidths=2,
                c=[i],                    # <- use same “c” as labels
                cmap=scatter.cmap,        # <- same colormap
                norm=scatter.norm,        # <- same normalisation
                edgecolor="black",
                label="Centroids" if i == 0 else "",
            )
        ax.legend()

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    if title:
        ax.set_title(title)

    fig.colorbar(scatter, ax=ax, label="Cluster label")
    fig.tight_layout()
    out_fig = fig
    plt.close(fig)
    return fig, ax


def plot_elbow(
    k_values: List[int],
    inertias: List[float],
    title: str = "Elbow Curve",
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot inertia vs k (elbow method).

    Parameters
    ----------
    k_values : list of int
    inertias : list of float
    title : str, default "Elbow Curve"

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    if len(k_values) != len(inertias):
        raise ValueError("k_values and inertias must have the same length.")

    fig, ax = plt.subplots()
    ax.plot(k_values, inertias, marker="o")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.set_title(title)
    ax.grid(True)
    fig.tight_layout()
    out_fig = fig
    plt.close(fig)
    return fig, ax