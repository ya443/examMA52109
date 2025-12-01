###
## cluster_maker
## Yas Akilakulasingam - University of Bath
## Decemeber 2025
###

####
### Agglomerative clustering module for cluster_maker.
### Provides a robust hierarchical clustering method using sklearn.
####

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np

# Safe import of sklearn
try:
    from sklearn.cluster import AgglomerativeClustering
except Exception:
    AgglomerativeClustering = None


def agglomerative_clustering(
    X: np.ndarray,
    k: int,
    linkage: str = "average",
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform hierarchical agglomerative clustering on data X.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int
        Number of clusters.
    linkage : {"ward", "complete", "average", "single"}, default "average"
        Linkage strategy used to merge clusters.
    metric : str, default "euclidean"
        Distance metric. For linkage="ward", metric MUST be "euclidean".

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
        Pseudo-centroids computed as the mean of points per cluster.
    """

    # ------------------------------------------------------------------
    # Validate environment
    # ------------------------------------------------------------------
    if AgglomerativeClustering is None:
        raise ImportError("scikit-learn is required for agglomerative clustering.")

    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    if k <= 0:
        raise ValueError("k must be a positive integer.")

    if k > X.shape[0]:
        raise ValueError("k cannot exceed the number of samples.")

    # ------------------------------------------------------------------
    # Validate linkage/metric combination
    # ------------------------------------------------------------------
    valid_linkages = {"ward", "complete", "average", "single"}
    if linkage not in valid_linkages:
        raise ValueError(f"Invalid linkage '{linkage}'. Must be one of {valid_linkages}.")

    # Ward linkage ONLY works with Euclidean
    if linkage == "ward" and metric != "euclidean":
        raise ValueError("linkage='ward' requires metric='euclidean'.")

    # ------------------------------------------------------------------
    # Create model (supports sklearn ≥ 1.3; fallback to affinity for older versions)
    # ------------------------------------------------------------------
    try:
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
            metric=metric
        )
    except TypeError:
        # Older sklearn (<1.3)
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
            affinity=metric
        )

    # ------------------------------------------------------------------
    # Fit model
    # ------------------------------------------------------------------
    labels = model.fit_predict(X)

    # ------------------------------------------------------------------
    # Compute pseudo-centroids (keeps package API consistent)
    # ------------------------------------------------------------------
    centroids = np.zeros((k, X.shape[1]))

    for cluster_id in range(k):
        mask = labels == cluster_id
        if not np.any(mask):
            # Instead of crashing, choose a random point → robust default
            idx = np.random.randint(0, X.shape[0])
            centroids[cluster_id] = X[idx]
        else:
            centroids[cluster_id] = X[mask].mean(axis=0)

    return labels, centroids