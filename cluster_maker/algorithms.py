###
## cluster_maker
## James Foadi - University of Bath
## November 2025
###

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from sklearn.cluster import KMeans


def init_centroids(
    X: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Initialise centroids by randomly sampling points from X without replacement.
    """
    
    if not isinstance(k, int):
        raise TypeError("k must be an integer.")
    # added an additional type check to ensure k is an integer.
    
    if k <= 0:
        raise ValueError("k must be a positive integer.")
    n_samples = X.shape[0]

    if k > n_samples:
        raise ValueError("k cannot be larger than the number of samples.")
    # Corrected to ensure that k is compared to the number of 
    # samples, n_samples, as defined (could alternatively have been X.shape[0])
    # previous version compared to X.shape[1], which is the number of features.
    
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    # added type check to ensure X is a NumPy array.


    rng = np.random.RandomState(random_state)
    indices = rng.choice(n_samples, size=k, replace=False)
    # changed to size=k from size=k+1
    # indices is an array of k unique indices from 0 to n_samples-1, it indicates 
    # which sample data point will be used as randomly selected inital centroids.
    # We want to choose k inital centroids from the n_samples data points, not k+1.
    return X[indices]


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """
    Assign each sample to the nearest centroid (Euclidean distance).
    """
    # X: (n_samples, n_features)
    # centroids: (k, n_features)
    
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    # added type check to ensure X is a NumPy array.
    
    if centroids.ndim != 2:
        raise ValueError("centroids must be a 2D array of shape (k, n_features).")
    # added check to ensure centroids is the correct shape.
    
    if centroids.shape[1] != X.shape[1]:
        raise ValueError("centroids and X must have the same number of features.")
    # added check to ensure centroids and X have the same number of features.
    
    # Broadcast to compute distances
    diff = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)  # (n_samples, k)
    # Changed to use np.linalg.norm across axis=2, as this is the axis representing the features
    # diff.shape is (n_samples, k, n_features), so we want to compute the norm 
    # across the features axis (axis=2)
    labels = np.argmin(distances, axis=1)
    # changed to argmin, not argmax, as we want to assign each point 
    # to the nearest centroid, not the farthest.
    return labels


def update_centroids(
    X: np.ndarray,
    labels: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Update centroids by taking the mean of points in each cluster.
    If a cluster becomes empty, re-initialise its centroid randomly from X.
    """
    
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")
    # added type check to ensure X is a NumPy array.
    
    if k > np.unique(X, axis=0).shape[0]:
        raise ValueError("k cannot exceed the number of unique data points.")
    # added check to ensure k does not exceed the number of unique data points in X.
    
    n_features = X.shape[1]
    new_centroids = np.zeros((k, n_features), dtype=float)
    rng = np.random.RandomState(random_state)

    for cluster_id in range(k):
        mask = labels == cluster_id
        if not np.any(mask):
            # Empty cluster: re-initialise randomly
            idx = rng.randint(0, X.shape[0])
            new_centroids[cluster_id] = X[idx]
        else:
            new_centroids[cluster_id] = X[mask].mean(axis=0)

    return new_centroids


def kmeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple manual K-means implementation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    k : int
        Number of clusters.
    max_iter : int, default 300
        Maximum number of iterations.
    tol : float, default 1e-4
        Convergence tolerance on centroid movement.
    random_state : int or None

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    centroids = init_centroids(X, k, random_state=random_state)
    for _ in range(max_iter):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k, random_state=random_state)
        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    labels = assign_clusters(X, centroids)
    return labels, centroids


def sklearn_kmeans(
    X: np.ndarray,
    k: int,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Thin wrapper around scikit-learn's KMeans.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
    centroids : ndarray of shape (k, n_features)
    """
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a NumPy array.")

    model = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=10,
    )
    model.fit(X)
    labels = model.labels_
    centroids = model.cluster_centers_
    return labels, centroids