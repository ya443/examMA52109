###
# test_preprocessing.py
# Yas Akilakulasingam - University of Bath
# December 2025


# These tests verify important and meaningful behaviours in preprocessing.py.
# Each test focuses on a real failure mode that could break the correctness
# or reliability of the clustering workflow.

import unittest
import numpy as np
import pandas as pd

from cluster_maker.preprocessing import select_features, standardise_features


class TestPreprocessing(unittest.TestCase):
        
    # ------------------------------------------------------------------
    # Test 1: select_features must preserve the order of requested features.
    #
    # WHY THIS MATTERS:
    # K-means relies on feature order. If preprocessing reorders columns,
    # users pass one set of features but clustering uses a different order,
    # producing incorrect results.
    # ------------------------------------------------------------------
    def test_select_features_preserves_order(self):
        df = pd.DataFrame({
            "a": [1,2,3],
            "b": [4,5,6],
            "c": [7,8,9],
        })

        selected = select_features(df, ["c", "a"])

        self.assertEqual(list(selected.columns), ["c", "a"])
        
    # ------------------------------------------------------------------
    # Test 2: standardise_features must handle constant columns correctly.
    #
    # WHY THIS MATTERS:
    # A feature with zero variance can cause divide-by-zero issues.
    # Correct behaviour: the column becomes all zeros after standardisation.
    # ------------------------------------------------------------------
    def test_standardise_features_with_constant_column(self):
        X = np.array([
            [10.0, 1.0],
            [10.0, 2.0],
            [10.0, 3.0],
        ])

        X_std = standardise_features(X)
        # First column has zero variance → should become all zeros
        self.assertTrue(np.allclose(X_std[:,0], 0.0))
        
    # ------------------------------------------------------------------
    # Test 3: standardise_features should not distort already-standardised data.
    #
    # WHY THIS MATTERS:
    # If standardisation is applied twice (a common user mistake), the
    # transformation should not drift the data away from mean 0 and std 1.
    # This ensures numerical stability in longer pipelines.
    # ------------------------------------------------------------------
    def test_standardise_features_is_idempotent(self):
        X = np.array([
            [-1.0, 0.5],
            [ 0.0, -1.0],
            [ 1.0, 0.5]
        ])

        # First standardisation
        X1 = standardise_features(X)

        # Second standardisation should not significantly change the dataset
        X2 = standardise_features(X1)

        self.assertTrue(np.allclose(X1, X2, atol=1e-8))


if __name__ == "__main__":
    unittest.main()
