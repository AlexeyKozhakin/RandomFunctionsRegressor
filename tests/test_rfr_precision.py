import unittest

import numpy as np
from scipy.spatial.distance import cdist

from experiments.rfr_precision import kernel_matrix, predict


class PrecisionBackendTests(unittest.TestCase):
    def test_kernel_matches_original_including_zero_and_nondefault_constants(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(20, 4))
        supports = X[:5].copy()
        distances = cdist(X, supports, metric="sqeuclidean")
        safe = np.where(distances > 0, distances, 1)
        reference = np.where(distances == 0, 11, 3 * (distances * np.log(safe / 11) + 7))
        np.testing.assert_allclose(kernel_matrix(X, supports, 3, 7, 11), reference, atol=1e-12)
        np.testing.assert_allclose(kernel_matrix(X, supports, 3, 7, 11, True),
                                   reference - 21, atol=1e-12)

    def test_centered_prediction_matches_original_and_keeps_float32(self):
        rng = np.random.default_rng(9)
        X = rng.uniform(size=(31, 4))
        supports = rng.uniform(size=(8, 4))
        q = rng.normal(size=8) * .01
        intercept = .5 - 1e5 * q.sum()
        reference = kernel_matrix(X, supports, 1, 1e5, 1e5) @ q + intercept
        result64 = predict(X, supports, q, intercept, 1, 1e5, 1e5, np.float64, batch_size=7)
        np.testing.assert_allclose(result64, reference, atol=1e-10)
        result32 = predict(X, supports, q, intercept, 1, 1e5, 1e5, batch_size=7)
        self.assertEqual(result32.dtype, np.float32)
        np.testing.assert_allclose(result32, reference, atol=2e-6)


if __name__ == "__main__":
    unittest.main()
