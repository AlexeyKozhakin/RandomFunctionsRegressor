import unittest

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, Ridge

from RandomFunctionsRegressor import RandomFunctionsRegressor


class LHSRegressorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(123)
        self.X = rng.uniform(-1, 1, size=(180, 3))
        self.y = np.sin(self.X[:, 0]) + self.X[:, 1] ** 2 - self.X[:, 2]

    def model(self, **kwargs):
        parameters = dict(n_support=12, random_state=42, n=20, t=20,
                          method_reg="ridge", alpha=0.2, batch_size=17)
        parameters.update(kwargs)
        return RandomFunctionsRegressor(**parameters)

    def explicit_design(self, model, X):
        squared = ((X[:, None, :] - model.support_points_[None, :, :]) ** 2).sum(axis=2)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(squared == 0, model.t,
                            model.C * (squared * np.log(squared / model.t) + model.n))

    def test_lhs_stratifies_training_bounds_and_constant_coordinates(self):
        X = np.column_stack([self.X, np.full(len(self.X), 7.0)])
        model = self.model().fit(X, self.y)
        self.assertEqual(model.support_points_.shape, (12, 4))
        np.testing.assert_array_equal(model.support_points_[:, 3], 7.0)
        for j in range(3):
            unit = ((model.support_points_[:, j] - X[:, j].min()) /
                    (X[:, j].max() - X[:, j].min()))
            np.testing.assert_array_equal(np.sort(np.floor(unit * 12)), np.arange(12))

    def test_coefficient_count_independent_of_training_rows(self):
        for size in (30, 180):
            model = self.model().fit(self.X[:size], self.y[:size])
            self.assertEqual(model.q.shape, (12,))
            self.assertEqual(model.coef_.shape, (12,))
            self.assertEqual(model.training_size_, size)
            self.assertFalse(hasattr(model, "X_train"))
            self.assertFalse(hasattr(model, "y_train"))

    def test_original_kf_and_ridge_predictions_match_explicit_reference(self):
        model = self.model(C=2, n=10, t=5).fit(self.X, self.y)
        design = self.explicit_design(model, self.X)
        reference = Ridge(alpha=0.2, fit_intercept=True, solver="svd").fit(design, self.y)
        np.testing.assert_allclose(model.predict(self.X), reference.predict(design), atol=1e-10)
        np.testing.assert_array_equal(model.Kf(np.array([0.0, 1.0])),
                                      [5.0, 2 * (np.log(1 / 5) + 10)])

    def test_all_unregularized_solvers_match_linear_regression(self):
        for method in ("mnk", "svd", "lstsq"):
            with self.subTest(method=method):
                model = self.model(method_reg=method).fit(self.X, self.y)
                reference = LinearRegression().fit(self.explicit_design(model, self.X), self.y)
                np.testing.assert_allclose(model.predict(self.X),
                                           reference.predict(self.explicit_design(model, self.X)),
                                           atol=1e-9)

    def test_sigma_is_coefficient_penalty_in_lhs_mode(self):
        for method in ("mnk", "svd", "lstsq"):
            model = self.model(method_reg=method, sig=0.4).fit(self.X, self.y)
            reference = Ridge(alpha=0.4 ** 2, solver="svd").fit(
                self.explicit_design(model, self.X), self.y)
            np.testing.assert_allclose(model.predict(self.X),
                                       reference.predict(self.explicit_design(model, self.X)),
                                       atol=1e-9)

    def test_seed_batching_and_prediction_bounds_do_not_change_supports(self):
        first = self.model(batch_size=1).fit(self.X, self.y)
        second = self.model(batch_size=1000).fit(self.X, self.y)
        np.testing.assert_array_equal(first.support_points_, second.support_points_)
        outside = self.X[:10] * 3
        supports = first.support_points_.copy()
        np.testing.assert_allclose(first.predict(outside), second.predict(outside), atol=1e-9)
        np.testing.assert_array_equal(first.support_points_, supports)

    def test_multioutput_and_no_intercept(self):
        y = np.column_stack([self.y, self.y ** 2])
        model = self.model(fit_intercept=False).fit(self.X, y)
        reference = Ridge(alpha=0.2, fit_intercept=False, solver="svd").fit(
            self.explicit_design(model, self.X), y)
        self.assertEqual(model.q.shape, (12, 2))
        self.assertEqual(model.predict(self.X).shape, y.shape)
        np.testing.assert_allclose(model.predict(self.X),
                                   reference.predict(self.explicit_design(model, self.X)), atol=1e-9)

    def test_all_constant_input_and_repeated_fit(self):
        model = self.model().fit(self.X, self.y)
        model.fit(np.ones((10, 2)), np.arange(10.0))
        np.testing.assert_array_equal(model.support_points_, np.ones((12, 2)))
        np.testing.assert_allclose(model.predict(np.ones((4, 2))), 4.5)

    def test_invalid_input_and_not_fitted(self):
        with self.assertRaises(NotFittedError):
            self.model().predict(self.X)
        for kwargs in ({"n_support": 0}, {"n_support": 2.5}, {"t": 0},
                       {"batch_size": 0}, {"alpha": -1}, {"method_reg": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.model(**kwargs).fit(self.X, self.y)
        X = self.X.copy()
        X[0, 0] = np.nan
        with self.assertRaises(ValueError):
            self.model().fit(X, self.y)
        model = self.model().fit(self.X, self.y)
        with self.assertRaises(ValueError):
            model.predict(np.ones((5, 2)))

    def test_sklearn_clone_and_fit_return_self(self):
        model = clone(self.model())
        self.assertIs(model.fit(self.X, self.y), model)
        self.assertTrue(np.isfinite(model.score(self.X, self.y)))

    def test_legacy_mnk_predictions_preserved(self):
        X, y = self.X[:10], self.y[:10]
        model = RandomFunctionsRegressor(sig=0.2)
        squared = ((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2)
        matrix = model.Kf(squared)
        np.fill_diagonal(matrix, matrix.diagonal() + 0.2 ** 2)
        coefficients = np.linalg.lstsq(matrix, y, rcond=None)[0]
        expected = model.Kf(squared) @ coefficients
        model.fit(X, y)
        np.testing.assert_allclose(model.predict(X), expected, atol=1e-8)


if __name__ == "__main__":
    unittest.main()
