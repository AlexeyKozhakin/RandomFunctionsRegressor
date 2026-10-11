"""LHS support-point regression using the estimator's existing Kf function."""

from numbers import Integral, Real

import numpy as np
from scipy.linalg import lstsq
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class _LHSBasisMixin(RegressorMixin, BaseEstimator):
    def _validate_lhs_parameters(self):
        for name in ("n_support", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("C", "n", "t", "sig", "alpha"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.t <= 0:
            raise ValueError("t must be positive for Kf")
        if self.sig < 0 or self.alpha < 0:
            raise ValueError("sig and alpha must be nonnegative")
        if not isinstance(self.fit_intercept, (bool, np.bool_)):
            raise ValueError("fit_intercept must be boolean")
        if self.method_reg not in ("mnk", "ridge", "svd", "lstsq"):
            raise ValueError("method_reg must be mnk, ridge, svd, or lstsq")

    def _kernel_features(self, X):
        # No n_samples x n_support x n_features intermediate is constructed.
        distances = cdist(X, self.support_points_, metric="sqeuclidean")
        values = self.Kf(distances)
        if not np.isfinite(values).all():
            raise ValueError("Kf produced nonfinite values; check data scale and parameters")
        return values

    def transform(self, X):
        """Return the dense Kf design matrix, shape (rows, n_support).

        Distances are evaluated in batches, but the returned matrix is dense.
        Input features remain in their original coordinates; no scaling or
        imputation is performed by this estimator.
        """
        check_is_fitted(self, "support_points_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        design = np.empty((len(X), self.n_support_), dtype=np.float64)
        for start in range(0, len(X), self.batch_size):
            stop = min(start + self.batch_size, len(X))
            design[start:stop] = self._kernel_features(X[start:stop])
        return design

    def _fit_lhs(self, X, y):
        self._validate_lhs_parameters()
        X, y = check_X_y(X, y, dtype=np.float64, multi_output=True, y_numeric=True)
        y = np.asarray(y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        self.n_support_ = int(self.n_support)
        self.training_size_ = len(X)
        self.feature_min_ = X.min(axis=0)
        self.feature_max_ = X.max(axis=0)
        widths = self.feature_max_ - self.feature_min_
        if not np.isfinite(widths).all():
            raise ValueError("Feature ranges overflow; rescale the input")
        varying = widths > 0
        self.support_points_ = np.broadcast_to(
            self.feature_min_, (self.n_support_, self.n_features_in_)
        ).copy()
        if varying.any():
            sampler = qmc.LatinHypercube(d=int(varying.sum()), seed=self.random_state)
            unit_points = sampler.random(n=self.n_support_)
            self.support_points_[:, varying] = (
                self.feature_min_[varying] + unit_points * widths[varying]
            )
        # Constant coordinates stay constant; LHS applies to varying coordinates.
        design = self.transform(X)
        if self.method_reg == "ridge":
            regressor = Ridge(alpha=self.alpha + self.sig ** 2,
                              fit_intercept=self.fit_intercept, solver="svd")
            regressor.fit(design, y)
            self.q = regressor.coef_.T.copy()
            self.intercept_ = np.asarray(regressor.intercept_).copy()
        else:
            x_mean = design.mean(axis=0) if self.fit_intercept else np.zeros(self.n_support_)
            y_mean = y.mean(axis=0) if self.fit_intercept else np.zeros(y.shape[1:] or ())
            if self.fit_intercept:
                design -= x_mean
            rhs = y - y_mean
            # Rectangular regression has no meaningful Gram-matrix diagonal
            # noise term. Here sig**2 is an L2 penalty on the coefficients.
            if self.sig > 0:
                design = np.vstack((design, self.sig * np.eye(self.n_support_)))
                zeros = np.zeros((self.n_support_,) + y.shape[1:])
                rhs = np.concatenate((rhs, zeros), axis=0)
            if self.method_reg == "mnk":
                self.q = np.linalg.lstsq(design, rhs, rcond=None)[0]
            else:
                self.q = lstsq(design, rhs, lapack_driver="gelsd")[0]
            self.intercept_ = y_mean - x_mean @ self.q
        self.coef_ = self.q.T.copy()
        # The LHS predictor stores only support points and fitted coefficients.
        for name in ("X_train", "y_train"):
            if hasattr(self, name):
                delattr(self, name)
        return self

    def _predict_lhs(self, X):
        check_is_fitted(self, ("support_points_", "coef_", "intercept_"))
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")
        prediction = np.empty((len(X),) + self.q.shape[1:], dtype=np.float64)
        for start in range(0, len(X), self.batch_size):
            stop = min(start + self.batch_size, len(X))
            prediction[start:stop] = self._kernel_features(X[start:stop]) @ self.q + self.intercept_
        return prediction
