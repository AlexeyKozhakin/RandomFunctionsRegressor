"""Experimental matrix-distance backend; the public RFR estimator is unchanged."""
import numpy as np


def kernel_matrix(X, supports, C, n, t, centered=False):
    """Evaluate the original Kf, optionally subtracting the constant C*n.

    Squared distances use matrix multiplication in the input dtype. Centering
    is an algebraic rearrangement, including the original special value at zero.
    """
    dtype = X.dtype
    two = dtype.type(2)
    norms = (X * X).sum(axis=1)[:, None] + (supports * supports).sum(axis=1)[None, :]
    distances = norms.copy()
    distances -= two * (X @ supports.T)
    np.maximum(distances, dtype.type(0), out=distances)
    # GEMM cancellation can make equal points slightly positive. When the
    # kernel is discontinuous at zero, resolve nearby pairs by subtraction.
    if t != C * n:
        rows, cols = np.nonzero(distances <= 8 * np.finfo(dtype).eps * norms)
        delta = X[rows] - supports[cols]
        distances[rows, cols] = (delta * delta).sum(axis=1)
    positive = distances > 0
    safe = np.where(positive, distances, dtype.type(1))
    values = dtype.type(C) * (distances * np.log(safe / dtype.type(t)))
    if centered:
        values[~positive] = dtype.type(t - C * n)
    else:
        values += dtype.type(C * n)
        values[~positive] = dtype.type(t)
    return values


def predict(X, supports, coefficients, intercept, C, n, t,
            dtype=np.float32, centered=True, batch_size=1024):
    dtype = np.dtype(dtype)
    X = np.asarray(X, dtype=dtype)
    supports = np.asarray(supports, dtype=dtype)
    # Form the shifted intercept before rounding coefficients to float32.
    bias = np.asarray(intercept, dtype=np.float64)
    if centered:
        bias = bias + C * n * np.asarray(coefficients, dtype=np.float64).sum(axis=0)
    bias = np.asarray(bias, dtype=dtype)
    coefficients = np.asarray(coefficients, dtype=dtype)
    result = np.empty((len(X),) + coefficients.shape[1:], dtype=dtype)
    for start in range(0, len(X), batch_size):
        stop = min(start + batch_size, len(X))
        design = kernel_matrix(X[start:stop], supports, C, n, t, centered=centered)
        result[start:stop] = design @ coefficients + bias
    return result
