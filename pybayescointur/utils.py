"""
Numerical helpers shared across :mod:`pybayescointur`.

The Bayesian unit-root / cointegration estimators in this package routinely
have to integrate posterior densities whose magnitude spans hundreds of orders
of magnitude (e.g. ``eta(rho) ** (n * T / 2)``).  Doing that in the natural
scale overflows immediately, so everything here is built around *log-space*
integration.

Author: Dr Merwan Roudane (merwanroudane920@gmail.com)
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logsumexp

__all__ = [
    "log_trapz",
    "log_integrate_grid",
    "logdet",
    "ensure_panel",
    "lag_matrix",
    "difference",
    "ols",
    "dickey_fuller_a",
]


# --------------------------------------------------------------------------- #
# Log-space integration                                                       #
# --------------------------------------------------------------------------- #
def log_trapz(log_y: ArrayLike, x: ArrayLike) -> float:
    r"""Return ``log(\int y dx)`` from the log-values ``log_y`` on the grid ``x``.

    Uses the composite trapezoidal rule evaluated entirely in log-space so that
    integrands which under/overflow in the natural scale are handled safely.

    Parameters
    ----------
    log_y : array_like
        ``log f(x_i)`` evaluated on the grid.  ``-inf`` entries (f == 0) are
        allowed.
    x : array_like
        Strictly increasing grid of abscissae, same length as ``log_y``.

    Returns
    -------
    float
        ``log`` of the approximated integral.  Returns ``-inf`` if the
        integrand is zero everywhere.
    """
    log_y = np.asarray(log_y, dtype=float)
    x = np.asarray(x, dtype=float)
    if log_y.shape != x.shape:
        raise ValueError("log_y and x must have the same shape")
    if log_y.size < 2:
        raise ValueError("need at least two grid points")

    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("x must be strictly increasing")

    # Each trapezoid:  0.5 * dx_i * (y_i + y_{i+1})
    seg = np.log(0.5 * dx) + np.logaddexp(log_y[:-1], log_y[1:])
    return float(logsumexp(seg))


def log_integrate_grid(log_func, lower: float, upper: float, n: int = 2000) -> float:
    """``log`` of ``\\int_lower^upper exp(log_func(t)) dt`` on a uniform grid.

    Parameters
    ----------
    log_func : callable
        Vectorised function returning ``log f(t)`` for an array ``t``.
    lower, upper : float
        Integration limits.
    n : int, default 2000
        Number of grid points.
    """
    x = np.linspace(lower, upper, n)
    log_y = np.asarray(log_func(x), dtype=float)
    return log_trapz(log_y, x)


# --------------------------------------------------------------------------- #
# Linear-algebra helpers                                                       #
# --------------------------------------------------------------------------- #
def logdet(mat: ArrayLike) -> float:
    """Numerically stable ``log|det(mat)|`` via the slogdet routine."""
    sign, val = np.linalg.slogdet(np.asarray(mat, dtype=float))
    if sign <= 0:
        # Fall back to eigenvalues for (near) singular / indefinite matrices.
        w = np.linalg.eigvalsh((np.asarray(mat) + np.asarray(mat).T) / 2.0)
        w = np.clip(w, 1e-12, None)
        return float(np.sum(np.log(w)))
    return float(val)


# --------------------------------------------------------------------------- #
# Panel-data plumbing                                                          #
# --------------------------------------------------------------------------- #
def ensure_panel(y: ArrayLike) -> np.ndarray:
    """Coerce input into a ``(T, n)`` float array (T periods, n cross units).

    Accepts a 1-D series (treated as ``n = 1``), a 2-D ``(T, n)`` array or a
    pandas DataFrame / Series.
    """
    if hasattr(y, "values"):  # pandas
        y = y.values
    arr = np.asarray(y, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("panel must be 1-D or 2-D (T x n)")
    return arr


def lag_matrix(y: np.ndarray, k: int = 1) -> np.ndarray:
    """Return ``y`` lagged by ``k`` periods with NaN padding at the top."""
    out = np.full_like(y, np.nan)
    if k < y.shape[0]:
        out[k:] = y[:-k]
    return out


def difference(y: np.ndarray, k: int = 1) -> np.ndarray:
    """First (or k-th) difference with NaN padding at the top."""
    return y - lag_matrix(y, k)


def ols(X: np.ndarray, y: np.ndarray):
    """Ordinary least squares.

    Returns
    -------
    beta : ndarray
        Coefficient vector.
    se : ndarray
        Standard errors.
    sigma2 : float
        Residual variance (with ``n - k`` correction).
    resid : ndarray
        Residuals.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    dof = max(X.shape[0] - X.shape[1], 1)
    sigma2 = float(resid @ resid / dof)
    se = np.sqrt(np.maximum(np.diag(XtX_inv) * sigma2, 0.0))
    return beta, se, sigma2, resid


def dickey_fuller_a(y: ArrayLike, trend: str = "ct") -> float:
    r"""Operational lower bound ``a`` of the stationary region.

    Schotman & Van Dijk (1991) suggest setting the lower integration limit
    ``a`` of the autoregressive parameter from a Dickey-Fuller style point
    estimate.  Here we pool the panel, run the ADF(0) regression and map the
    estimated ``rho`` to a conservative lower bound ``a = max(-1+eps,
    2 * rho_hat - 1)`` (a symmetric reflection that keeps the true value inside
    ``(a, 1)``).

    Parameters
    ----------
    y : array_like
        Panel ``(T, n)``.
    trend : {"n", "c", "ct"}
        Deterministic part: none, constant, constant + trend.
    """
    Y = ensure_panel(y)
    T, n = Y.shape
    rows_y, rows_X = [], []
    for i in range(n):
        yi = Y[:, i]
        ylag = yi[:-1]
        ycur = yi[1:]
        t = np.arange(1, T)
        cols = [ylag]
        if trend in ("c", "ct"):
            cols.append(np.ones_like(ylag))
        if trend == "ct":
            cols.append(t.astype(float))
        rows_X.append(np.column_stack(cols))
        rows_y.append(ycur)
    X = np.vstack(rows_X)
    yv = np.concatenate(rows_y)
    beta, *_ = ols(X, yv)
    rho_hat = float(np.clip(beta[0], -0.999, 0.999))
    a = max(-0.999, 2.0 * rho_hat - 1.0)
    return float(min(a, 0.0))
