r"""
Bayesian panel unit-root test via the Posterior Odds Ratio (POR).

Reference
---------
Kumar, J., Chaturvedi, A. & Afifa, U. (2016).
*Bayesian Unit Root Test for Panel Data.*
EERI Research Paper Series 14/2016.

Model
-----
For a panel ``{y_it ; i = 1..n ; t = 1..T}``::

    y_it   = mu_i + delta_i * t + u_it
    u_it   = rho * u_{i,t-1} + eps_it,      eps_it ~ N(0, 1/tau)

The unit-root null ``H0: rho = 1`` (difference stationary) is tested against
``H1: a < rho < 1`` (trend stationary).  Two specifications are supported:

* **Theorem 1** - linear time trend only.
* **Theorem 2** - linear time trend with ``k_i`` augmentation (lagged-difference)
  terms per unit.

The posterior odds ratio ``beta_01`` integrates the autoregressive coefficient
out of the posterior.  ``beta_01 < 1`` is evidence *against* the unit root
(the series is trend stationary); ``beta_01 > 1`` favours a unit root.

Author: Dr Merwan Roudane (merwanroudane920@gmail.com)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .utils import (
    ensure_panel,
    log_trapz,
    logdet,
    ols,
    dickey_fuller_a,
)

__all__ = ["PanelPORResult", "bayesian_panel_unit_root"]


@dataclass
class PanelPORResult:
    """Container for :func:`bayesian_panel_unit_root` output."""

    por: float
    log_por: float
    decision: str
    rho_hat: float
    se_rho: float
    sigma2: float
    n_units: int
    T: int
    model: str
    k: int
    a: float
    p0: float
    vartheta: float
    per_unit: pd.DataFrame = field(default=None, repr=False)

    # -- pretty printing ---------------------------------------------------- #
    def to_frame(self) -> pd.DataFrame:
        """Summary as a one-row :class:`pandas.DataFrame`."""
        return pd.DataFrame(
            {
                "model": [self.model],
                "n_units": [self.n_units],
                "T": [self.T],
                "rho_hat": [self.rho_hat],
                "se(rho)": [self.se_rho],
                "sigma2": [self.sigma2],
                "POR (beta_01)": [self.por],
                "decision": [self.decision],
            }
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        lines = [
            "Bayesian Panel Unit-Root Test  (Posterior Odds Ratio)",
            "Kumar, Chaturvedi & Afifa (2016)",
            "-" * 56,
            f"  model                 : {self.model}",
            f"  cross-section units n : {self.n_units}",
            f"  time periods T        : {self.T}",
            f"  augmentation k        : {self.k}",
            f"  prior P(H0)=p0        : {self.p0:.3f}",
            f"  vartheta              : {self.vartheta:g}",
            f"  lower bound a         : {self.a:.4f}",
            "-" * 56,
            f"  rho_hat (MLE)         : {self.rho_hat:.4f}",
            f"  se(rho_hat)           : {self.se_rho:.4f}",
            f"  sigma^2               : {self.sigma2:.4f}",
            f"  POR  beta_01          : {self.por:.6g}",
            f"  log POR               : {self.log_por:.4f}",
            "-" * 56,
            f"  Decision: {self.decision}",
        ]
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# internal builders                                                            #
# --------------------------------------------------------------------------- #
def _stack_panel(Y: np.ndarray):
    """Return y0, y, y_lag, dy and dimensions from a (T_obs, n) panel."""
    T_obs, n = Y.shape
    y0 = Y[0].copy()
    ycur = Y[1:]            # (T, n), T = T_obs - 1
    ylag = Y[:-1]
    T = ycur.shape[0]
    y = ycur.T.reshape(-1)      # stack unit by unit -> (n*T,)
    ym1 = ylag.T.reshape(-1)
    dy = y - ym1
    return y0, y, ym1, dy, n, T


def _design_Z(n: int, T: int) -> np.ndarray:
    """Z = [I_n (x) l_T , I_n (x) trend]  shape (n*T, 2n)."""
    lT = np.ones(T)
    trend = np.arange(1, T + 1, dtype=float)
    Zc = np.kron(np.eye(n), lT[:, None])      # intercepts -> (n*T, n)
    Zt = np.kron(np.eye(n), trend[:, None])   # trends     -> (n*T, n)
    return np.hstack([Zc, Zt])


def _V_rho(rho: float, n: int, vartheta: float) -> np.ndarray:
    a11 = (1.0 + rho) / (1.0 - rho)
    a22 = vartheta / (1.0 - rho) ** 2
    top = np.eye(n) * a11
    bot = np.eye(n) * a22
    V = np.zeros((2 * n, 2 * n))
    V[:n, :n] = top
    V[n:, n:] = bot
    return V


def _aug_matrix(Y: np.ndarray, k: int) -> np.ndarray:
    """Block-diagonal X of k lagged differences per unit, aligned to t=1..T.

    Returns an ``(n*T, n*k)`` matrix (zeros where lags are unavailable).
    """
    T_obs, n = Y.shape
    T = T_obs - 1
    dyfull = np.diff(Y, axis=0)              # (T, n) differences for t=1..T
    blocks = []
    for i in range(n):
        cols = []
        for j in range(1, k + 1):
            c = np.zeros(T)
            if j < T:
                c[j:] = dyfull[:-j, i]
            cols.append(c)
        blocks.append(np.column_stack(cols) if cols else np.zeros((T, 0)))
    # assemble block diagonal
    X = np.zeros((n * T, n * k))
    for i, b in enumerate(blocks):
        X[i * T:(i + 1) * T, i * k:(i + 1) * k] = b
    return X


def _mle(Y: np.ndarray, trend: str, k: int):
    """Pooled ADF-style OLS giving rho_hat, se, sigma2."""
    T_obs, n = Y.shape
    rows_y, rows_X = [], []
    dyfull = np.diff(Y, axis=0)
    for i in range(n):
        yi = Y[:, i]
        ylag = yi[:-1]
        ycur = yi[1:]
        T = len(ycur)
        t = np.arange(1, T + 1, dtype=float)
        cols = [ylag, np.ones(T)]
        if trend == "ct":
            cols.append(t)
        for j in range(1, k + 1):
            c = np.zeros(T)
            if j < T:
                c[j:] = dyfull[:-j, i]
            cols.append(c)
        rows_X.append(np.column_stack(cols))
        rows_y.append(ycur)
    X = np.vstack(rows_X)
    yv = np.concatenate(rows_y)
    beta, se, sigma2, _ = ols(X, yv)
    return float(beta[0]), float(se[0]), float(sigma2)


# --------------------------------------------------------------------------- #
# public API                                                                   #
# --------------------------------------------------------------------------- #
def bayesian_panel_unit_root(
    y,
    model: str = "trend",
    k: int = 0,
    a: float | None = None,
    p0: float = 0.5,
    vartheta: float = 1.0,
    n_grid: int = 4000,
) -> PanelPORResult:
    r"""Posterior odds ratio for the panel unit-root hypothesis.

    Parameters
    ----------
    y : array_like or DataFrame
        Panel of shape ``(T_obs, n)`` (rows = time, columns = units).
    model : {"trend", "augmented"}
        ``"trend"`` uses Theorem 1 (linear trend); ``"augmented"`` uses
        Theorem 2 (trend + ``k`` lagged differences per unit).
    k : int
        Number of augmentation terms per unit (used when ``model="augmented"``).
    a : float, optional
        Lower bound of the stationary region.  If ``None`` it is set from a
        pooled Dickey-Fuller estimate (Schotman & Van Dijk, 1991).
    p0 : float, default 0.5
        Prior probability of the unit-root hypothesis ``H0``.
    vartheta : float, default 1.0
        Prior precision hyper-parameter ``vartheta`` for the trend coefficient.
    n_grid : int, default 4000
        Number of grid points for the log-space integration over ``rho``.

    Returns
    -------
    PanelPORResult
    """
    Y = ensure_panel(y)
    if Y.shape[0] < 4:
        raise ValueError("need at least 4 time observations")
    trend = "ct"
    if model == "augmented":
        k = max(int(k), 1)
    elif model == "trend":
        k = 0
    else:
        raise ValueError("model must be 'trend' or 'augmented'")

    if a is None:
        a = dickey_fuller_a(Y, trend="ct")
    a = float(np.clip(a, -0.999, 0.0))

    y0, y, ym1, dy, n, T = _stack_panel(Y)
    Z = _design_Z(n, T)
    phi0 = np.concatenate([y0, np.zeros(n)])
    rho_hat, se_rho, sigma2 = _mle(Y, trend, k)

    grid = np.linspace(a, 0.9995, n_grid)

    if model == "trend":
        # eta = dy' R dy  with R = I - (vartheta+T)^{-1}(I_n x l_T l_T')
        c = 1.0 / (vartheta + T)
        eta = 0.0
        for i in range(n):
            blk = dy[i * T:(i + 1) * T]
            eta += float(blk @ blk - c * blk.sum() ** 2)
        eta = max(eta, 1e-300)

        def log_integrand(rho_arr):
            out = np.empty_like(rho_arr)
            for idx, rho in enumerate(rho_arr):
                G = Z.T @ Z + _V_rho(rho, n, vartheta)
                resid = y - rho * ym1
                m = Z.T @ resid + (1.0 - rho) * phi0
                eta_rho = (
                    (1.0 - rho) ** 2 * float(y0 @ y0)
                    + float(resid @ resid)
                    - float(m @ np.linalg.solve(G, m))
                )
                eta_rho = max(eta_rho, 1e-300)
                out[idx] = (
                    0.5 * n * np.log1p(rho)
                    - 1.5 * n * np.log1p(-rho)
                    - 0.5 * logdet(G)
                    - 0.5 * n * T * np.log(eta_rho)
                )
            return out

        log_I = log_trapz(log_integrand(grid), grid)
        log_por = (
            np.log(p0 / (1.0 - p0))
            + np.log(1.0 - a)
            + np.log(vartheta + T)
            - 0.5 * n * T * np.log(eta)
            - log_I
        )

    else:  # augmented
        X = _aug_matrix(Y, k)
        XtX = X.T @ X
        Sigma = np.eye(n * T) - X @ np.linalg.pinv(XtX) @ X.T
        IL = np.kron(np.eye(n), np.ones(T)[:, None])   # I_n x l_T  (n*T, n)
        A = IL.T @ Sigma @ IL + vartheta * np.eye(n)
        # varsigma = dy'[Sigma - Sigma IL A^-1 IL' Sigma] dy
        SiL = Sigma @ IL
        M = Sigma - SiL @ np.linalg.solve(A, SiL.T)
        varsigma = max(float(dy @ M @ dy), 1e-300)
        ktot = n * k
        expo = 0.5 * (n * T - ktot)

        def log_integrand(rho_arr):
            out = np.empty_like(rho_arr)
            for idx, rho in enumerate(rho_arr):
                B = Z.T @ Sigma @ Z + _V_rho(rho, n, vartheta)
                resid = y - rho * ym1
                m = Z.T @ (Sigma @ resid) + (1.0 - rho**2) * phi0
                zr = (
                    float(resid @ Sigma @ resid)
                    - float(m @ np.linalg.solve(B, m))
                )
                zr = max(zr, 1e-300)
                out[idx] = (
                    0.5 * n * np.log1p(rho)
                    - 1.5 * n * np.log1p(-rho)
                    - 0.5 * logdet(B)
                    - expo * np.log(zr)
                )
            return out

        log_I = log_trapz(log_integrand(grid), grid)
        log_por = (
            np.log(p0 / (1.0 - p0))
            + np.log(1.0 - a)
            - 0.5 * logdet(A)
            - expo * np.log(varsigma)
            - log_I
        )

    por = float(np.exp(log_por))
    decision = (
        "Reject H0 (unit root) -> series is TREND STATIONARY"
        if por < 1.0
        else "Do not reject H0 -> series is DIFFERENCE STATIONARY (unit root)"
    )

    per_unit = pd.DataFrame(
        {
            "unit": [f"unit_{i+1}" for i in range(n)],
            "mean": Y.mean(axis=0),
            "std": Y.std(axis=0, ddof=1),
        }
    )

    return PanelPORResult(
        por=por,
        log_por=float(log_por),
        decision=decision,
        rho_hat=rho_hat,
        se_rho=se_rho,
        sigma2=sigma2,
        n_units=n,
        T=T,
        model="linear trend" if model == "trend" else f"trend + aug(order {k})",
        k=k,
        a=a,
        p0=p0,
        vartheta=vartheta,
        per_unit=per_unit,
    )
