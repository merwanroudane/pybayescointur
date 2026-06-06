r"""
Bayesian model comparison for panel unit roots under cross-sectional dependence.

Reference
---------
Meligkotsidou, L., Tzavalis, E. & Vrontos, I. D.
*A Bayesian Analysis of Unit Roots in Panel Data Models with Cross-sectional
Dependence.*

Idea
----
Eight competing models are compared via their marginal likelihoods and the
implied posterior model probabilities::

    m1 : stationary, no trend,  independent errors
    m2 : stationary, no trend,  cross-sectional dependence
    m3 : stationary, trend,     independent errors
    m4 : stationary, trend,     cross-sectional dependence
    m5 : pure random walk,      independent errors
    m6 : pure random walk,      cross-sectional dependence
    m7 : random walk + drift,   independent errors
    m8 : random walk + drift,   cross-sectional dependence

For the stationary models the autoregressive coefficient ``phi`` (Beta prior
concentrated near unity) is integrated out numerically; everything else is
integrated analytically.  The random-walk marginal likelihoods are available in
closed form.  Cross-sectional dependence is handled with an inverted-Wishart
prior on the full ``N x N`` error covariance matrix; the independent-error
models apply the same machinery unit-by-unit.

This module uses a unified matrix-variate-Normal prior on the deterministic
coefficients for every model, a common and convenient simplification of the
original prior set.

Author: Dr Merwan Roudane (merwanroudane920@gmail.com)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from scipy.stats import beta as beta_dist

from .utils import ensure_panel, logdet, log_trapz

__all__ = ["CSDResult", "bayesian_csd_comparison"]

_MODEL_DESC = {
    "m1": "stationary, no trend, independent",
    "m2": "stationary, no trend, cross-dependence",
    "m3": "stationary, trend, independent",
    "m4": "stationary, trend, cross-dependence",
    "m5": "pure random walk, independent",
    "m6": "pure random walk, cross-dependence",
    "m7": "random walk + drift, independent",
    "m8": "random walk + drift, cross-dependence",
}


@dataclass
class CSDResult:
    """Output of :func:`bayesian_csd_comparison`."""

    log_ml: dict          # log marginal likelihood per model
    post_prob: dict       # posterior model probability per model
    best_model: str
    Sigma_post: np.ndarray
    corr_post: np.ndarray
    names: list

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for m in [f"m{i}" for i in range(1, 9)]:
            rows.append(
                {
                    "model": m,
                    "description": _MODEL_DESC[m],
                    "log_ML": self.log_ml[m],
                    "post_prob": self.post_prob[m],
                }
            )
        return pd.DataFrame(rows).sort_values("post_prob", ascending=False)

    def correlation_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.corr_post, index=self.names, columns=self.names)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        df = self.to_frame()
        lines = [
            "Bayesian Panel Unit-Root Model Comparison",
            "Meligkotsidou, Tzavalis & Vrontos  (cross-sectional dependence)",
            "-" * 60,
        ]
        for _, r in df.iterrows():
            star = "  <= best" if r["model"] == self.best_model else ""
            lines.append(
                f"  {r['model']:>3}  P={r['post_prob']:.4f}  "
                f"logML={r['log_ML']:>10.2f}  {r['description']}{star}"
            )
        lines.append("-" * 60)
        lines.append(f"  Most probable model: {self.best_model} "
                     f"({_MODEL_DESC[self.best_model]})")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# design builders                                                              #
# --------------------------------------------------------------------------- #
def _stationary_design(Yraw: np.ndarray, phi: float, trend: bool):
    """Exact-likelihood Y, X for a stationary AR(1) at coefficient ``phi``."""
    T1, N = Yraw.shape          # T1 includes the initial obs (row 0 = y1)
    s = np.sqrt(max(1 - phi**2, 1e-12))
    Y = np.empty((T1, N))
    Y[0] = Yraw[0] * s
    Y[1:] = Yraw[1:] - phi * Yraw[:-1]
    if trend:
        X = np.empty((T1, 2))
        X[0] = [s, s]
        tt = np.arange(2, T1 + 1, dtype=float)
        X[1:, 0] = 1 - phi
        X[1:, 1] = tt - tt * phi + phi
    else:
        X = np.empty((T1, 1))
        X[0, 0] = s
        X[1:, 0] = 1 - phi
    return Y, X


def _rw_design(Yraw: np.ndarray, S: float, drift: bool):
    """Exact-likelihood Y0, X0 for a random walk (phi = 1)."""
    T1, N = Yraw.shape
    f = (S + 1.0) ** (-0.5)
    Y0 = np.empty((T1, N))
    Y0[0] = Yraw[0] * f
    Y0[1:] = Yraw[1:] - Yraw[:-1]
    if drift:
        X0 = np.zeros((T1, 2))
        X0[0] = [f, f]
        X0[1:, 1] = 1.0
    else:
        X0 = np.zeros((T1, 1))
        X0[0, 0] = f
    return Y0, X0


# --------------------------------------------------------------------------- #
# core marginal-likelihood kernel (matrix-variate normal / inverted Wishart)  #
# --------------------------------------------------------------------------- #
def _log_kernel(Y, X, B0, h, Q, v, N, T):
    """Marginal log-density after integrating out B and Sigma analytically."""
    Hinv = np.eye(X.shape[1]) / h
    K = X.T @ X + Hinv
    M = X.T @ Y + Hinv @ B0
    C = Y.T @ Y + B0.T @ Hinv @ B0
    KiM = np.linalg.solve(K, M)
    SS = C - M.T @ KiM + Q
    gam = sum(gammaln((T + v + 1 - i) / 2.0) - gammaln((v + 1 - i) / 2.0)
              for i in range(1, N + 1))
    log_h_term = -0.5 * N * X.shape[1] * np.log(h)  # = -N/2 log|H|, H = h I
    return (
        -0.5 * N * T * np.log(np.pi)
        - 0.5 * N * logdet(K)
        + log_h_term
        + 0.5 * v * logdet(Q)
        - 0.5 * (T + v) * logdet(SS)
        + gam
    )


def _stationary_logml(Yraw, trend, dep, h, S_unused, n_grid, beta_ab):
    """Marginal likelihood of a stationary model (integrate phi)."""
    T1, N = Yraw.shape
    T = T1
    a, b = beta_ab
    phis = np.linspace(1e-4, 1 - 1e-4, n_grid)

    def units():
        return range(N) if not dep else [None]

    # prior pieces
    def eval_logf(phi):
        Y, X = _stationary_design(Yraw, phi, trend)
        logp_phi = beta_dist.logpdf(phi, a, b)
        if dep:
            Q = np.cov(Yraw.T) * (N) if N > 1 else np.array([[np.var(Yraw)]])
            Q = np.atleast_2d(Q)
            v = N + 2
            B0 = np.zeros((X.shape[1], N))
            B0[0] = Yraw[0]                      # locate intercept near data level
            val = _log_kernel(Y, X, B0, h, Q, v, N, T)
            val += 0.5 * N * np.log(max(1 - phi**2, 1e-12)) + logp_phi
            return val
        # independent: product over units (sum of logs)
        total = 0.0
        for j in range(N):
            Yj = Y[:, [j]]
            Qj = np.array([[max(np.var(Yraw[:, j]), 1e-8)]])
            B0j = np.zeros((X.shape[1], 1))
            B0j[0, 0] = Yraw[0, j]
            total += _log_kernel(Yj, X, B0j, h, Qj, 3, 1, T)
            total += 0.5 * np.log(max(1 - phi**2, 1e-12))
        total += logp_phi
        return total

    logf = np.array([eval_logf(p) for p in phis])
    return log_trapz(logf, phis)


def _rw_logml(Yraw, drift, dep, h, S):
    T1, N = Yraw.shape
    T = T1
    Y0, X0 = _rw_design(Yraw, S, drift)
    if dep:
        Q = np.cov(Yraw.T) * N if N > 1 else np.array([[np.var(Yraw)]])
        Q = np.atleast_2d(Q)
        v = N + 2
        B0 = np.zeros((X0.shape[1], N))
        val = _log_kernel(Y0, X0, B0, h, Q, v, N, T)
        return val, Q, v
    total = 0.0
    for j in range(N):
        Yj = Y0[:, [j]]
        Qj = np.array([[max(np.var(Yraw[:, j]), 1e-8)]])
        B0j = np.zeros((X0.shape[1], 1))
        total += _log_kernel(Yj, X0, B0j, h, Qj, 3, 1, T)
    return total, None, None


# --------------------------------------------------------------------------- #
# public API                                                                   #
# --------------------------------------------------------------------------- #
def bayesian_csd_comparison(
    y,
    h: float | None = None,
    S: float = 1.0,
    n_grid: int = 200,
    beta_ab: tuple = (5.0, 0.5),
    prior_model=None,
) -> CSDResult:
    r"""Compare the eight panel unit-root models of Meligkotsidou et al.

    Parameters
    ----------
    y : array_like or DataFrame
        Panel ``(T, N)`` (rows = time, columns = cross-sectional units).
    h : float, optional
        Prior variance scale for the deterministic coefficients (``H = h I``).
        Defaults to ``T`` (as in the paper's ``H = T I_2``).
    S : float, default 1.0
        Start-up time hyper-parameter of the S-prior on the initial condition
        for the random-walk models.
    n_grid : int, default 200
        Grid size for the numerical integration over ``phi``.
    beta_ab : (float, float), default (5.0, 0.5)
        Parameters of the Beta prior on ``phi`` (concentrated near unity).
    prior_model : dict, optional
        Prior model probabilities ``{"m1": .., ...}``.  Defaults to uniform.

    Returns
    -------
    CSDResult
    """
    Yraw = ensure_panel(y)
    T1, N = Yraw.shape
    names = list(y.columns) if hasattr(y, "columns") else [f"u{i+1}" for i in range(N)]
    if h is None:
        h = float(T1)

    log_ml = {}
    log_ml["m1"] = _stationary_logml(Yraw, False, False, h, S, n_grid, beta_ab)
    log_ml["m2"] = _stationary_logml(Yraw, False, True, h, S, n_grid, beta_ab)
    log_ml["m3"] = _stationary_logml(Yraw, True, False, h, S, n_grid, beta_ab)
    log_ml["m4"] = _stationary_logml(Yraw, True, True, h, S, n_grid, beta_ab)
    log_ml["m5"], _, _ = _rw_logml(Yraw, False, False, h, S)
    m6, Q6, v6 = _rw_logml(Yraw, False, True, h, S)
    log_ml["m6"] = m6
    log_ml["m7"], _, _ = _rw_logml(Yraw, True, False, h, S)
    m8, Q8, v8 = _rw_logml(Yraw, True, True, h, S)
    log_ml["m8"] = m8

    models = [f"m{i}" for i in range(1, 9)]
    if prior_model is None:
        log_prior = {m: np.log(1.0 / 8) for m in models}
    else:
        s = sum(prior_model.values())
        log_prior = {m: np.log(prior_model[m] / s) for m in models}

    log_joint = np.array([log_ml[m] + log_prior[m] for m in models])
    log_norm = logsumexp(log_joint)
    post = {m: float(np.exp(lj - log_norm)) for m, lj in zip(models, log_joint)}
    best = max(post, key=post.get)

    # posterior covariance / correlation under the best dependence RW model
    Yd = np.diff(Yraw, axis=0)
    Sig = np.cov(Yd.T) if N > 1 else np.array([[np.var(Yd)]])
    Sig = np.atleast_2d(Sig)
    d = np.sqrt(np.diag(Sig))
    corr = Sig / np.outer(d, d)

    return CSDResult(
        log_ml=log_ml,
        post_prob=post,
        best_model=best,
        Sigma_post=Sig,
        corr_post=corr,
        names=names,
    )
