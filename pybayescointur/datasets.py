"""
Data-generating processes and ready-to-use example panels.

Every paper implemented in :mod:`pybayescointur` ships with a matching
simulator so the worked examples in the README and ``examples/`` folder are
fully reproducible without external downloads.

Author: Dr Merwan Roudane (merwanroudane920@gmail.com)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "simulate_par_panel",
    "simulate_break_panel",
    "simulate_csd_panel",
    "simulate_vecm_panel",
    "load_g7_like_gdp",
]


def _rng(seed):
    return np.random.default_rng(seed)


# --------------------------------------------------------------------------- #
# Paper 1 - PAR(1) panel with trend                                           #
# --------------------------------------------------------------------------- #
def simulate_par_panel(
    n: int = 4,
    T: int = 60,
    rho: float = 0.95,
    mu=None,
    delta=None,
    sigma: float = 1.0,
    y0=None,
    seed: int | None = 0,
) -> pd.DataFrame:
    r"""Simulate a panel AR(1) process with individual trend.

    Implements ``y_it = mu_i + delta_i * t + u_it`` with
    ``u_it = rho * u_{i,t-1} + eps_it`` and ``eps_it ~ N(0, sigma^2)``
    (Kumar, Chaturvedi & Afifa, 2016).

    Returns
    -------
    pandas.DataFrame
        Shape ``(T, n)`` with columns ``unit_1 ... unit_n``.
    """
    g = _rng(seed)
    mu = np.full(n, 5.0) if mu is None else np.asarray(mu, float)
    delta = np.full(n, 0.05) if delta is None else np.asarray(delta, float)
    y0 = np.zeros(n) if y0 is None else np.asarray(y0, float)
    Y = np.zeros((T, n))
    for i in range(n):
        u = y0[i]
        for t in range(T):
            u = rho * u + g.normal(0, sigma)
            Y[t, i] = mu[i] + delta[i] * (t + 1) + u
    cols = [f"unit_{i+1}" for i in range(n)]
    return pd.DataFrame(Y, columns=cols)


# --------------------------------------------------------------------------- #
# Paper 2 - structural break in mean & variance                               #
# --------------------------------------------------------------------------- #
def simulate_break_panel(
    n: int = 3,
    T: int = 25,
    TB: int = 15,
    rho: float = 0.95,
    lam: float = 5.0,
    mu1=(30, 40, 50),
    mu2=(300, 400, 500),
    tau: float = 0.025,
    y0=(100, 120, 140),
    seed: int | None = 0,
) -> pd.DataFrame:
    r"""Simulate a PAR(1) panel with a break in mean and error variance.

    Reproduces the design of Kumar & Agiwal (2019, sec. 6.1):
    ``y_it = rho y_{i,t-1} + (1-rho) mu_i(t) + e_it`` where the mean shifts
    from ``mu1`` to ``mu2`` at ``TB`` and the innovation standard deviation is
    multiplied by ``sqrt(lam)`` after the break.

    Parameters
    ----------
    tau : float
        Error *precision* (variance ``= 1/tau``) before the break.
    lam : float
        Variance-ratio multiplier applied after the break.
    """
    g = _rng(seed)
    mu1 = np.asarray(mu1, float)
    mu2 = np.asarray(mu2, float)
    y0 = np.asarray(y0, float)
    sd = 1.0 / np.sqrt(tau)
    Y = np.zeros((T, n))
    for i in range(n):
        prev = y0[i]
        for t in range(T):
            if t < TB:
                m, s = mu1[i], sd
            else:
                m, s = mu2[i], sd * np.sqrt(lam)
            val = rho * prev + (1 - rho) * m + g.normal(0, s)
            Y[t, i] = val
            prev = val
    cols = [f"unit_{i+1}" for i in range(n)]
    return pd.DataFrame(Y, columns=cols)


# --------------------------------------------------------------------------- #
# Paper 3 - cross-sectional dependence                                        #
# --------------------------------------------------------------------------- #
def simulate_csd_panel(
    N: int = 7,
    T: int = 35,
    phi: float = 1.0,
    delta=None,
    beta=None,
    rho_cs: float = 0.8,
    sigma: float = 0.05,
    seed: int | None = 0,
) -> pd.DataFrame:
    r"""Simulate a panel with contemporaneous cross-sectional dependence.

    ``y_t = delta + beta t + phi (y_{t-1} - delta - beta(t-1)) + u_t`` with
    ``u_t ~ N(0, Sigma)`` and ``Sigma`` an equicorrelated matrix with
    off-diagonal correlation ``rho_cs`` (Meligkotsidou, Tzavalis & Vrontos).
    ``phi = 1`` yields a cross-sectionally dependent random walk.
    """
    g = _rng(seed)
    delta = np.full(N, 10.0) if delta is None else np.asarray(delta, float)
    beta = np.zeros(N) if beta is None else np.asarray(beta, float)
    Sigma = (np.eye(N) * (1 - rho_cs) + rho_cs) * sigma**2
    L = np.linalg.cholesky(Sigma)
    Y = np.zeros((T, N))
    prev = delta.copy()
    for t in range(T):
        u = L @ g.standard_normal(N)
        if phi == 1.0:
            cur = beta + prev + u
        else:
            cur = delta + beta * (t + 1) + phi * (prev - delta - beta * t) + u
        Y[t] = cur
        prev = cur
    cols = ["Canada", "France", "Germany", "Italy", "Japan", "UK", "US"][:N]
    if len(cols) < N:
        cols = [f"country_{i+1}" for i in range(N)]
    return pd.DataFrame(Y, columns=cols)


def load_g7_like_gdp(seed: int | None = 2006) -> pd.DataFrame:
    """A synthetic G7-style log-GDP panel (35 years, 7 countries).

    Designed to resemble the empirical illustration of Meligkotsidou et al.:
    cross-sectionally dependent random walks with drift.
    """
    df = simulate_csd_panel(
        N=7, T=35, phi=1.0,
        delta=[9.4, 9.5, 9.6, 9.3, 9.2, 9.5, 9.8],
        beta=[0.025, 0.022, 0.020, 0.024, 0.028, 0.018, 0.021],
        rho_cs=0.85, sigma=0.02, seed=seed,
    )
    df.index = pd.RangeIndex(1970, 1970 + len(df), name="year")
    return df


# --------------------------------------------------------------------------- #
# Paper 4 - cointegrating panel VECM                                          #
# --------------------------------------------------------------------------- #
def simulate_vecm_panel(
    N: int = 2,
    n: int = 2,
    T: int = 100,
    ranks=None,
    beta_true=None,
    alpha_scale: float = 0.2,
    sigma: float = 1.0,
    seed: int | None = 0,
):
    r"""Simulate a panel of vector error-correction models.

    For each individual ``i`` the data follow
    ``Delta y_{i,t} = alpha_i beta_i' y_{i,t-1} + eps_{i,t}`` with
    ``eps ~ N(0, sigma^2 I_n)`` (Koop, Leon-Gonzalez & Strachan, 2006).

    Parameters
    ----------
    ranks : sequence of int, optional
        Cointegrating rank ``r_i`` for each individual (0 <= r_i <= n).
        Defaults to ``r_i = 1`` for all.
    beta_true : ndarray, optional
        ``n x max_rank`` matrix of true cointegrating vectors shared across
        units (truncated to ``r_i`` columns per unit).

    Returns
    -------
    list of ndarray
        ``N`` arrays each of shape ``(T, n)``.
    """
    g = _rng(seed)
    ranks = [1] * N if ranks is None else list(ranks)
    if beta_true is None:
        beta_true = np.eye(n)
        beta_true[1:, 0] = -1.0  # e.g. (1, -1, ...) great-ratio style
    panels = []
    for i in range(N):
        r = ranks[i]
        if r == 0:
            Y = np.cumsum(g.normal(0, sigma, size=(T, n)), axis=0)
            panels.append(Y)
            continue
        beta = beta_true[:, :r]
        alpha = -alpha_scale * (g.standard_normal((n, r)) * 0.0 + 1.0)
        Pi = alpha @ beta.T
        Y = np.zeros((T, n))
        prev = g.normal(0, 1, size=n)
        for t in range(T):
            eps = g.normal(0, sigma, size=n)
            cur = prev + Pi @ prev + eps
            Y[t] = cur
            prev = cur
        panels.append(Y)
    return panels
