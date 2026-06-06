r"""
Bayesian inference in a cointegrating panel data model.

Reference
---------
Koop, G., Leon-Gonzalez, R. & Strachan, R. (2006).
*Bayesian Inference in a Cointegrating Panel Data Model.*
University of Strathclyde / University of Leicester.

Model
-----
Each cross-sectional unit ``i`` has its own vector error-correction model::

    Delta y_{i,t} = alpha_i beta_i' y_{i,t-1}
                    + sum_{h=1}^{l} Gamma_{i,h} Delta y_{i,t-h}
                    + Phi_i d_t + eps_{i,t}

with ``Pi_i = alpha_i beta_i'`` of (possibly unit-specific) cointegrating rank
``r_i``.  The cointegrating rank of each unit is treated as an unknown random
variable and inferred through Savage-Dickey density-ratio (SDDR) Bayes factors
against the no-cointegration model (``r_i = 0``).

Implementation notes
--------------------
The posterior simulator follows the spirit of Koop et al.: a Gibbs sampler over
``(alpha, beta, Sigma)`` with proper Normal priors on ``alpha`` / ``beta`` (so
posterior odds are well defined) and an inverted-Wishart prior on the
within-unit error covariance.  Local non-identification at ``alpha = 0`` is
handled by the proper priors, and the SDDR tests ``Pi_i = 0`` directly through
``alpha_i = 0``.  Error correlation *across* units (``Sigma_ij``, ``i != j``) is
taken to be zero (the Larsson-Lyhagen-Lothgren assumption), which keeps the
sampler tractable and lets the joint rank posterior factorise over units.

Author: Dr Merwan Roudane (merwanroudane920@gmail.com)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from numpy.linalg import slogdet
from scipy.stats import invwishart

from .utils import ensure_panel

__all__ = ["UnitCointResult", "PanelCointResult", "bayesian_panel_cointegration"]


# --------------------------------------------------------------------------- #
# regressor construction                                                       #
# --------------------------------------------------------------------------- #
def _vecm_regressors(Y: np.ndarray, lags: int, deterministic: str):
    """Build Delta y, lagged level and short-run/deterministic regressors.

    Parameters
    ----------
    Y : ndarray (T_obs, n)
    lags : int
        Number of lagged differences (``l``).
    deterministic : {"n", "c", "ct"}
    """
    dY_full = np.diff(Y, axis=0)            # (T_obs-1, n)
    Tfull = dY_full.shape[0]
    start = lags
    dY = dY_full[start:]                    # (T, n)
    T = dY.shape[0]
    ylag = Y[lags:-1]                       # level y_{t-1} aligned with dY (T, n)
    cols = []
    for h in range(1, lags + 1):
        cols.append(dY_full[start - h: start - h + T])
    if deterministic in ("c", "ct"):
        cols.append(np.ones((T, 1)))
    if deterministic == "ct":
        cols.append(np.arange(1, T + 1, dtype=float)[:, None])
    W = np.hstack(cols) if cols else np.zeros((T, 0))
    return dY, ylag, W


# --------------------------------------------------------------------------- #
# multivariate normal log-density at zero                                       #
# --------------------------------------------------------------------------- #
def _mvn_logpdf_at_zero(mean: np.ndarray, cov: np.ndarray) -> float:
    k = mean.size
    sign, ld = slogdet(cov)
    if sign <= 0:
        cov = cov + 1e-8 * np.eye(k)
        sign, ld = slogdet(cov)
    sol = np.linalg.solve(cov, mean)
    quad = float(mean @ sol)
    return -0.5 * (k * np.log(2 * np.pi) + ld + quad)


# --------------------------------------------------------------------------- #
# per-unit, per-rank Gibbs sampler + SDDR                                       #
# --------------------------------------------------------------------------- #
def _fit_unit_rank(
    dY, ylag, W, r,
    draws=1500, burn=300, tau_b=10.0, tau_beta=10.0,
    v_sigma=None, seed=0,
):
    """Return log Bayes factor BF(r vs 0) and posterior summaries for one unit."""
    rng = np.random.default_rng(seed)
    T, n = dY.shape
    k = W.shape[1]
    v_sigma = (n + 2) if v_sigma is None else v_sigma
    Q_sigma = np.eye(n)

    # ---- rank 0: pure VAR in differences (closed form not needed for BF) -- #
    if r == 0:
        return 0.0, None, None, None     # reference model, log BF = 0

    # initialise
    beta = rng.standard_normal((n, r))
    Sigma = np.cov(dY.T) + 1e-6 * np.eye(n)
    if Sigma.ndim == 0:
        Sigma = Sigma.reshape(1, 1)

    sddr_log_terms = []
    alpha_store, beta_store, Sigma_store = [], [], []

    p = k + r                            # rows of coefficient matrix B
    for it in range(draws + burn):
        Sinv = np.linalg.inv(Sigma)

        # ---- draw B = (alpha; C) given beta, Sigma ----------------------- #
        Z = ylag @ beta                  # (T, r)
        X = np.hstack([Z, W])            # (T, p)
        XtX = X.T @ X
        # precision of vec(B): Sinv (x) XtX + I/tau_b^2
        prec = np.kron(Sinv, XtX) + np.eye(p * n) / tau_b**2
        rhs = (X.T @ dY @ Sinv).reshape(-1, order="F")   # vec(X'dY Sinv)
        covB = np.linalg.inv(prec)
        meanB = covB @ rhs
        Lb = np.linalg.cholesky((covB + covB.T) / 2 + 1e-10 * np.eye(p * n))
        vecB = meanB + Lb @ rng.standard_normal(p * n)
        B = vecB.reshape(p, n, order="F")
        alpha = B[:r, :].T               # (n, r)
        C = B[r:, :]                     # (k, n)

        # ---- record SDDR numerator term: density of alpha at 0 ----------- #
        # indices in vecB belonging to alpha rows 0..r-1 (column-major)
        idx = np.concatenate([j * p + np.arange(r) for j in range(n)])
        mean_a = meanB[idx]
        cov_a = covB[np.ix_(idx, idx)]
        sddr_log_terms.append(_mvn_logpdf_at_zero(mean_a, cov_a))

        # ---- draw beta given alpha, Sigma -------------------------------- #
        M = dY - W @ C                   # (T, n)
        aSa = alpha.T @ Sinv @ alpha     # (r, r)
        YtY = ylag.T @ ylag              # (n, n)
        prec_b = np.kron(aSa, YtY) + np.eye(n * r) / tau_beta**2
        rhs_b = (ylag.T @ M @ Sinv @ alpha).reshape(-1, order="F")
        cov_b = np.linalg.inv(prec_b)
        mean_b = cov_b @ rhs_b
        Lbb = np.linalg.cholesky((cov_b + cov_b.T) / 2 + 1e-10 * np.eye(n * r))
        vecbeta = mean_b + Lbb @ rng.standard_normal(n * r)
        beta = vecbeta.reshape(n, r, order="F")

        # ---- draw Sigma -------------------------------------------------- #
        E = dY - X @ B
        scale = Q_sigma + E.T @ E
        Sigma = invwishart.rvs(df=v_sigma + T, scale=scale, random_state=rng)
        Sigma = np.atleast_2d(Sigma)

        if it >= burn:
            alpha_store.append(alpha)
            beta_store.append(beta.copy())
            Sigma_store.append(Sigma.copy())

    # SDDR:  B_{0,r} = avg posterior(alpha=0) / prior(alpha=0)
    from scipy.special import logsumexp
    log_num = logsumexp(np.array(sddr_log_terms[burn:])) - np.log(len(sddr_log_terms[burn:]))
    log_prior0 = -0.5 * (n * r) * np.log(2 * np.pi * tau_b**2)
    log_B0r = log_num - log_prior0       # log Bayes factor (M0 vs Mr)
    log_BF_r0 = -log_B0r                  # log Bayes factor (Mr vs M0)

    alpha_mean = np.mean(alpha_store, axis=0)
    # normalise the cointegrating vectors for interpretability
    beta_mean = np.mean(beta_store, axis=0)
    beta_norm = beta_mean / np.linalg.norm(beta_mean, axis=0, keepdims=True)
    Sigma_mean = np.mean(Sigma_store, axis=0)
    return log_BF_r0, alpha_mean, beta_norm, Sigma_mean


# --------------------------------------------------------------------------- #
# result containers                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class UnitCointResult:
    name: str
    rank_logbf: dict            # log BF(r vs 0) per rank
    rank_prob: dict             # posterior P(r_i = r)
    map_rank: int
    alpha: np.ndarray = field(default=None, repr=False)
    beta: np.ndarray = field(default=None, repr=False)
    Sigma: np.ndarray = field(default=None, repr=False)

    def __repr__(self):  # pragma: no cover - cosmetic
        probs = "  ".join(f"r={r}:{p:.3f}" for r, p in self.rank_prob.items())
        return (f"[{self.name}] MAP rank={self.map_rank} | {probs}")


@dataclass
class PanelCointResult:
    units: list                 # list[UnitCointResult]
    common_rank_prob: dict      # posterior P(common rank = r)
    map_common_rank: int

    def rank_frame(self) -> pd.DataFrame:
        rows = []
        for u in self.units:
            row = {"unit": u.name, "MAP_rank": u.map_rank}
            row.update({f"P(r={r})": p for r, p in u.rank_prob.items()})
            rows.append(row)
        return pd.DataFrame(rows)

    def __repr__(self):  # pragma: no cover - cosmetic
        lines = [
            "Bayesian Panel Cointegration  (Koop, Leon-Gonzalez & Strachan)",
            "-" * 60,
        ]
        for u in self.units:
            lines.append("  " + repr(u))
        lines.append("-" * 60)
        cp = "  ".join(f"r={r}:{p:.3f}" for r, p in self.common_rank_prob.items())
        lines.append(f"  Common-rank posterior:  {cp}")
        lines.append(f"  MAP common rank: {self.map_common_rank}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# public API                                                                   #
# --------------------------------------------------------------------------- #
def bayesian_panel_cointegration(
    panels,
    lags: int = 1,
    deterministic: str = "c",
    max_rank: int | None = None,
    draws: int = 1500,
    burn: int = 300,
    tau_b: float = 10.0,
    tau_beta: float = 10.0,
    names=None,
    seed: int = 0,
) -> PanelCointResult:
    r"""Infer cointegrating ranks for a panel of VECMs.

    Parameters
    ----------
    panels : sequence of array_like
        One ``(T_obs, n)`` array per cross-sectional unit (all sharing the same
        ``n``).  A single DataFrame / array is treated as one unit.
    lags : int, default 1
        Number of lagged differences ``l`` in the VECM.
    deterministic : {"n", "c", "ct"}, default "c"
        Deterministic terms: none, constant, or constant + trend.
    max_rank : int, optional
        Largest rank considered (defaults to ``n``).
    draws, burn : int
        Gibbs iterations retained / discarded per (unit, rank).
    tau_b, tau_beta : float
        Prior standard deviations (shrinkage) for ``alpha``/short-run
        coefficients and for the cointegrating vectors.
    names : sequence of str, optional
        Labels for the units.
    seed : int
        Base random seed.

    Returns
    -------
    PanelCointResult
    """
    if isinstance(panels, (np.ndarray, pd.DataFrame)):
        panels = [panels]
    arrs = [ensure_panel(p) for p in panels]
    n = arrs[0].shape[1]
    if any(a.shape[1] != n for a in arrs):
        raise ValueError("all units must have the same number of variables n")
    if max_rank is None:
        max_rank = n
    if names is None:
        names = [f"unit_{i+1}" for i in range(len(arrs))]

    unit_results = []
    # for the common-rank posterior we accumulate log BFs across units
    common_logbf = {r: 0.0 for r in range(max_rank + 1)}

    for ui, (Y, nm) in enumerate(zip(arrs, names)):
        dY, ylag, W = _vecm_regressors(Y, lags, deterministic)
        rank_logbf = {0: 0.0}
        for r in range(1, max_rank + 1):
            lbf, a, b, S = _fit_unit_rank(
                dY, ylag, W, r,
                draws=draws, burn=burn, tau_b=tau_b, tau_beta=tau_beta,
                seed=seed + 100 * ui + r,
            )
            rank_logbf[r] = lbf
        # posterior over ranks (equal model priors)
        from scipy.special import logsumexp
        keys = list(rank_logbf.keys())
        lv = np.array([rank_logbf[r] for r in keys])
        post = np.exp(lv - logsumexp(lv))
        rank_prob = {r: float(p) for r, p in zip(keys, post)}
        map_rank = max(rank_prob, key=rank_prob.get)

        # refit at MAP rank to expose alpha/beta/Sigma
        a = b = S = None
        if map_rank > 0:
            _, a, b, S = _fit_unit_rank(
                dY, ylag, W, map_rank,
                draws=draws, burn=burn, tau_b=tau_b, tau_beta=tau_beta,
                seed=seed + 100 * ui + map_rank,
            )
        unit_results.append(
            UnitCointResult(nm, rank_logbf, rank_prob, map_rank, a, b, S)
        )
        for r in range(max_rank + 1):
            common_logbf[r] += rank_logbf[r]

    from scipy.special import logsumexp
    keys = list(common_logbf.keys())
    lv = np.array([common_logbf[r] for r in keys])
    cpost = np.exp(lv - logsumexp(lv))
    common_rank_prob = {r: float(p) for r, p in zip(keys, cpost)}
    map_common = max(common_rank_prob, key=common_rank_prob.get)

    return PanelCointResult(unit_results, common_rank_prob, map_common)
