r"""
Bayesian panel unit-root test with a structural break in mean and variance.

Reference
---------
Kumar, J. & Agiwal, V. (2019).
*Panel data unit root test with structural break: A Bayesian approach.*
Hacettepe Journal of Mathematics and Statistics 48(4), 1213-1231.
doi:10.15672/HJMS.2018.626

Model
-----
A PAR(1) panel with a single common break at ``TB``::

    y_it = rho * y_{i,t-1} + (1-rho) * mu_i(t) + e_it
    e_it ~ N(0, 1/tau)   for t <= TB
    e_it ~ N(0, lambda/tau) for t >  TB

Eight nested hypotheses (H1-H8) combine ``rho = 1`` vs ``rho in (a,1)`` with
the presence / absence of a break in the mean (``mu_i1 != mu_i2``) and in the
variance (``lambda != 1``).  Their marginal posterior probabilities give rise
to 14 posterior odds ratios (POR1-POR14), each comparing one hypothesis
against another (``beta < 1`` favours the numerator / null hypothesis).

Author: Dr Merwan Roudane (merwanroudane920@gmail.com)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import gammaln

from .utils import ensure_panel, log_trapz, dickey_fuller_a

__all__ = ["StructuralBreakResult", "bayesian_break_unit_root"]


# Human-readable description of each posterior odds ratio (null vs alternative)
_POR_LABELS = {
    1: "DS, break in var  vs  TS, break in mean & var",
    2: "TS, break in var  vs  TS, break in mean & var",
    3: "TS, break in mean vs  TS, break in mean & var",
    4: "DS, no break      vs  TS, break in mean & var",
    5: "TS, no break      vs  TS, break in mean & var",
    6: "DS, break in var  vs  TS, break in var",
    7: "TS, break in mean vs  TS, break in var",
    8: "DS, no break      vs  TS, break in var",
    9: "TS, no break      vs  TS, break in var",
    10: "DS, no break      vs  TS, break in mean",
    11: "TS, no break      vs  TS, break in mean",
    12: "DS, break in var  vs  TS, break in mean",
    13: "DS, break in var  vs  TS, no break",
    14: "DS, no break      vs  TS, no break",
}


@dataclass
class StructuralBreakResult:
    """Output of :func:`bayesian_break_unit_root`."""

    TB: int
    n_units: int
    T: int
    a: float
    theta: float
    log_post: dict          # log P(y|H) for the distinct hypotheses
    por: dict               # POR1 .. POR14 (natural scale)
    labels: dict = field(default_factory=lambda: dict(_POR_LABELS))

    def to_frame(self) -> pd.DataFrame:
        """The 14 posterior odds ratios as a tidy DataFrame."""
        rows = []
        for j in range(1, 15):
            val = self.por[j]
            rows.append(
                {
                    "POR": f"beta_{j:02d}",
                    "comparison": self.labels[j],
                    "value": val,
                    "decision": "reject null" if val < 1 else "accept null",
                }
            )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        head = [
            "Bayesian Panel Unit-Root Test with Structural Break",
            "Kumar & Agiwal (2019)",
            "-" * 60,
            f"  break point TB : {self.TB}",
            f"  units n        : {self.n_units}",
            f"  periods T       : {self.T}",
            f"  prior theta     : {self.theta:.3f}",
            f"  lower bound a   : {self.a:.4f}",
            "-" * 60,
            "  Posterior odds ratios (beta < 1 -> favour numerator):",
        ]
        for j in range(1, 15):
            head.append(f"   POR{j:>2}  {self.por[j]:>12.4g}   {self.labels[j]}")
        return "\n".join(head)


# --------------------------------------------------------------------------- #
# residual sums (depend on rho)                                                #
# --------------------------------------------------------------------------- #
def _resid_sums(Y: np.ndarray, TB: int, rho: float):
    """Per-unit residual sums for both regimes at a given ``rho``.

    Returns dict of per-unit arrays: d1, s1 (segment 1), d2, s2 (segment 2),
    plus y0.  ``y_it - rho y_{i,t-1}`` for t = 1..T with y_{i0} = Y[0].
    """
    y0 = Y[0]
    resid = Y[1:] - rho * Y[:-1]            # (T, n), t = 1..T
    seg1 = resid[:TB]                       # t = 1..TB
    seg2 = resid[TB:]                       # t = TB+1..T
    return {
        "y0": y0,
        "d1": seg1.sum(axis=0),
        "s1": (seg1**2).sum(axis=0),
        "d2": seg2.sum(axis=0),
        "s2": (seg2**2).sum(axis=0),
    }


# --------------------------------------------------------------------------- #
# log marginal posterior probabilities (constants cancel in every ratio)       #
# --------------------------------------------------------------------------- #
def _log_post_probs(Y, TB, a, n_rho, n_lam, lam_lo, lam_hi):
    L, n = Y.shape
    T = L - 1
    const = gammaln(n * T / 2.0) - (n * T / 2.0) * np.log(np.pi)

    rho_grid = np.linspace(a, 0.9995, n_rho)
    lam_grid = np.exp(np.linspace(np.log(lam_lo), np.log(lam_hi), n_lam))

    # difference-based quantities (rho = 1) -------------------------------- #
    rs1 = _resid_sums(Y, TB, 1.0)
    S_seg1 = rs1["s1"].sum()
    S_seg2 = rs1["s2"].sum()
    # S(lambda) for H2/H3 and the (yit-yit-1)^2 total for H6/H7
    diff_total = (np.diff(Y, axis=0) ** 2).sum()

    # ---- H6 / H7 : difference stationary, no break (closed form) --------- #
    logH6 = const - (n * T / 2.0) * np.log(diff_total)

    # ---- H2 / H3 : difference stationary, break in variance -------------- #
    #   integral over lambda of 1/(lambda S(lambda)^{nT/2})
    def log_h2_lam(lam):
        S = S_seg1 + S_seg2 / lam
        return -np.log(lam) - (n * T / 2.0) * np.log(S)

    logH2 = const + log_trapz(np.array([log_h2_lam(l) for l in lam_grid]), lam_grid)

    # ---- H8 : trend stationary (level), no break ------------------------- #
    def log_h8_rho(rho):
        rs = _resid_sums(Y, TB, rho)
        P = T * (1 - rho) ** 2 + (1 - rho**2)
        # Q_i and R over all t = 1..T
        sumsq = rs["s1"] + rs["s2"]
        d_all = rs["d1"] + rs["d2"]
        Q = (1 - rho) * d_all + (1 - rho**2) * rs["y0"]
        R = (sumsq.sum()
             + (1 - rho**2) * (rs["y0"] ** 2).sum()
             - (Q**2 / P).sum())
        return (n / 2.0) * np.log1p(-rho**2) - np.log1p(-a) \
            - (n / 2.0) * np.log(P) - (n * T / 2.0) * np.log(max(R, 1e-300))

    logH8 = const + log_trapz(np.array([log_h8_rho(r) for r in rho_grid]), rho_grid)

    # ---- H5 : trend stationary, break in mean only ----------------------- #
    def log_h5_rho(rho):
        rs = _resid_sums(Y, TB, rho)
        M = TB * (1 - rho) ** 2 + (1 - rho**2)
        N = (T - TB) * (1 - rho) ** 2 + (1 - rho**2)
        K = (1 - rho) * rs["d1"] + (1 - rho**2) * rs["y0"]
        Lv = (1 - rho) * rs["d2"] + (1 - rho**2) * rs["y0"]
        sumsq = rs["s1"] + rs["s2"]
        O = (sumsq.sum()
             + 2 * (1 - rho**2) * (rs["y0"] ** 2).sum()
             - (K**2 / M).sum()
             - (Lv**2 / N).sum())
        return n * np.log1p(-rho**2) - np.log1p(-a) \
            - (n / 2.0) * np.log(M) - (n / 2.0) * np.log(N) \
            - (n * T / 2.0) * np.log(max(O, 1e-300))

    logH5 = const + log_trapz(np.array([log_h5_rho(r) for r in rho_grid]), rho_grid)

    # ---- H4 : trend stationary, break in variance only ------------------- #
    def log_h4_rho(rho):
        rs = _resid_sums(Y, TB, rho)

        def inner(lam):
            A = TB * (1 - rho) ** 2 + ((T - TB) / lam) * (1 - rho) ** 2 + (1 - rho**2)
            B = ((1 - rho) * rs["d1"]
                 + ((1 - rho) / lam) * rs["d2"]
                 + (1 - rho**2) * rs["y0"])
            C = (rs["s1"].sum()
                 + rs["s2"].sum() / lam
                 + (1 - rho**2) * (rs["y0"] ** 2).sum()
                 - (B**2 / A).sum())
            return -np.log(lam) - (n / 2.0) * np.log(A) - (n * T / 2.0) * np.log(max(C, 1e-300))

        lam_part = log_trapz(np.array([inner(l) for l in lam_grid]), lam_grid)
        return (n / 2.0) * np.log1p(-rho**2) - np.log1p(-a) + lam_part

    logH4 = const + log_trapz(np.array([log_h4_rho(r) for r in rho_grid]), rho_grid)

    # ---- H1 : trend stationary, break in mean AND variance --------------- #
    def log_h1_rho(rho):
        rs = _resid_sums(Y, TB, rho)

        def inner(lam):
            M = TB * (1 - rho) ** 2 + (1 - rho**2)
            G = ((T - TB) / lam) * (1 - rho) ** 2 + (1 - rho**2)
            K = (1 - rho) * rs["d1"] + (1 - rho**2) * rs["y0"]
            H = ((1 - rho) / lam) * rs["d2"] + (1 - rho**2) * rs["y0"]
            I = (rs["s1"].sum()
                 + rs["s2"].sum() / lam
                 + 2 * (1 - rho**2) * (rs["y0"] ** 2).sum()
                 - (K**2 / M).sum()
                 - (H**2 / G).sum())
            return (-np.log(lam) - (n / 2.0) * np.log(M) - (n / 2.0) * np.log(G)
                    - (n * T / 2.0) * np.log(max(I, 1e-300)))

        lam_part = log_trapz(np.array([inner(l) for l in lam_grid]), lam_grid)
        return n * np.log1p(-rho**2) - np.log1p(-a) + lam_part

    logH1 = const + log_trapz(np.array([log_h1_rho(r) for r in rho_grid]), rho_grid)

    return {
        "H1": logH1, "H2": logH2, "H4": logH4,
        "H5": logH5, "H6": logH6, "H8": logH8,
    }


# --------------------------------------------------------------------------- #
# public API                                                                   #
# --------------------------------------------------------------------------- #
def bayesian_break_unit_root(
    y,
    TB: int,
    a: float | None = None,
    theta: float = 0.5,
    n_rho: int = 200,
    n_lam: int = 160,
    lam_lo: float = 0.05,
    lam_hi: float = 20.0,
) -> StructuralBreakResult:
    r"""Compute the 14 posterior odds ratios of Kumar & Agiwal (2019).

    Parameters
    ----------
    y : array_like or DataFrame
        Panel of shape ``(T_obs, n)``.  The first row is treated as the initial
        condition ``y_{i0}``; the effective time dimension is ``T = T_obs - 1``.
    TB : int
        Break point (in ``1 .. T``).
    a : float, optional
        Lower bound of the stationary region (defaults to a Dickey-Fuller value).
    theta : float, default 0.5
        Prior probability of the null hypothesis, ``O(H0) = theta/(1-theta)``.
    n_rho, n_lam : int
        Grid sizes for the ``rho`` and ``lambda`` integrations.
    lam_lo, lam_hi : float
        Integration limits for the variance-ratio ``lambda`` (log-spaced grid).

    Returns
    -------
    StructuralBreakResult
    """
    Y = ensure_panel(y)
    L, n = Y.shape
    T = L - 1
    if not (1 <= TB <= T - 1):
        raise ValueError(f"TB must lie in 1..{T-1}")
    if a is None:
        a = dickey_fuller_a(Y, trend="c")
    a = float(np.clip(a, -0.999, 0.0))

    lp = _log_post_probs(Y, TB, a, n_rho, n_lam, lam_lo, lam_hi)
    odds = np.log(theta / (1.0 - theta))

    def ratio(num, den):
        return float(np.exp(odds + lp[num] - lp[den]))

    # P(H2)=P(H3); P(H6)=P(H7) - use the representative key
    por = {
        1: ratio("H2", "H1"),
        2: ratio("H4", "H1"),
        3: ratio("H5", "H1"),
        4: ratio("H6", "H1"),
        5: ratio("H8", "H1"),
        6: ratio("H2", "H4"),
        7: ratio("H5", "H4"),
        8: ratio("H6", "H4"),
        9: ratio("H8", "H4"),
        10: ratio("H6", "H5"),
        11: ratio("H8", "H5"),
        12: ratio("H2", "H5"),
        13: ratio("H2", "H8"),
        14: ratio("H6", "H8"),
    }

    return StructuralBreakResult(
        TB=TB, n_units=n, T=T, a=a, theta=theta,
        log_post=lp, por=por,
    )
