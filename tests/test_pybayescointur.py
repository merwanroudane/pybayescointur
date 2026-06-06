"""Test suite for pybayescointur."""
import numpy as np
import pytest

import pybayescointur as pbc


# --------------------------------------------------------------------------- #
# utilities                                                                    #
# --------------------------------------------------------------------------- #
def test_log_trapz_matches_known_integral():
    from pybayescointur.utils import log_trapz
    x = np.linspace(0, 1, 5000)
    log_y = np.log(np.exp(x))            # f(x) = e^x, integral = e - 1
    val = np.exp(log_trapz(log_y, x))
    assert abs(val - (np.e - 1)) < 1e-3


def test_parula_endpoints_and_length():
    cols = pbc.parula_colors(64)
    assert len(cols) == 64
    assert cols[0].startswith("#")
    rgb = pbc.parula_colors(64, as_hex=False)
    assert rgb.shape == (64, 3)
    assert rgb.min() >= 0 and rgb.max() <= 1


def test_resolve_colorscale_passthrough():
    assert pbc.resolve_colorscale("Viridis") == "Viridis"
    cs = pbc.resolve_colorscale("Parula", 4)
    assert len(cs) == 4 and cs[0][0] == 0.0 and cs[-1][0] == 1.0


# --------------------------------------------------------------------------- #
# Paper 1 - panel POR                                                          #
# --------------------------------------------------------------------------- #
def test_por_random_walk_favours_unit_root():
    df = pbc.simulate_par_panel(n=4, T=60, rho=1.0,
                                delta=[0, 0, 0, 0], seed=2)
    res = pbc.bayesian_panel_unit_root(df, model="trend")
    assert res.por > 1.0          # unit root favoured
    assert 0.8 < res.rho_hat <= 1.05


def test_por_stationary_augmented_rejects_unit_root():
    df = pbc.simulate_par_panel(n=4, T=60, rho=0.5, seed=1)
    res = pbc.bayesian_panel_unit_root(df, model="augmented", k=2)
    assert res.por < 1.0          # trend stationary
    assert res.to_frame().shape[0] == 1


# --------------------------------------------------------------------------- #
# Paper 2 - structural break                                                   #
# --------------------------------------------------------------------------- #
def test_break_por_count_and_consistency():
    df = pbc.simulate_break_panel(seed=0)
    res = pbc.bayesian_break_unit_root(df, TB=15, n_rho=120, n_lam=100)
    assert len(res.por) == 14
    # internal consistency:  beta1 / beta6 == beta2  (= P(H4)/P(H1))
    ratio = res.por[1] / res.por[6]
    assert abs(np.log(ratio) - np.log(res.por[2])) < 1e-6


def test_break_invalid_TB():
    df = pbc.simulate_break_panel(seed=0)
    with pytest.raises(ValueError):
        pbc.bayesian_break_unit_root(df, TB=0)


# --------------------------------------------------------------------------- #
# Paper 3 - cross-sectional dependence                                         #
# --------------------------------------------------------------------------- #
def test_csd_probabilities_sum_to_one_and_prefer_dependence():
    g7 = pbc.load_g7_like_gdp()
    res = pbc.bayesian_csd_comparison(g7)
    assert abs(sum(res.post_prob.values()) - 1.0) < 1e-8
    # the winning model should be a cross-dependence one (even index m2/4/6/8)
    assert res.best_model in {"m2", "m4", "m6", "m8"}
    corr = res.corr_post
    assert np.allclose(np.diag(corr), 1.0)
    assert (corr[np.triu_indices_from(corr, 1)] > 0).mean() > 0.5


# --------------------------------------------------------------------------- #
# Paper 4 - cointegration                                                      #
# --------------------------------------------------------------------------- #
def test_cointegration_recovers_ranks():
    panels = pbc.simulate_vecm_panel(N=2, n=2, T=120, ranks=[1, 0], seed=3)
    res = pbc.bayesian_panel_cointegration(
        panels, draws=500, burn=150, seed=7,
    )
    assert res.units[0].map_rank == 1     # cointegrated unit
    assert res.units[1].map_rank == 0     # no cointegration
    # rank posteriors are proper distributions
    for u in res.units:
        assert abs(sum(u.rank_prob.values()) - 1.0) < 1e-8
