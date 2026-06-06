"""
pybayescointur
==============

Bayesian unit-root and cointegration tests for **panel data**, implementing
four published methodologies behind a single, consistent API with publication
quality tables and visualisations.

Implemented methods
-------------------
1. **Panel POR unit-root test** - Kumar, Chaturvedi & Afifa (2016).
   :func:`bayesian_panel_unit_root`
2. **Structural-break unit-root test** - Kumar & Agiwal (2019).
   :func:`bayesian_break_unit_root`
3. **Cross-sectional-dependence model comparison** - Meligkotsidou,
   Tzavalis & Vrontos. :func:`bayesian_csd_comparison`
4. **Panel cointegration (VECM)** - Koop, Leon-Gonzalez & Strachan (2006).
   :func:`bayesian_panel_cointegration`

Author
------
Dr Merwan Roudane  -  merwanroudane920@gmail.com
https://github.com/merwanroudane/pybayescointur
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Dr Merwan Roudane"
__email__ = "merwanroudane920@gmail.com"
__url__ = "https://github.com/merwanroudane/pybayescointur"

# core estimators
from .por_panel import bayesian_panel_unit_root, PanelPORResult
from .structural_break import bayesian_break_unit_root, StructuralBreakResult
from .cross_section import bayesian_csd_comparison, CSDResult
from .cointegration import (
    bayesian_panel_cointegration,
    PanelCointResult,
    UnitCointResult,
)

# data
from .datasets import (
    simulate_par_panel,
    simulate_break_panel,
    simulate_csd_panel,
    simulate_vecm_panel,
    load_g7_like_gdp,
)

# tables
from . import tables
from .tables import por_table, break_table, csd_table, coint_rank_table

# visualisation
from . import viz
from .viz import (
    parula_colors,
    matlab_jet_colors,
    turbo_colors,
    resolve_colorscale,
    plot_panel,
    plot_por_sensitivity,
    plot_break_por,
    plot_csd_probabilities,
    plot_correlation_heatmap,
    plot_rank_posterior,
)

__all__ = [
    "__version__",
    # estimators
    "bayesian_panel_unit_root", "PanelPORResult",
    "bayesian_break_unit_root", "StructuralBreakResult",
    "bayesian_csd_comparison", "CSDResult",
    "bayesian_panel_cointegration", "PanelCointResult", "UnitCointResult",
    # data
    "simulate_par_panel", "simulate_break_panel", "simulate_csd_panel",
    "simulate_vecm_panel", "load_g7_like_gdp",
    # tables
    "tables", "por_table", "break_table", "csd_table", "coint_rank_table",
    # viz
    "viz", "parula_colors", "matlab_jet_colors", "turbo_colors",
    "resolve_colorscale", "plot_panel", "plot_por_sensitivity",
    "plot_break_por", "plot_csd_probabilities", "plot_correlation_heatmap",
    "plot_rank_posterior",
]
