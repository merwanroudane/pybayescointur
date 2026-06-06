# Changelog

All notable changes to **pybayescointur** are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-06-06
### Added
- **Panel POR unit-root test** (`bayesian_panel_unit_root`) - Kumar,
  Chaturvedi & Afifa (2016): Theorem 1 (linear trend) and Theorem 2
  (trend + augmentation), log-space Monte-Carlo integration over `rho`.
- **Structural-break unit-root test** (`bayesian_break_unit_root`) - Kumar &
  Agiwal (2019): all eight hypotheses and the fourteen posterior odds ratios,
  with joint integration over `rho` and the variance ratio `lambda`.
- **Cross-sectional-dependence model comparison** (`bayesian_csd_comparison`) -
  Meligkotsidou, Tzavalis & Vrontos: marginal likelihoods and posterior
  probabilities for the eight competing models plus the posterior correlation
  matrix.
- **Panel cointegration** (`bayesian_panel_cointegration`) - Koop,
  Leon-Gonzalez & Strachan (2006): per-unit Gibbs sampler with Savage-Dickey
  Bayes factors for cointegrating-rank inference.
- Built-in simulators and a synthetic G7-style GDP panel.
- `rich`-powered console tables and matplotlib/plotly visualisations.
- MATLAB **Parula** colormap helper (`parula_colors`) plus `matlab_jet_colors`,
  `turbo_colors` and `resolve_colorscale`.
- Worked examples (`examples/`) and a `pytest` suite (`tests/`).
