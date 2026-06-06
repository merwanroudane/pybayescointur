"""
Assemble and execute the pybayescointur tutorial notebook.

Produces:
  notebooks/pybayescointur_tutorial.ipynb   (executed, outputs embedded)
  docs/notebook/index.html                  (HTML export for the docs site)

Run from repo root:  python docs/build_notebook.py
"""
from __future__ import annotations

import os
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import HTMLExporter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(ROOT, "notebooks")
HTML_DIR = os.path.join(ROOT, "docs", "notebook")
os.makedirs(NB_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

md = new_markdown_cell
code = new_code_cell
cells = []

# --------------------------------------------------------------------------- #
cells.append(md(r"""
# `pybayescointur` — a hands-on tutorial

### Bayesian unit-root &amp; cointegration tests for panel data

**Author:** Dr Merwan Roudane &nbsp;·&nbsp;
✉️ [merwanroudane920@gmail.com](mailto:merwanroudane920@gmail.com) &nbsp;·&nbsp;
📦 [PyPI](https://pypi.org/project/pybayescointur/) &nbsp;·&nbsp;
🐙 [GitHub](https://github.com/merwanroudane/pybayescointur)

---

This notebook walks through the **four** Bayesian panel methods implemented in
`pybayescointur`, with a little theory, realistic data, clean tables and
light-themed visualisations.

| # | Method | Function |
|---|--------|----------|
| 1 | Panel unit root via Posterior Odds Ratio | `bayesian_panel_unit_root` |
| 2 | Unit root with structural break (mean &amp; variance) | `bayesian_break_unit_root` |
| 3 | Model comparison under cross-sectional dependence | `bayesian_csd_comparison` |
| 4 | Panel cointegration (VECM, rank inference) | `bayesian_panel_cointegration` |
"""))

cells.append(md(r"""
## A little theory

Classical unit-root and cointegration tests rely on asymptotic distributions
and a sharp null hypothesis. The **Bayesian** approach instead compares models
through their *marginal likelihoods* and reports directly interpretable
probabilities and odds.

**Posterior odds ratio.** For a null $H_0$ against an alternative $H_1$,

$$
\beta_{01}=\underbrace{\frac{P(H_0)}{P(H_1)}}_{\text{prior odds}}\times
\underbrace{\frac{P(y\mid H_0)}{P(y\mid H_1)}}_{\text{Bayes factor}}
=\frac{P(H_0\mid y)}{P(H_1\mid y)} .
$$

The **Bayes factor** integrates each model's parameters out of the likelihood,

$$
P(y\mid H_k)=\int P(y\mid\theta_k,H_k)\,p(\theta_k\mid H_k)\,d\theta_k ,
$$

so it automatically penalises complexity (an Occam factor) — no degrees-of-freedom
corrections needed.

**Decision rule.** $\beta_{01}<1$ is evidence *against* the null (e.g. reject a
unit root → trend stationary); $\beta_{01}>1$ favours the null.

**Why panels?** Pooling the cross-section ($n$ units) with the time dimension
($T$) sharply increases the power to distinguish a unit root from a stationary
root close to one — the central motivation of the panel literature.

Numerically, these marginal likelihoods involve quantities such as
$\eta(\rho)^{\,nT/2}$ that overflow instantly, so `pybayescointur` integrates
everything in **log-space** (`utils.log_trapz`).
"""))

cells.append(code(r"""
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import pybayescointur as pbc

# clean LIGHT theme for every figure
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#d0d7de", "axes.grid": True,
    "grid.color": "#e7ebf0", "grid.linewidth": 0.8,
    "axes.titleweight": "bold", "legend.frameon": False,
    "figure.dpi": 110, "font.size": 11,
})
pd.set_option("display.float_format", lambda v: f"{v:.4g}")
print("pybayescointur", pbc.__version__)
"""))

# --- Method 1 ----------------------------------------------------------- #
cells.append(md(r"""
## 1 · Panel unit root via the Posterior Odds Ratio
*Kumar, Chaturvedi &amp; Afifa (2016)*

Model: $y_{it}=\mu_i+\delta_i t+u_{it}$, $u_{it}=\rho\,u_{i,t-1}+\varepsilon_{it}$.
Test $H_0:\rho=1$ (difference stationary) vs $H_1:a<\rho<1$ (trend stationary).
We integrate $\rho$ out of the posterior; `model="augmented"` adds $k$
lagged-difference terms per unit (more reliable when $\hat\rho$ is near one).
"""))

cells.append(code(r"""
panel = pbc.simulate_par_panel(n=4, T=72, rho=0.94, delta=[0.04]*4, seed=11)
panel.columns = ["ICICI", "KM", "SBI", "UTI"]

res1 = pbc.bayesian_panel_unit_root(panel, model="augmented", k=2)
print(res1)
res1.to_frame()
"""))

cells.append(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
pbc.plot_panel(panel, title="Near-unit-root panel", ax=axes[0])
pbc.plot_por_sensitivity(panel, model="augmented", k=2, ax=axes[1])
plt.tight_layout(); plt.show()
"""))

# --- Method 2 ----------------------------------------------------------- #
cells.append(md(r"""
## 2 · Unit root with a structural break in mean &amp; variance
*Kumar &amp; Agiwal (2019)*

A PAR(1) panel with a single common break at $T_B$ that may shift both the mean
($\mu_{i1}\neq\mu_{i2}$) and the error variance ($\lambda\neq1$). Eight nested
hypotheses give **14 posterior odds ratios**; $\beta<1$ favours the numerator.
"""))

cells.append(code(r"""
brk = pbc.simulate_break_panel(n=3, T=34, TB=20, rho=0.93, lam=4.0,
                               mu1=(40,55,70), mu2=(120,150,180), seed=5)
brk.columns = ["N", "P", "K"]
res2 = pbc.bayesian_break_unit_root(brk, TB=20)
res2.to_frame()
"""))

cells.append(code(r"""
fig, ax = plt.subplots(figsize=(11, 4.6))
pbc.plot_break_por(res2, ax=ax); plt.tight_layout(); plt.show()
"""))

# --- Method 3 ----------------------------------------------------------- #
cells.append(md(r"""
## 3 · Model comparison under cross-sectional dependence
*Meligkotsidou, Tzavalis &amp; Vrontos*

Eight models — {stationary, random walk} × {trend, no trend} ×
{independent, cross-sectionally dependent} — are ranked by posterior
probability. Ignoring cross-sectional dependence can badly distort unit-root
inference, so the dependent models usually win for macro panels.
"""))

cells.append(code(r"""
gdp = pbc.load_g7_like_gdp()
res3 = pbc.bayesian_csd_comparison(gdp)
print("Most probable model:", res3.best_model)
res3.to_frame()
"""))

cells.append(code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
pbc.plot_csd_probabilities(res3, ax=axes[0])
pbc.plot_correlation_heatmap(res3, ax=axes[1])
plt.tight_layout(); plt.show()
res3.correlation_frame().round(2)
"""))

# --- Method 4 ----------------------------------------------------------- #
cells.append(md(r"""
## 4 · Bayesian panel cointegration
*Koop, Leon-Gonzalez &amp; Strachan (2006)*

Each unit has its own VECM
$\;\Delta y_{i,t}=\alpha_i\beta_i'y_{i,t-1}+\dots+\varepsilon_{i,t}\;$
with possibly **unit-specific cointegrating rank** $r_i$. Ranks are inferred via
**Savage–Dickey** Bayes factors against the no-cointegration model ($\alpha_i=0$).
"""))

cells.append(code(r"""
panels = pbc.simulate_vecm_panel(N=3, n=2, T=140, ranks=[1,0,1], seed=3)
res4 = pbc.bayesian_panel_cointegration(panels, lags=1, deterministic="c",
                                        draws=1500, burn=300,
                                        names=["France","Germany","UK"], seed=7)
print(res4)
res4.rank_frame()
"""))

cells.append(code(r"""
fig, ax = plt.subplots(figsize=(8, 3.4))
pbc.plot_rank_posterior(res4, ax=ax); plt.tight_layout(); plt.show()

print("Recovered cointegrating vector of unit 1 (~ (1, -1)/sqrt(2)):")
print(np.round(res4.units[0].beta, 3))
"""))

# --- Palette ------------------------------------------------------------ #
cells.append(md(r"""
## Colour palette

Every heatmap defaults to the MATLAB **Parula** colormap, reproduced from its
64 RGB control points. Helpers: `parula_colors`, `matlab_jet_colors`,
`turbo_colors`, and `resolve_colorscale` (for plotly).
"""))

cells.append(code(r"""
cols = pbc.parula_colors(64, as_hex=False)
fig, ax = plt.subplots(figsize=(9, 1.2))
ax.imshow([cols], aspect="auto", extent=[0,1,0,1]); ax.set_yticks([]); ax.grid(False)
ax.set_title("Parula (64 colours)"); plt.tight_layout(); plt.show()
pbc.parula_colors(6)
"""))

cells.append(md(r"""
## References

1. Kumar, Chaturvedi &amp; Afifa (2016). *Bayesian Unit Root Test for Panel Data.* EERI RP 14/2016.
2. Kumar &amp; Agiwal (2019). *Panel data unit root test with structural break: A Bayesian approach.* Hacettepe J. Math. Stat. 48(4). doi:10.15672/HJMS.2018.626
3. Meligkotsidou, Tzavalis &amp; Vrontos. *A Bayesian Analysis of Unit Roots in Panel Data Models with Cross-sectional Dependence.*
4. Koop, Leon-Gonzalez &amp; Strachan (2006). *Bayesian Inference in a Cointegrating Panel Data Model.*

---
*© 2026 Dr Merwan Roudane — MIT License — built with*
[`pybayescointur`](https://github.com/merwanroudane/pybayescointur).
"""))

# --------------------------------------------------------------------------- #
nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
})

print("Executing notebook ...")
ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
ep.preprocess(nb, {"metadata": {"path": ROOT}})

ipynb_path = os.path.join(NB_DIR, "pybayescointur_tutorial.ipynb")
with open(ipynb_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("wrote", os.path.relpath(ipynb_path, ROOT))

exporter = HTMLExporter(template_name="lab")
body, _ = exporter.from_notebook_node(nb)
html_path = os.path.join(HTML_DIR, "index.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(body)
print("wrote", os.path.relpath(html_path, ROOT))
