"""
Generate every figure (PNG) and table (HTML + CSV) used by the documentation
site and the README, from realistic reproductions of each paper's application.

All plots use a clean **light** theme (white background) and the MATLAB Parula
colormap.  Run from the repo root:

    python docs/generate_assets.py

Author: Dr Merwan Roudane (merwanroudane920@gmail.com)
"""
from __future__ import annotations

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pybayescointur as pbc

HERE = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(HERE, "images")
TAB = os.path.join(HERE, "tables")
os.makedirs(IMG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

ACCENT = "#2563eb"   # site accent (royal blue)

# --------------------------------------------------------------------------- #
# clean LIGHT theme                                                            #
# --------------------------------------------------------------------------- #
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#d0d7de",
    "axes.labelcolor": "#1f2328",
    "axes.titlecolor": "#1f2328",
    "axes.grid": True,
    "grid.color": "#e7ebf0",
    "grid.linewidth": 0.8,
    "xtick.color": "#57606a",
    "ytick.color": "#57606a",
    "text.color": "#1f2328",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "legend.frameon": False,
    "figure.dpi": 130,
})


def save(fig, name):
    path = os.path.join(IMG, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  fig ->", os.path.relpath(path, HERE))


def save_table(df: pd.DataFrame, name: str, caption: str, fmt=None,
               highlight=None):
    """Write a styled light-theme HTML table + a CSV."""
    df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
    sty = df.style.hide(axis="index").set_caption(caption)
    if fmt:
        sty = sty.format(fmt)
    sty = sty.set_table_styles([
        {"selector": "caption",
         "props": "caption-side:top;font-weight:700;font-size:1.0rem;"
                  "color:#1f2328;padding:8px 0;text-align:left;"},
        {"selector": "th",
         "props": "background:#f1f5ff;color:#1f2328;font-weight:600;"
                  "padding:8px 12px;border-bottom:2px solid #c9d6f5;text-align:right;"},
        {"selector": "td",
         "props": "padding:7px 12px;border-bottom:1px solid #eef1f5;text-align:right;"},
        {"selector": "tr:hover td", "props": "background:#f7faff;"},
        {"selector": "table",
         "props": "border-collapse:collapse;font-family:'Segoe UI',system-ui,sans-serif;"
                  "font-size:0.92rem;width:100%;"},
    ])
    if highlight is not None:
        sty = highlight(sty)
    html = sty.to_html()
    with open(os.path.join(TAB, name + ".html"), "w", encoding="utf-8") as f:
        f.write(html)
    print("  tab ->", os.path.relpath(os.path.join(TAB, name + '.html'), HERE))


# =========================================================================== #
# METHOD 1 - Panel POR unit-root test                                         #
# =========================================================================== #
def method1():
    print("Method 1: Panel POR unit-root test")
    # realistic near-unit-root trending panel (cf. NPS NAV application)
    panel = pbc.simulate_par_panel(n=4, T=72, rho=0.94, delta=[0.04]*4, seed=11)
    panel.columns = ["ICICI", "KM", "SBI", "UTI"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    pbc.plot_panel(panel, title="Panel series (4 fund-style units, near unit root)", ax=ax)
    save(fig, "m1_panel.png")

    fig, ax = plt.subplots(figsize=(8, 4.6))
    pbc.plot_por_sensitivity(panel, model="augmented", k=2, ax=ax)
    save(fig, "m1_por_sensitivity.png")

    # comparison table across specifications
    rows = []
    for model, k in [("trend", 0), ("augmented", 1), ("augmented", 2)]:
        r = pbc.bayesian_panel_unit_root(panel, model=model, k=k)
        rows.append({
            "specification": r.model,
            "rho_hat": r.rho_hat,
            "se(rho)": r.se_rho,
            "sigma^2": r.sigma2,
            "POR (beta_01)": r.por,
            "decision": "trend stationary" if r.por < 1 else "unit root",
        })
    df = pd.DataFrame(rows)
    save_table(df, "m1_por", "Posterior odds ratios across specifications",
               fmt={"rho_hat": "{:.4f}", "se(rho)": "{:.4f}",
                    "sigma^2": "{:.3f}", "POR (beta_01)": "{:.3e}"})


# =========================================================================== #
# METHOD 2 - structural break in mean & variance                              #
# =========================================================================== #
def method2():
    print("Method 2: structural break")
    panel = pbc.simulate_break_panel(
        n=3, T=34, TB=20, rho=0.93, lam=4.0,
        mu1=(40, 55, 70), mu2=(120, 150, 180), seed=5)
    panel.columns = ["N (nitrogen)", "P (phosphate)", "K (potash)"]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    pbc.plot_panel(panel, title="Panel with a structural break (TB = 20)", ax=ax)
    ax.axvline(20, color="#e8590c", ls="--", lw=1.4, label="break point")
    ax.legend(fontsize=8, ncol=2)
    save(fig, "m2_panel.png")

    res = pbc.bayesian_break_unit_root(panel, TB=20)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    pbc.plot_break_por(res, ax=ax)
    save(fig, "m2_break_por.png")

    df = res.to_frame()
    save_table(df, "m2_break", "The 14 posterior odds ratios (TB = 20)",
               fmt={"value": "{:.3e}"})


# =========================================================================== #
# METHOD 3 - cross-sectional dependence                                       #
# =========================================================================== #
def method3():
    print("Method 3: cross-sectional dependence")
    gdp = pbc.load_g7_like_gdp()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    pbc.plot_panel(gdp, title="G7-style log-GDP panel (1970-2004)", ax=ax)
    save(fig, "m3_panel.png")

    res = pbc.bayesian_csd_comparison(gdp)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    pbc.plot_csd_probabilities(res, ax=ax)
    save(fig, "m3_probs.png")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    pbc.plot_correlation_heatmap(res, ax=ax)
    save(fig, "m3_corr.png")

    df = res.to_frame()
    save_table(df, "m3_models", "Posterior model probabilities (8 models)",
               fmt={"log_ML": "{:.2f}", "post_prob": "{:.4f}"})
    corr = res.correlation_frame().round(2).reset_index().rename(
        columns={"index": "country"})
    save_table(corr, "m3_corr", "Estimated cross-sectional correlation matrix")


# =========================================================================== #
# METHOD 4 - panel cointegration                                              #
# =========================================================================== #
def method4():
    print("Method 4: panel cointegration")
    panels = pbc.simulate_vecm_panel(N=3, n=2, T=140, ranks=[1, 0, 1], seed=3)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    pbc.plot_panel(pd.DataFrame(panels[0], columns=["exchange rate", "money/income"]),
                   title="Unit 1: a cointegrated bivariate system", ax=ax)
    save(fig, "m4_panel.png")

    res = pbc.bayesian_panel_cointegration(
        panels, lags=1, deterministic="c", draws=1500, burn=300,
        names=["France", "Germany", "UK"], seed=7)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    pbc.plot_rank_posterior(res, ax=ax)
    save(fig, "m4_rank.png")

    df = res.rank_frame()
    save_table(df, "m4_rank", "Cointegrating-rank posterior probabilities",
               fmt={c: "{:.3f}" for c in df.columns if c.startswith("P(")})


# =========================================================================== #
# Parula colormap showcase                                                     #
# =========================================================================== #
def palette():
    print("Parula palette swatch")
    cols = pbc.parula_colors(64, as_hex=False)
    fig, ax = plt.subplots(figsize=(9, 1.3))
    ax.imshow([cols], aspect="auto", extent=[0, 1, 0, 1])
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.set_title("MATLAB Parula colormap (64 control points)", fontweight="bold")
    ax.grid(False)
    save(fig, "parula.png")


if __name__ == "__main__":
    method1()
    method2()
    method3()
    method4()
    palette()
    print("\nAll assets written to", os.path.relpath(IMG), "and", os.path.relpath(TAB))
