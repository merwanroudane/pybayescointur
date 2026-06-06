"""
Visualisations for :mod:`pybayescointur`.

Static figures use matplotlib; interactive figures use plotly.  Following the
author's house style every heatmap / 3-D / contour plot defaults to the MATLAB
R2014b **Parula** colormap, reproduced from its 64 RGB control points via
``grDevices::colorRampPalette``-style interpolation.

Author: Dr Merwan Roudane (merwanroudane920@gmail.com)
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "parula_colors",
    "matlab_jet_colors",
    "turbo_colors",
    "resolve_colorscale",
    "plot_panel",
    "plot_por_sensitivity",
    "plot_break_por",
    "plot_csd_probabilities",
    "plot_correlation_heatmap",
    "plot_rank_posterior",
]

# --------------------------------------------------------------------------- #
# MATLAB R2014b Parula - 64 RGB control points (0-1 scale)                     #
# --------------------------------------------------------------------------- #
_PARULA_64 = np.array([
    [0.2081, 0.1663, 0.5292], [0.2116, 0.1898, 0.5777], [0.2123, 0.2138, 0.6270],
    [0.2081, 0.2386, 0.6771], [0.1959, 0.2645, 0.7279], [0.1707, 0.2919, 0.7792],
    [0.1253, 0.3242, 0.8303], [0.0591, 0.3598, 0.8683], [0.0117, 0.3875, 0.8820],
    [0.0060, 0.4086, 0.8828], [0.0165, 0.4266, 0.8786], [0.0329, 0.4430, 0.8720],
    [0.0498, 0.4586, 0.8641], [0.0629, 0.4737, 0.8554], [0.0723, 0.4887, 0.8467],
    [0.0779, 0.5040, 0.8384], [0.0793, 0.5200, 0.8312], [0.0749, 0.5375, 0.8263],
    [0.0641, 0.5570, 0.8240], [0.0488, 0.5772, 0.8228], [0.0343, 0.5966, 0.8199],
    [0.0265, 0.6137, 0.8135], [0.0239, 0.6287, 0.8038], [0.0231, 0.6418, 0.7913],
    [0.0228, 0.6535, 0.7768], [0.0267, 0.6642, 0.7607], [0.0384, 0.6743, 0.7436],
    [0.0590, 0.6838, 0.7254], [0.0843, 0.6928, 0.7062], [0.1133, 0.7015, 0.6859],
    [0.1453, 0.7098, 0.6646], [0.1801, 0.7177, 0.6424], [0.2178, 0.7250, 0.6193],
    [0.2586, 0.7317, 0.5954], [0.3022, 0.7376, 0.5712], [0.3482, 0.7424, 0.5473],
    [0.3953, 0.7459, 0.5244], [0.4420, 0.7481, 0.5033], [0.4871, 0.7491, 0.4840],
    [0.5300, 0.7491, 0.4661], [0.5709, 0.7485, 0.4494], [0.6099, 0.7473, 0.4337],
    [0.6473, 0.7456, 0.4188], [0.6834, 0.7435, 0.4044], [0.7184, 0.7411, 0.3905],
    [0.7525, 0.7384, 0.3768], [0.7858, 0.7356, 0.3633], [0.8185, 0.7327, 0.3498],
    [0.8507, 0.7299, 0.3360], [0.8824, 0.7274, 0.3217], [0.9139, 0.7258, 0.3063],
    [0.9450, 0.7261, 0.2886], [0.9739, 0.7314, 0.2666], [0.9938, 0.7455, 0.2403],
    [0.9990, 0.7653, 0.2164], [0.9955, 0.7861, 0.1967], [0.9880, 0.8066, 0.1794],
    [0.9789, 0.8271, 0.1633], [0.9697, 0.8481, 0.1475], [0.9626, 0.8705, 0.1309],
    [0.9589, 0.8949, 0.1132], [0.9598, 0.9218, 0.0948], [0.9661, 0.9514, 0.0755],
    [0.9763, 0.9831, 0.0538],
])


def _interp_palette(anchors: np.ndarray, n: int) -> np.ndarray:
    """colorRampPalette-style linear interpolation to ``n`` colours."""
    m = anchors.shape[0]
    src = np.linspace(0, 1, m)
    dst = np.linspace(0, 1, n)
    out = np.empty((n, 3))
    for c in range(3):
        out[:, c] = np.interp(dst, src, anchors[:, c])
    return np.clip(out, 0, 1)


def _to_hex(rgb: np.ndarray) -> list:
    return ["#{:02x}{:02x}{:02x}".format(*(np.round(c * 255).astype(int)))
            for c in rgb]


def parula_colors(n: int = 64, as_hex: bool = True):
    """Return ``n`` colours of the MATLAB Parula colormap.

    Parameters
    ----------
    n : int, default 64
        Number of colours.
    as_hex : bool, default True
        Return ``#rrggbb`` strings; otherwise an ``(n, 3)`` float array.
    """
    rgb = _interp_palette(_PARULA_64, n)
    return _to_hex(rgb) if as_hex else rgb


# a couple of companion maps (anchors only -> interpolated on demand)
_JET = np.array([[0, 0, 0.5], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0.5, 0, 0]])
_TURBO = np.array([
    [0.190, 0.072, 0.232], [0.275, 0.408, 0.859], [0.180, 0.718, 0.642],
    [0.534, 0.886, 0.221], [0.962, 0.728, 0.176], [0.880, 0.215, 0.027],
    [0.479, 0.012, 0.013],
])


def matlab_jet_colors(n: int = 64, as_hex: bool = True):
    """``n`` colours of the MATLAB Jet colormap."""
    rgb = _interp_palette(_JET, n)
    return _to_hex(rgb) if as_hex else rgb


def turbo_colors(n: int = 64, as_hex: bool = True):
    """``n`` colours of the Google Turbo colormap."""
    rgb = _interp_palette(_TURBO, n)
    return _to_hex(rgb) if as_hex else rgb


def resolve_colorscale(name: str = "Parula", n: int = 32):
    """Return a plotly colorscale ``[[value, hex], ...]`` for ``name``.

    Recognises ``"Parula"`` (default), ``"Jet"`` and ``"Turbo"``; any other name
    is passed straight through for plotly's built-in scales.
    """
    fac = {"parula": parula_colors, "jet": matlab_jet_colors, "turbo": turbo_colors}
    key = name.lower()
    if key not in fac:
        return name
    hexes = fac[key](n)
    vals = np.linspace(0, 1, n)
    return [[float(v), h] for v, h in zip(vals, hexes)]


def _mpl_cmap(name="Parula", n=256):
    from matplotlib.colors import LinearSegmentedColormap
    rgb = parula_colors(n, as_hex=False) if name.lower() == "parula" else \
        (matlab_jet_colors(n, as_hex=False) if name.lower() == "jet"
         else turbo_colors(n, as_hex=False))
    return LinearSegmentedColormap.from_list(name, rgb, N=n)


# --------------------------------------------------------------------------- #
# generic panel plot                                                          #
# --------------------------------------------------------------------------- #
def plot_panel(y, title="Panel series", ax=None):
    """Line plot of every cross-sectional series in a panel."""
    import matplotlib.pyplot as plt
    from .utils import ensure_panel

    Y = ensure_panel(y)
    names = list(y.columns) if hasattr(y, "columns") else \
        [f"unit {i+1}" for i in range(Y.shape[1])]
    idx = y.index if hasattr(y, "index") else np.arange(Y.shape[0])
    colors = parula_colors(max(Y.shape[1], 2))
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    for j in range(Y.shape[1]):
        ax.plot(idx, Y[:, j], color=colors[j % len(colors)], lw=1.6, label=names[j])
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    return ax


# --------------------------------------------------------------------------- #
# Paper 1 - POR sensitivity to the prior P(H0)                                 #
# --------------------------------------------------------------------------- #
def plot_por_sensitivity(y, model="augmented", k=2, p0_grid=None, ax=None):
    """POR as a function of the prior probability ``p0`` of the unit root."""
    import matplotlib.pyplot as plt
    from .por_panel import bayesian_panel_unit_root

    if p0_grid is None:
        p0_grid = np.linspace(0.1, 0.9, 17)
    pors = [bayesian_panel_unit_root(y, model=model, k=k, p0=p).por for p in p0_grid]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p0_grid, pors, "-o", color=parula_colors(3)[1], lw=2)
    ax.axhline(1.0, color="crimson", ls="--", lw=1.2, label="POR = 1 (indifference)")
    ax.set_yscale("log")
    ax.set_xlabel("prior P(H0) = p0")
    ax.set_ylabel("Posterior Odds Ratio  (log scale)")
    ax.set_title("POR sensitivity to the unit-root prior", fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


# --------------------------------------------------------------------------- #
# Paper 2 - structural-break POR bar chart                                     #
# --------------------------------------------------------------------------- #
def plot_break_por(result, ax=None):
    """Bar chart of the 14 structural-break posterior odds ratios."""
    import matplotlib.pyplot as plt

    df = result.to_frame()
    vals = np.log10(np.clip(df["value"].values, 1e-300, None))
    colors = ["#2ca25f" if v < 0 else "#de2d26" for v in vals]
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df["POR"], vals, color=colors)
    ax.axhline(0, color="black", lw=1)
    ax.set_ylabel("log10(POR)   (<0 -> reject null)")
    ax.set_title(f"Structural-break posterior odds ratios (TB={result.TB})",
                 fontweight="bold")
    ax.tick_params(axis="x", rotation=60)
    ax.grid(alpha=0.3, axis="y")
    return ax


# --------------------------------------------------------------------------- #
# Paper 3 - model probabilities + correlation heatmap                          #
# --------------------------------------------------------------------------- #
def plot_csd_probabilities(result, ax=None):
    """Bar chart of posterior model probabilities for the 8 CSD models."""
    import matplotlib.pyplot as plt

    df = result.to_frame().sort_values("model")
    colors = parula_colors(len(df))
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["model"], df["post_prob"], color=colors)
    ax.set_ylabel("posterior model probability")
    ax.set_title("Bayesian model comparison (cross-sectional dependence)",
                 fontweight="bold")
    ax.grid(alpha=0.3, axis="y")
    return ax


def plot_correlation_heatmap(result, ax=None, cmap="Parula"):
    """Heatmap of the estimated cross-sectional correlation matrix."""
    import matplotlib.pyplot as plt

    corr = result.corr_post
    names = result.names
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap=_mpl_cmap(cmap), vmin=corr.min(), vmax=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    color="white" if corr[i, j] < 0.6 else "black", fontsize=7)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Cross-sectional correlation (Parula)", fontweight="bold")
    return ax


# --------------------------------------------------------------------------- #
# Paper 4 - rank posterior heatmap                                            #
# --------------------------------------------------------------------------- #
def plot_rank_posterior(result, ax=None, cmap="Parula"):
    """Heatmap of cointegrating-rank posterior probabilities per unit."""
    import matplotlib.pyplot as plt

    df = result.rank_frame().set_index("unit")
    prob_cols = [c for c in df.columns if c.startswith("P(r=")]
    mat = df[prob_cols].values.astype(float)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.8 * len(df) + 2))
    im = ax.imshow(mat, cmap=_mpl_cmap(cmap), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(prob_cols)))
    ax.set_xticklabels(prob_cols)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] < 0.5 else "black", fontsize=8)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Cointegrating-rank posterior probabilities", fontweight="bold")
    return ax
