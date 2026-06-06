"""
Example 3 - Bayesian model comparison under cross-sectional dependence.

Meligkotsidou, Tzavalis & Vrontos.

Run:  python examples/03_cross_sectional_dependence.py
"""
import matplotlib.pyplot as plt

import pybayescointur as pbc

# ------------------------------------------------------------------ #
# 1. A synthetic, cross-sectionally dependent G7-style log-GDP panel.
# ------------------------------------------------------------------ #
gdp = pbc.load_g7_like_gdp()
print(gdp.head(), "\n")

# ------------------------------------------------------------------ #
# 2. Compare the 8 competing models (stationary / RW, trend / no
#    trend, independent / cross-sectionally dependent errors).
# ------------------------------------------------------------------ #
res = pbc.bayesian_csd_comparison(gdp)
print(res, "\n")

pbc.csd_table(res)

print("\nEstimated cross-sectional correlation matrix:")
print(res.correlation_frame().round(2).to_string())

# ------------------------------------------------------------------ #
# 3. Figures: posterior model probabilities + correlation heatmap.
# ------------------------------------------------------------------ #
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
pbc.plot_csd_probabilities(res, ax=axes[0])
pbc.plot_correlation_heatmap(res, ax=axes[1])   # Parula colormap
fig.tight_layout()
fig.savefig("example3_csd.png", dpi=120)
print("\nSaved figure -> example3_csd.png")
