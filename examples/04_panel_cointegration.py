"""
Example 4 - Bayesian inference in a cointegrating panel data model.

Koop, Leon-Gonzalez & Strachan (2006).

Run:  python examples/04_panel_cointegration.py
"""
import numpy as np
import matplotlib.pyplot as plt

import pybayescointur as pbc

# ------------------------------------------------------------------ #
# 1. Simulate a panel of VECMs with heterogeneous cointegrating rank.
#    unit 1: rank 1, unit 2: rank 0 (no cointegration), unit 3: rank 1.
# ------------------------------------------------------------------ #
panels = pbc.simulate_vecm_panel(N=3, n=2, T=150, ranks=[1, 0, 1], seed=3)

# ------------------------------------------------------------------ #
# 2. Infer each unit's cointegrating rank via Savage-Dickey Bayes
#    factors, plus a common-rank posterior.
# ------------------------------------------------------------------ #
res = pbc.bayesian_panel_cointegration(
    panels, lags=1, deterministic="c", draws=1500, burn=300, seed=7,
)
print(res, "\n")

pbc.coint_rank_table(res)

# ------------------------------------------------------------------ #
# 3. The recovered cointegrating vector of unit 1 (~ a great ratio).
# ------------------------------------------------------------------ #
u1 = res.units[0]
if u1.beta is not None:
    print("\nUnit 1 cointegrating vector (normalised):")
    print(np.round(u1.beta, 3))

# ------------------------------------------------------------------ #
# 4. Heatmap of the rank posterior probabilities.
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(8, 4))
pbc.plot_rank_posterior(res, ax=ax)
fig.tight_layout()
fig.savefig("example4_coint_rank.png", dpi=120)
print("\nSaved figure -> example4_coint_rank.png")
