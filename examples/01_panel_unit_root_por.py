"""
Example 1 - Bayesian panel unit-root test via the Posterior Odds Ratio.

Kumar, Chaturvedi & Afifa (2016).

Run:  python examples/01_panel_unit_root_por.py
"""
import matplotlib.pyplot as plt

import pybayescointur as pbc

# ------------------------------------------------------------------ #
# 1. Simulate a near-unit-root trend-stationary panel (rho = 0.92).
# ------------------------------------------------------------------ #
panel = pbc.simulate_par_panel(n=4, T=80, rho=0.92, seed=1)
print(panel.head(), "\n")

# ------------------------------------------------------------------ #
# 2. Test with a linear trend, then with augmentation terms.
# ------------------------------------------------------------------ #
res_trend = pbc.bayesian_panel_unit_root(panel, model="trend")
res_aug = pbc.bayesian_panel_unit_root(panel, model="augmented", k=2)

print(res_trend, "\n")
print(res_aug, "\n")

# Pretty table (rich) + DataFrame for export.
pbc.por_table(res_aug)
res_aug.to_frame().to_csv("por_result.csv", index=False)

# ------------------------------------------------------------------ #
# 3. How sensitive is the conclusion to the prior P(H0)?
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(8, 5))
pbc.plot_por_sensitivity(panel, model="augmented", k=2, ax=ax)
fig.tight_layout()
fig.savefig("example1_por_sensitivity.png", dpi=120)
print("\nSaved figure -> example1_por_sensitivity.png")
