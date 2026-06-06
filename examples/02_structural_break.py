"""
Example 2 - Panel unit-root test with a structural break in mean and variance.

Kumar & Agiwal (2019).

Run:  python examples/02_structural_break.py
"""
import matplotlib.pyplot as plt

import pybayescointur as pbc

# ------------------------------------------------------------------ #
# 1. Simulate a PAR(1) panel with a break in mean and variance at TB.
# ------------------------------------------------------------------ #
TB = 15
panel = pbc.simulate_break_panel(
    n=3, T=25, TB=TB, rho=0.95, lam=5.0,
    mu1=(30, 40, 50), mu2=(300, 400, 500), seed=0,
)

# ------------------------------------------------------------------ #
# 2. Compute the 14 posterior odds ratios.
# ------------------------------------------------------------------ #
res = pbc.bayesian_break_unit_root(panel, TB=TB)
print(res, "\n")

# Tidy table of all 14 PORs (rich-highlighted).
pbc.break_table(res)
res.to_frame().to_csv("break_pors.csv", index=False)

# ------------------------------------------------------------------ #
# 3. Visualise the decisions (log10 POR; < 0 means reject the null).
# ------------------------------------------------------------------ #
fig, ax = plt.subplots(figsize=(11, 5))
pbc.plot_break_por(res, ax=ax)
fig.tight_layout()
fig.savefig("example2_break_por.png", dpi=120)
print("\nSaved figure -> example2_break_por.png")

# ------------------------------------------------------------------ #
# 4. Search the break point (e.g. between periods 8 and 20).
# ------------------------------------------------------------------ #
print("\nPOR4 (DS-no-break vs TS-break-both) across candidate break points:")
for tb in range(8, 21):
    r = pbc.bayesian_break_unit_root(panel, TB=tb, n_rho=120, n_lam=100)
    print(f"  TB={tb:>2}  POR4 = {r.por[4]:.3e}")
