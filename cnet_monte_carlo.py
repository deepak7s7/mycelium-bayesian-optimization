# Monte Carlo propagation of parameter uncertainty through the Cnet model (SI Note 5).
# Usage:  from cnet_monte_carlo import CnetMonteCarlo
#         CnetMonteCarlo(n=10000, seed=42).run(df, scenario='B')
# df: one row per composition with comp_id, the nine ingredients (mass %) and G.
import numpy as np
import pandas as pd
import torch

INGREDIENTS = ['Soil', 'Local_waste', 'Clay', 'Malt', 'Agar',
               'Peptone', 'Xanthan_gum', 'Glycerol', 'Water']

# SI Table 2: cradle-to-gate emission factors (kg CO2-eq/kg) and sigma_log (SI Table 4)
MU = np.array([0.020, 0.085, 0.090, 0.850, 2.400, 8.500, 4.700, 1.200, 0.0004])
SIGMA = np.array([0.15, 0.25, 0.15, 0.25, 0.40, 0.40, 0.40, 0.25, 0.15])
# SI Table 3: manufacturing steps (kg CO2-eq/kg dry)
MFG = np.array([0.020, 0.050, 0.060, 0.020, 0.010, 0.100])
# true densities (g/cm3) of the dry ingredients; glycerol and water do not enter the solid phase
RHO = np.array([2.65, 0.32, 2.50, 1.35, 1.30, 1.27, 1.50])
RHO_MYC, F_MYC = 1.20, 0.07


class CnetMonteCarlo:
    def __init__(self, n=10000, seed=42, f_c=0.415, kappa=0.279, grid=0.412,
                 sec=1.0, f_evap=0.20, mu_dry_b=0.060):
        self.n, self.f_c, self.kappa, self.grid = n, f_c, kappa, grid
        self.sec, self.f_evap, self.mu_dry_b = sec, f_evap, mu_dry_b
        self.rng = np.random.default_rng(seed)

    def lognormal(self, median, sigma, size):
        return median * np.exp(sigma * self.rng.standard_normal(size))

    def density(self, w):
        # dry density (kg/m3): rule of mixtures on solids, 7 % mycelium, porosity correction
        solid = w[:7] / w[:7].sum() * (1 - F_MYC)
        rho_solid = 1 / (np.sum(solid / RHO) + F_MYC / RHO_MYC)
        phi = np.clip(0.30 + 0.45 * w[8] + 0.20 * w[1], 0.30, 0.70)
        return (1 - phi) * rho_solid * 1000

    def cnet(self, comp, G, scenario='A'):
        # comp: mass % of the nine ingredients, G: growth (% hyphal coverage)
        w = np.asarray(comp, float) / 100
        assert np.isfinite(w).all() and (w >= 0).all() and abs(w.sum() - 1) < 0.01, 'invalid composition'
        assert scenario in ('A', 'B') and np.isfinite(G) and G >= 0, 'invalid scenario or G'
        n = self.n
        mu = self.lognormal(MU, SIGMA, (n, 9))
        mfg = self.lognormal(MFG, 0.40, (n, 6)).sum(1)
        f_c = self.lognormal(self.f_c, 0.10, n)
        kappa = self.lognormal(self.kappa, 0.30, n)
        if scenario == 'A':
            sec = self.lognormal(self.sec, 0.20, n)
            dry = sec * w[8] * (1 - self.f_evap) / (1 - w[8]) * self.grid
        else:
            dry = self.lognormal(self.mu_dry_b, 0.20, n)
        per_kg = ((mu * w).sum(1) / (1 - w[8]) + mfg + kappa * G * 1e-3 + dry
                  - 44 / 12 * w[1] * f_c / (1 - w[8]))
        return per_kg, per_kg * self.density(w)

    def run(self, df, scenario='A'):
        rows = []
        for _, r in df.iterrows():
            kg, m3 = self.cnet(r[INGREDIENTS].values, r['G'], scenario)
            rows.append([r['comp_id'], np.median(kg), *np.percentile(kg, [2.5, 97.5]),
                         np.median(m3), *np.percentile(m3, [2.5, 97.5])])
        return pd.DataFrame(rows, columns=['comp_id', 'Cnet_kg_median', 'Cnet_kg_p2.5', 'Cnet_kg_p97.5',
                                           'Cnet_m3_median', 'Cnet_m3_p2.5', 'Cnet_m3_p97.5'])


def cnet_torch(X, G, scenario='B', f_c=0.415, kappa=0.279, grid=0.412, sec=1.0, f_evap=0.20, mu_dry_b=0.060):
    # Deterministic Cnet (kg CO2-eq/m3) at nominal parameters, differentiable, for the optimizer.
    # X: (..., 9) mass %, G: (...) growth. Same model as CnetMonteCarlo with all parameters at their medians.
    w = X / 100
    mu = torch.tensor(MU, dtype=X.dtype)
    rho_s = torch.tensor(RHO, dtype=X.dtype)
    w_water = w[..., 8]
    dry_w = (1 - w_water).clamp_min(1e-6)
    if scenario == 'A':
        dry = sec * w_water * (1 - f_evap) / dry_w * grid
    else:
        dry = mu_dry_b
    per_kg = ((w * mu).sum(-1) / dry_w + MFG.sum() + kappa * G * 1e-3 + dry
              - 44 / 12 * w[..., 1] * f_c / dry_w)
    solid = w[..., :7] / w[..., :7].sum(-1, keepdim=True).clamp_min(1e-9) * (1 - F_MYC)
    rho_solid = 1 / ((solid / rho_s).sum(-1) + F_MYC / RHO_MYC)
    phi = torch.clamp(0.30 + 0.45 * w_water + 0.20 * w[..., 1], 0.30, 0.70)
    return per_kg * (1 - phi) * rho_solid * 1000
