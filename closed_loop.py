# Closed-loop multi-objective Bayesian optimization of mycelium composites.
# Usage:  from closed_loop import ClosedLoop
#         loop = ClosedLoop(data_file='compositions.csv', upto_round=3)
#         loop.evaluate()               # Pareto front size and hypervolume per round
#         loop.propose_experiments()    # next 4 compositions, objectives P, G, S
#         from cnet_monte_carlo import cnet_torch
#         loop.propose_experiments(cnet_fn=cnet_torch)    # four objectives, Cnet at nominal parameters
# data_file: one row per composition with the nine ingredients (mass %), round, P, G, S, Cnet.
import numpy as np
import pandas as pd
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP, ModelListGP
from botorch.models.transforms import Normalize, Standardize
from botorch.models.utils.gpytorch_modules import (
    get_matern_kernel_with_gamma_prior, get_gaussian_likelihood_with_gamma_prior)
from botorch.acquisition.multi_objective.monte_carlo import qNoisyExpectedHypervolumeImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood

torch.set_default_dtype(torch.float64)

INGREDIENTS = ['Soil', 'Local_waste', 'Clay', 'Malt', 'Agar',
               'Peptone', 'Xanthan_gum', 'Glycerol', 'Water']
LOWER = np.array([0, 0, 0, 0, 0, 0, 0, 0, 30.])
UPPER = np.array([50, 50, 10, 5, 5, 0.5, 5, 5, 70.])


class ClosedLoop:
    def __init__(self, data_file='data/compositions.csv', upto_round=None, seed=42):
        # one row per composition: P, G, S are replicate means, Cnet is one value per composition
        df = pd.read_csv(data_file)
        if upto_round is not None:
            df = df[df['round'] <= upto_round]
        assert len(df) > 0 and np.isfinite(df[INGREDIENTS + ['P', 'G', 'S']].values).all(), 'invalid data'
        self.df = df
        self.X = df[INGREDIENTS].values
        self.seed = seed
        self.bounds = torch.tensor(np.stack([LOWER, UPPER]))

    def fit_model(self, X, Y):
        models = []
        for j in range(Y.shape[1]):
            gp = SingleTaskGP(torch.tensor(X), torch.tensor(Y[:, [j]]),
                              covar_module=get_matern_kernel_with_gamma_prior(X.shape[1]),
                              likelihood=get_gaussian_likelihood_with_gamma_prior(),
                              input_transform=Normalize(X.shape[1], bounds=self.bounds),
                              outcome_transform=Standardize(1))
            fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
            models.append(gp)
        return ModelListGP(*models)

    def loo_r2(self, col):
        y = self.df[col].values
        pred = np.zeros(len(y))
        for i in range(len(y)):
            k = np.arange(len(y)) != i
            model = self.fit_model(self.X[k], y[k, None])
            with torch.no_grad():
                pred[i] = model.models[0].posterior(torch.tensor(self.X[[i]])).mean.item()
        return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)

    def propose_experiments(self, n_experiments=4, cnet_fn=None):
        # cnet_fn(X, G): deterministic Cnet, if None only P, G, S are optimised
        torch.manual_seed(self.seed)
        model = self.fit_model(self.X, self.df[['P', 'G', 'S']].values)
        ref, objective = [0., 0., 0.], None
        if cnet_fn is not None:
            ref = ref + [-1000.]
            objective = cnet_objective(cnet_fn)
        acq = qNoisyExpectedHypervolumeImprovement(
            model=model, ref_point=ref, X_baseline=torch.tensor(self.X),
            sampler=SobolQMCNormalSampler(torch.Size([128]), seed=self.seed),
            objective=objective, prune_baseline=True, cache_root=False)
        eq = [(torch.arange(9), torch.ones(9, dtype=torch.float64), 100.)]
        cand, _ = optimize_acqf(acq, bounds=self.bounds, q=n_experiments, num_restarts=20,
                                raw_samples=512, equality_constraints=eq, sequential=True,
                                options={'seed': self.seed, 'batch_limit': 5, 'maxiter': 200})
        cand = np.array([round_feasible(c) for c in cand.detach().numpy()])
        return pd.DataFrame(cand, columns=INGREDIENTS)

    def evaluate(self):
        # Pareto front size and hypervolume after each round (objectives P, G, S, -Cnet)
        # reference point is worse than every observed value: P, G, S > 0 and Cnet < 1000 kg CO2-eq/m3
        ref = np.array([0, 0, 0, -1000.])
        out = []
        for r in sorted(self.df['round'].unique()):
            d = self.df[self.df['round'] <= r]
            Y = np.c_[d['P'], d['G'], d['S'], -d['Cnet']]
            Y = Y[(Y > ref).all(1)]
            front = pareto_mask(Y)
            out.append([r, front.sum(), hypervolume(Y[front], ref)])
        out = pd.DataFrame(out, columns=['round', 'front_size', 'hypervolume'])
        out['gain_pct'] = (out['hypervolume'] / out['hypervolume'][0] - 1) * 100
        return out


def round_feasible(c, decimals=2):
    # round to weighing precision, water balances to 100; if water leaves its bounds,
    # the difference is moved to the largest other ingredient that can take it
    c = np.array(c, float)
    c[:8] = np.clip(np.round(c[:8], decimals), LOWER[:8], UPPER[:8])
    c[8] = np.round(100 - c[:8].sum(), decimals)
    for _ in range(20):
        excess = np.clip(c[8], LOWER[8], UPPER[8]) - c[8]
        if abs(excess) < 1e-9:
            break
        room = (UPPER[:8] - c[:8]) if excess < 0 else (c[:8] - LOWER[:8])
        j = room.argmax()
        step = np.sign(-excess) * min(abs(excess), room[j])
        c[j] += step
        c[8] = np.round(100 - c[:8].sum(), decimals)
    assert (c >= LOWER - 1e-9).all() and (c <= UPPER + 1e-9).all() and abs(c.sum() - 100) < 1e-6, 'infeasible candidate'
    return c


def cnet_objective(cnet_fn):
    from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective

    class Objective(MCMultiOutputObjective):
        def forward(self, samples, X=None):
            Xe = X.expand(*samples.shape[:-1], X.shape[-1])
            cnet = cnet_fn(Xe, samples[..., 1])
            return torch.cat([samples[..., :3], -cnet.unsqueeze(-1)], dim=-1)
    return Objective()


def pareto_mask(Y):
    keep = np.ones(len(Y), bool)
    for i in range(len(Y)):
        if ((Y >= Y[i]).all(1) & (Y > Y[i]).any(1)).any():
            keep[i] = False
    return keep


def hypervolume(Y, ref):
    # exact, by slicing along the last objective (all objectives maximised)
    if len(Y) == 0:
        return 0.
    if Y.shape[1] == 1:
        return Y[:, 0].max() - ref[0]
    Y = Y[np.argsort(-Y[:, -1])]
    hv = 0.
    for i in range(len(Y)):
        nxt = Y[i + 1, -1] if i + 1 < len(Y) else ref[-1]
        if Y[i, -1] > nxt:
            hv += (Y[i, -1] - nxt) * hypervolume(Y[:i + 1, :-1], ref[:-1])
    return hv
