# Bayesian optimization of mycelium composites

Code for "Closed-Loop Bayesian Optimization for the Design of Printable, Carbon-Negative, High-Strength Mycelium Composites".

closed_loop.py: fits Gaussian process models to printability, growth and specific strength, suggests the next four compositions, and reports Pareto front size and hypervolume per round.

cnet_monte_carlo.py: net carbon footprint model from SI Note 5 with Monte Carlo uncertainty (10,000 samples, seed 42).

Requires Python 3, numpy, pandas, scipy, torch and botorch 0.18.1.

    from closed_loop import ClosedLoop
    print(ClosedLoop(data_file='compositions.csv').evaluate())

This should give Pareto fronts of 10, 12, 10, 11 and 14 compositions (Fig. 4C).

The rounds in the paper were run step by step in the lab, so rerunning the proposal step gives valid compositions from the same method, not the exact tested ones.
