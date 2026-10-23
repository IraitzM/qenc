"""Controlled-rank regressions and restarts (main text Section 3.2, SI S10).

Targets of exact rank 1, 2, 4, 6 and 9 and bandwidth 4 on a 12x12 grid (six target
draws) are fitted by the two-qubit ladder with k = 0..3 CRZ entanglers at depth 4,
from 64 initializations each (three LBFGS rounds of 150 iterations). The reach of a cell
is the share of starts that finish within 0.01 of the best fit any start found.
The 7680 fits run in parallel processes.

    uv run experiments/rank_restarts.py --all
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch

from qenc.rank import EntanglerLadder, controlled_rank_target, model_ceiling_r2, recommended_placement
from qenc.utils import bundle, cli

OUT = bundle("rank")
DEGREE, BANDWIDTH, N_GRID = 4, 4, 12
RANKS, BUDGETS, TARGET_SEEDS, N_INITS = (1, 2, 4, 6, 9), (0, 1, 2, 3), 6, 64
LBFGS_ROUNDS, LBFGS_ITERS = 3, 150


def fit(spec):
    target_rank, target_seed, k, init = spec
    torch.set_num_threads(1)
    axis = np.arange(N_GRID) * 2 * np.pi / N_GRID
    x_grid, y_grid = np.meshgrid(axis, axis, indexing="ij")
    xs, ys = (torch.tensor(a.ravel(), dtype=torch.float64) for a in (x_grid, y_grid))
    target = controlled_rank_target(target_rank, BANDWIDTH, N_GRID, rng=np.random.default_rng(1000 + target_seed))
    flat = torch.tensor(target.ravel(), dtype=torch.float64)
    torch.manual_seed(10_000 * k + 100 * target_seed + init)
    model = EntanglerLadder(DEGREE, recommended_placement(k, DEGREE))
    for _ in range(LBFGS_ROUNDS):
        opt = torch.optim.LBFGS(model.parameters(), max_iter=LBFGS_ITERS, history_size=50,
                                line_search_fn="strong_wolfe", tolerance_grad=1e-12, tolerance_change=1e-14)

        def closure():
            opt.zero_grad()
            loss = ((model(xs, ys) - flat) ** 2).mean()
            loss.backward()
            return loss

        opt.step(closure)
    with torch.no_grad():
        pred = model(xs, ys).numpy()
    r2 = 1.0 - (float(np.sum((target.ravel() - pred) ** 2))
                / (float(np.sum((target.ravel() - target.mean()) ** 2)) + 1e-12))
    return dict(target_rank=target_rank, target_seed=target_seed, k=k, init=init, r2=r2,
                ceiling=model_ceiling_r2(target, k, DEGREE))


def run(job):
    if (OUT / "fits.csv").exists():
        print(OUT / "fits.csv", "exists, skipping")
        return
    specs = [(r, s, k, i) for r in RANKS for s in range(TARGET_SEEDS) for k in BUDGETS for i in range(N_INITS)]
    workers = max(1, min(len(os.sched_getaffinity(0)), 18))
    print(f"{len(specs)} fits over {workers} processes", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(fit, specs, chunksize=1))
    pd.DataFrame(rows).to_csv(OUT / "fits.csv", index=False)


if __name__ == "__main__":
    cli(["fits"], run, __doc__)
