"""Retained detail at fixed capacity (main text Section 3.1, SI S7).

Same draws as the spatial pilot (seeds 62-66). F is the fine grid (4x4 or 4^3 block
means) and PF the coarse grid (2x4 or 2^3) repeated back onto F's shape, so both enter
the same circuit. Within each family, F and PF start from the same weights and see the
same minibatch order; Delta_detail = BA(F) - BA(PF) on the test split.

    uv run experiments/detail.py --list | --job PneumoniaMNIST-62 | --all
"""
from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from qenc.data import spatial_draw
from qenc.models import AmplitudeClassifier, QSMClassifier, block_upload, required_qubits, trajectory_matrix
from qenc.pooling import expanded_coarse, features
from qenc.protocols import SPATIAL_SEEDS, SPATIAL_TASKS
from qenc.utils import bundle, cli, write_json

OUT = bundle("detail")
JOBS = [f"{name}-{seed}" for name in SPATIAL_TASKS for seed in SPATIAL_SEEDS]
FAMILIES = (("qsm", .005), ("amplitude", .05))


def grids(name):
    return ((4, 4), (2, 4)) if name == "PneumoniaMNIST" else ((4, 4, 4), (2, 2, 2))


def build(family, grid):
    if family == "qsm":
        return QSMClassifier(required_qubits(grid), n_uploads=4, batched=True)
    return AmplitudeClassifier(int(np.prod(grid)), n_layers=3)


def model_input(grid_values, family, geometry):
    if family == "qsm":
        upload = block_upload(trajectory_matrix(grid_values, geometry), time=2.0, n_qubits=required_qubits(geometry))
        return torch.tensor(upload, dtype=torch.complex128)
    return torch.tensor(grid_values.reshape(len(grid_values), -1), dtype=torch.float64)


def train(model, x, y, fit, lr, order_seed, epochs=20, batch=32):
    opt, yy = torch.optim.Adam(model.parameters(), lr=lr), torch.tensor(y, dtype=torch.long)
    generator = torch.Generator().manual_seed(order_seed)
    for _ in range(epochs):
        order = fit[torch.randperm(len(fit), generator=generator).numpy()]
        for start in range(0, len(order), batch):
            ix = order[start:start + batch]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(x[ix]), yy[ix]).backward()
            opt.step()


def predict(model, x):
    with torch.no_grad():
        return torch.cat([model(part) for part in x.split(32)]).numpy().argmax(1)


def run(job):
    name, seed = job.rsplit("-", 1)
    seed, folder = int(seed), OUT / job
    if (folder / "complete.json").exists():
        return
    folder.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    d = spatial_draw(name, seed)
    fine_grid, coarse_grid = grids(name)
    arms = {}
    for split, x in (("train", d["x"]), ("test", d["xt"])):
        fine = features(x, d["shape"], fine_grid).reshape(-1, *fine_grid)
        arms[split] = {"F": fine, "PF": expanded_coarse(fine, fine_grid, coarse_grid)}
    rows = []
    for family, lr in FAMILIES:
        torch.manual_seed(seed)
        initial = copy.deepcopy(build(family, fine_grid).state_dict())
        for arm in ("F", "PF"):
            start, model = time.monotonic(), build(family, fine_grid)
            model.load_state_dict(initial)
            x = model_input(arms["train"][arm], family, fine_grid)
            train(model, x, d["y"], d["fit"], lr, order_seed=seed + 1000)
            pred_val = predict(model, x[d["val"]])
            pred_test = predict(model, model_input(arms["test"][arm], family, fine_grid))
            rows.append(dict(candidate=f"{family}-{arm}", family=family, arm=arm, n_qubits=model.n_qubits,
                             parameters=sum(p.numel() for p in model.parameters()),
                             val_bacc=float(balanced_accuracy_score(d["y"][d["val"]], pred_val)),
                             test_bacc=float(balanced_accuracy_score(d["yt"], pred_test)),
                             seconds=time.monotonic() - start))
            print(job, rows[-1]["candidate"], "test", round(rows[-1]["test_bacc"], 4), flush=True)
    pd.DataFrame(rows).to_csv(folder / "quantum_scores.csv", index=False)
    write_json(folder / "complete.json", dict(status="complete"))


if __name__ == "__main__":
    cli(JOBS, run, __doc__)
