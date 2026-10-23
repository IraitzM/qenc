"""Spatial pilot (SI S1, S3): the three encodings on PneumoniaMNIST and NoduleMNIST3D grids.

Seeds 62-66; 180 fit / 60 validation / 120 test examples, z-scored, then block means on
a coarse (2x4, 2^3) and a fine (4x4, 4^3) grid. Candidates: QSM on the native trajectory
matrix (four uploads, t = 2) and amplitude on each grid, and Angle-8 on the coarse grid,
all trained for 20 epochs. An RBF-SVM on each grid, gamma chosen on validation, is the
classical reference. The saved models feed the probes and the ladder.

    uv run experiments/spatial_pilot.py --list | --job PneumoniaMNIST-62 | --all
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qenc.data import spatial_draw
from qenc.pooling import features
from qenc.protocols import SPATIAL_SEEDS, SPATIAL_TASKS, spatial_grids, spatial_input, spatial_model, spatial_specs
from qenc.utils import bundle, cli, digest, write_json

OUT = bundle("spatial-pilot")
JOBS = [f"{name}-{seed}" for name in SPATIAL_TASKS for seed in SPATIAL_SEEDS]
EPOCHS = 20


def predict(model, x):
    with torch.no_grad():
        return torch.cat([model(part) for part in x.split(32)]).numpy().argmax(1)


def rbf(d, name):
    rows = []
    for grid in spatial_grids(name):
        z, zt, best = features(d["x"], d["shape"], grid), features(d["xt"], d["shape"], grid), None
        for gamma in np.logspace(-3, 3, 11) / z.shape[1]:
            model = make_pipeline(StandardScaler(), SVC(C=1, gamma=float(gamma))).fit(z[d["fit"]], d["y"][d["fit"]])
            score = float(balanced_accuracy_score(d["y"][d["val"]], model.predict(z[d["val"]])))
            if best is None or score > best[0]:
                best = (score, float(gamma), model)
        rows.append(dict(candidate=f"grid{int(np.prod(grid))}", grid=grid, gamma=best[1], val_bacc=best[0],
                         test_bacc=float(balanced_accuracy_score(d["yt"], best[2].predict(zt)))))
    return rows


def run(job):
    name, seed = job.rsplit("-", 1)
    seed, folder = int(seed), OUT / job
    if (folder / "complete.json").exists():
        return
    folder.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    d = spatial_draw(name, seed)
    fit, val = d["fit"], d["val"]
    np.savez_compressed(folder / "split.npz", fit=fit, val=val, y=d["y"], yt=d["yt"], train_ids=d["train_ids"],
                        test_ids=d["test_ids"])
    write_json(folder / "inputs.json", dict(train_sha256=digest(d["x"]), test_sha256=digest(d["xt"]), shape=d["shape"]))
    yy, rows = torch.tensor(d["y"], dtype=torch.long), []
    for spec in spatial_specs(name):
        start, xx = time.monotonic(), spatial_input(d["x"], d["shape"], spec)
        model = spatial_model(spec, seed)
        opt = torch.optim.Adam(model.parameters(), lr=spec["lr"])
        for _ in range(EPOCHS):
            order = fit[torch.randperm(len(fit)).numpy()]
            for begin in range(0, len(fit), spec["batch"]):
                ix = order[begin:begin + spec["batch"]]
                opt.zero_grad()
                torch.nn.functional.cross_entropy(model(xx[ix]), yy[ix]).backward()
                opt.step()
        pred_val = predict(model, xx[val])
        pred_test = predict(model, spatial_input(d["xt"], d["shape"], spec))
        torch.save(model.state_dict(), folder / f"{spec['candidate']}.pt")
        np.save(folder / f"{spec['candidate']}-validation.npy", pred_val)
        np.save(folder / f"{spec['candidate']}-test.npy", pred_test)
        rows.append(dict(spec, epochs=EPOCHS, val_bacc=float(balanced_accuracy_score(d["y"][val], pred_val)),
                         test_bacc=float(balanced_accuracy_score(d["yt"], pred_test)),
                         parameters=sum(p.numel() for p in model.parameters()),
                         fit_validation_seconds=time.monotonic() - start))
        print(job, spec["candidate"], "test", round(rows[-1]["test_bacc"], 4), flush=True)
    pd.DataFrame(rows).assign(dataset=name, seed=seed).to_csv(folder / "scores.csv", index=False)
    pd.DataFrame(rbf(d, name)).assign(dataset=name, seed=seed).to_csv(folder / "classical.csv", index=False)
    write_json(folder / "complete.json", dict(status="complete"))


if __name__ == "__main__":
    cli(JOBS, run, __doc__)
