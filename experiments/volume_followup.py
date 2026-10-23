"""Volume follow-up (SI S13): validation selection among three encodings of NoduleMNIST3D.

Same volume draws as the five-task study (seeds 52-56, 225 fit / 75 validation / 120
test). Candidates: Angle-8 and amplitude on the eight 2^3 block means, and amplitude on
the 64 4^3 block means, ten epochs each. Validation picks one; test scores follow.

    uv run experiments/volume_followup.py --list | --job 52 | --all
"""
from __future__ import annotations

import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from qenc.pooling import features
from qenc.protocols import FIVE_TASK_SEEDS, fit_five_task, five_task_data, predict_five_task
from qenc.utils import bundle, cli, write_json

OUT = bundle("volume-followup")
JOBS = [str(seed) for seed in FIVE_TASK_SEEDS]
SPECS = [dict(candidate="angle8", kind="angle", spatial_grid=(2, 2, 2), grid=(8,), n_qubits=8, depth=4, epochs=10,
              lr=.05, batch=32),
         dict(candidate="amplitude8", kind="amplitude", spatial_grid=(2, 2, 2), grid=None, n_qubits=3, depth=3,
              epochs=10, lr=.05, batch=32),
         dict(candidate="amplitude64", kind="amplitude", spatial_grid=(4, 4, 4), grid=None, n_qubits=6, depth=3,
              epochs=10, lr=.05, batch=32)]


def run(job):
    seed, folder = int(job), OUT / job
    if (folder / "complete.json").exists():
        return
    folder.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    d = five_task_data("NoduleMNIST3D", seed)
    fit, val = d["fit"], d["val"]
    rows, fitted = [], {}
    for spec in SPECS:
        x = features(d["x"], d["shape"], spec["spatial_grid"])
        model = fit_five_task(x[fit], d["y"][fit], (x.shape[1],), spec, seed)
        pred = predict_five_task(model, x[val], (x.shape[1],), spec)
        rows.append(dict(spec, features=x.shape[1], val_bacc=float(balanced_accuracy_score(d["y"][val], pred))))
        fitted[spec["candidate"]] = model
        print(job, spec["candidate"], "validation", round(rows[-1]["val_bacc"], 4), flush=True)
    selected = min(rows, key=lambda r: (-r["val_bacc"], r["n_qubits"], r["depth"], r["candidate"]))["candidate"]
    for row, spec in zip(rows, SPECS):
        xt = features(d["xt"], d["shape"], spec["spatial_grid"])
        pred = predict_five_task(fitted[spec["candidate"]], xt, (xt.shape[1],), spec)
        row.update(test_bacc=float(balanced_accuracy_score(d["yt"], pred)), selected=spec["candidate"] == selected)
    pd.DataFrame(rows).assign(seed=seed).to_csv(folder / "scores.csv", index=False)
    write_json(folder / "complete.json", dict(status="complete", selected=selected))


if __name__ == "__main__":
    cli(JOBS, run, __doc__)
