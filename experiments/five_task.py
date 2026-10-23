"""Five-task study (SI S1, S3, S6, S13): the three encodings on ECG, MNIST1D, EEG and
PneumoniaMNIST, and the classical pooling screen on all five tasks.

Each job is one task and seed (52-56). Every candidate is fitted on the fitting split,
the complete pipeline is selected on validation (exact ties by qubits, depth, name), and
each model is scored once on the test split. The historical threshold rule is recorded
for the policy comparison. Then an RBF-SVM probe, tuned by nested cross-validation on the
training draw, scores every pooled representation; G_A is its full-minus-compressed gap.
NoduleMNIST3D only runs the classical probe.

    uv run experiments/five_task.py --list | --job ECG-52 | --all
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qenc.pooling import features
from qenc.protocols import (
    FIVE_TASK_SEEDS,
    FIVE_TASKS,
    fit_five_task,
    five_task_candidates,
    five_task_data,
    predict_five_task,
)
from qenc.utils import bundle, cli, digest, write_json

OUT = bundle("five-task")
JOBS = [f"{name}-{seed}" for name in FIVE_TASKS for seed in FIVE_TASK_SEEDS]


# ---------------------------------------------------------------- historical threshold rule

def signal_diagnostics(x, n_pooled=8, eps=0.05):
    """Median omega_max and mean pooled spectral energy loss over the first 50 z-scored signals."""
    omega_max, delta_e = [], []
    for signal in x[:50]:
        z = (signal - signal.mean()) / (signal.std() + 1e-8)
        power = np.abs(np.fft.rfft(z)) ** 2
        total = power.sum() + 1e-12
        cumulative, selected = 0.0, set()
        for idx in np.argsort(power[1:])[::-1] + 1:  # largest bins first, DC excluded
            cumulative += power[idx] / total
            selected.add(idx)
            if cumulative >= 1.0 - eps:
                break
        omega_max.append(max(selected) if selected else 0)
        delta_e.append(float(power[n_pooled // 2 + 1:].sum() / total) if len(signal) > n_pooled else 0.0)
    return int(np.median(omega_max)), float(np.mean(delta_e))


def image_diagnostics(x, shape=(28, 28), grid=(2, 4), fraction=0.95):
    """Native-2D counterparts over the first 50 images.

    omega_max of the top-95% power set, and the energy outside the grid's band.
    """
    images = np.asarray(x, dtype=float)[:50].reshape(-1, *shape)
    z = images - images.mean(axis=(1, 2), keepdims=True)
    power = abs(np.fft.fft2(z)) ** 2
    power /= power.sum(axis=(1, 2), keepdims=True)
    fy, fx = [np.rint(np.fft.fftfreq(d) * d).astype(int) for d in shape]
    ky, kx = np.meshgrid(fy, fx, indexing="ij")
    inside = (abs(ky) <= grid[0] // 2) & (abs(kx) <= grid[1] // 2)
    shell = np.maximum(abs(ky), abs(kx)).ravel()
    maximum = []
    for p in power:
        order = np.argsort(p.ravel(), kind="stable")[::-1]
        end = int(np.searchsorted(np.cumsum(p.ravel()[order]), fraction)) + 1
        maximum.append(int(shell[order[:end]].max()))
    return int(np.median(maximum)), float(power[:, ~inside].sum(axis=1).mean())


def historical_rule(x, shape):
    """The earlier threshold selector: angle when pooling keeps 95% of the energy, else QSM for
    spectrally hard signals it can run (d <= 256), else amplitude."""
    omega_max, delta_e = image_diagnostics(x) if len(shape) == 2 else signal_diagnostics(x)
    if delta_e < 0.05:
        return "angle"
    return "qsm" if omega_max > 8 and x.shape[1] <= 256 else "amplitude"


# ---------------------------------------------------------------- quantum comparison

def select(rows):
    return min(rows, key=lambda r: (-r["val_bacc"], r["n_qubits"], r["depth"], r["candidate"]))["candidate"]


def quantum(d, folder, seed):
    x, y, xt, yt, shape, fit, val = (d[k] for k in ("x", "y", "xt", "yt", "shape", "fit", "val"))
    specs, models, rows = five_task_candidates(shape), {}, []
    for spec in specs:
        start = time.monotonic()
        model = fit_five_task(x[fit], y[fit], shape, spec, seed)
        pred = predict_five_task(model, x[val], shape, spec)
        models[spec["candidate"]] = model
        torch.save(model.state_dict(), folder / f"{spec['candidate']}.pt")
        np.save(folder / f"{spec['candidate']}-validation.npy", pred)
        rows.append(dict(spec, val_bacc=float(balanced_accuracy_score(y[val], pred)),
                         fit_validation_seconds=time.monotonic() - start))
        print(folder.name, spec["candidate"], "validation", round(rows[-1]["val_bacc"], 4), flush=True)
    selected = select(rows)
    common = select([r for r in rows if r["candidate"] != "angle4x2"])
    legacy = historical_rule(x[fit], shape)
    write_json(folder / "encoding-selection.json", dict(selected=selected, common_selected=common,
                                                        legacy_selected=legacy, validation=rows))
    # Test predictions only after every choice is saved.
    for row, spec in zip(rows, specs):
        pred = predict_five_task(models[spec["candidate"]], xt, shape, spec)
        np.save(folder / f"{spec['candidate']}-test.npy", pred)
        row.update(test_bacc=float(balanced_accuracy_score(yt, pred)), selected=spec["candidate"] == selected,
                   common_selected=spec["candidate"] == common, legacy_selected=spec["candidate"] == legacy)
    pd.DataFrame(rows).to_csv(folder / "quantum_scores.csv", index=False)


# ---------------------------------------------------------------- classical pooling screen

def representations(shape):
    if len(shape) == 1:
        return [("pool8", (8,)), ("full", None)]
    if len(shape) == 2:
        return [("grid2x4", (2, 4)), ("grid4x2", (4, 2)), ("grid4x4", (4, 4)), ("full", None)]
    return [("grid2x2x2", (2, 2, 2)), ("grid4x4x4", (4, 4, 4)), ("full", None)]


def cv_splits(x, y, groups, seed):
    if groups is None:
        return list(StratifiedKFold(3, shuffle=True, random_state=seed).split(x, y))
    return list(StratifiedGroupKFold(3, shuffle=True, random_state=seed).split(x, y, groups))


def rbf_search(x, y, groups, seed):
    """RBF-SVM, C = 1, 11 gamma values around 1 / dimension, three inner folds."""
    grid = {"svc__gamma": np.geomspace(1e-3 / x.shape[1], 1e3 / x.shape[1], 11)}
    return GridSearchCV(make_pipeline(StandardScaler(), SVC(C=1, kernel="rbf")), grid, scoring="balanced_accuracy",
                        cv=cv_splits(x, y, groups, seed), n_jobs=1).fit(x, y)


def classical(d, folder, seed):
    x, y, xt, yt, shape, groups = (d[k] for k in ("x", "y", "xt", "yt", "shape", "groups"))
    folds, rows, fitted = cv_splits(x, y, groups, seed), [], {}
    for name, grid in representations(shape):
        xf, pred, fold_scores = features(x, shape, grid), np.empty_like(y), []
        for k, (tr, va) in enumerate(folds):
            pred[va] = rbf_search(xf[tr], y[tr], None if groups is None else groups[tr], seed + 100 + k).predict(xf[va])
            fold_scores.append(float(balanced_accuracy_score(y[va], pred[va])))
        fitted[name] = rbf_search(xf, y, groups, seed + 200)
        rows.append(dict(representation=name, n_features=xf.shape[1], cv_bacc=float(balanced_accuracy_score(y, pred)),
                         cv_fold_scores=fold_scores, gamma=float(fitted[name].best_params_["svc__gamma"])))
        print(folder.name, "probe", name, round(rows[-1]["cv_bacc"], 4), flush=True)
    selected = min(rows, key=lambda r: (-r["cv_bacc"], r["n_features"], r["representation"]))["representation"]
    for row, (name, grid) in zip(rows, representations(shape)):
        row.update(test_bacc=float(balanced_accuracy_score(yt, fitted[name].predict(features(xt, shape, grid)))),
                   selected=name == selected)
    pd.DataFrame(rows).to_csv(folder / "representation_scores.csv", index=False)


def run(job):
    name, seed = job.rsplit("-", 1)
    seed, folder = int(seed), OUT / job
    if (folder / "complete.json").exists():
        return
    folder.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    d = five_task_data(name, seed)
    np.savez_compressed(folder / "splits.npz", fit=d["fit"], validation=d["val"], train_labels=d["y"],
                        test_labels=d["yt"])
    write_json(folder / "inputs.json", dict(dataset=name, seed=seed, shape=d["shape"], n_fit=len(d["fit"]),
                                            n_validation=len(d["val"]), n_test=len(d["yt"]),
                                            train_sha256=digest(d["x"]), test_sha256=digest(d["xt"])))
    if five_task_candidates(d["shape"]):
        quantum(d, folder, seed)
    classical(d, folder, seed)
    write_json(folder / "complete.json", dict(status="complete"))


if __name__ == "__main__":
    cli(JOBS, run, __doc__)
