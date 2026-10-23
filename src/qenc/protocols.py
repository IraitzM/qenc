"""Data draws, candidates, model inputs and training of the two studies whose 95 saved
models the probes and the ladder analyse: the five-task study and the spatial pilot.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from qenc import data
from qenc.models import (
    AmplitudeClassifier,
    AngleClassifier,
    QSMClassifier,
    block_upload,
    hankel_upload,
    required_qubits,
    trajectory_matrix,
)
from qenc.pooling import features

# ---------------------------------------------------------------- five-task study

FIVE_TASKS = ("ECG", "MNIST1D", "EEG", "PneumoniaMNIST", "NoduleMNIST3D")
FIVE_TASK_SEEDS = (52, 53, 54, 55, 56)


def five_task_data(name: str, seed: int) -> dict:
    """Training and test arrays, fit/validation indices and native shape of one task and seed.

    Signals and images are z-scored per sample; volumes stay in [0, 1]. Validation
    takes a stratified quarter of the training draw, except for EEG, whose fit,
    validation and test windows come from disjoint recordings.
    """
    if name == "EEG":
        x, y, groups, (fit, val, test) = data.eeg_grouped_split(seed)
        train = np.r_[fit, val]
        return dict(x=data.znorm_signal(x[train]), y=y[train], xt=data.znorm_signal(x[test]), yt=y[test],
                    fit=np.arange(len(fit)), val=np.arange(len(fit), len(train)), groups=groups[train],
                    test_groups=groups[test], shape=(178,))
    if name == "NoduleMNIST3D":
        x, y, xt, yt = data.nodule_draw(seed)
        shape = (28, 28, 28)
    else:
        if name == "MNIST1D":
            m = data.mnist1d()
            keep_train, keep_test = np.isin(m["y_train"], [0, 1, 2, 3]), np.isin(m["y_test"], [0, 1, 2, 3])
            xt, yt = data.subsample(m["x_test"][keep_test], m["y_test"][keep_test], 40, np.random.default_rng(42))
            x_pool, y_pool, n_per = m["x_train"][keep_train], m["y_train"][keep_train], 150
        elif name == "ECG":
            e = data.ecg_beats()
            x_pool, y_pool, xt, yt, n_per = e["x_train"], e["y_train"], e["x_test"], e["y_test"], 300
        else:
            x_pool, y_pool, xt, yt = data.pneumonia_pool()
            x_pool, xt, n_per = x_pool.reshape(len(x_pool), -1), xt.reshape(len(xt), -1), 300
        x, y = data.subsample(x_pool, y_pool, n_per, np.random.default_rng(seed))
        x, xt = data.znorm_signal(x), data.znorm_signal(xt)
        shape = (28, 28) if name == "PneumoniaMNIST" else (x.shape[1],)
    fit, val = train_test_split(np.arange(len(y)), test_size=.25, stratify=y, random_state=seed)
    return dict(x=x, y=y, xt=xt, yt=yt, fit=fit, val=val, groups=None, test_groups=None, shape=shape)


def five_task_candidates(shape) -> list[dict]:
    """Angle on eight block means (both 2x4 and 4x2 for images), amplitude on the full input, QSM on signals."""
    if len(shape) == 3:
        return []  # volumes only run the classical pooling screen
    angle = dict(candidate="angle", kind="angle", grid=(8,) if len(shape) == 1 else (2, 4),
                 n_qubits=8, depth=4, epochs=10, lr=.05, batch=32)
    specs = [angle] + ([dict(angle, candidate="angle4x2", grid=(4, 2))] if len(shape) == 2 else [])
    d = int(np.prod(shape))
    specs.append(dict(candidate="amplitude", kind="amplitude", grid=None, n_qubits=max(1, int(np.ceil(np.log2(d)))),
                      depth=3, epochs=10, lr=.05, batch=32))
    if len(shape) == 1 and d <= 256:
        specs.append(dict(candidate="qsm", kind="qsm", grid=None, n_qubits=QSMClassifier.n_qubits_for(d),
                          depth=8, epochs=20, lr=.005, batch=16))
    return specs


def five_task_input(x, shape, spec):
    if spec["kind"] == "angle":
        return torch.tensor(features(x, shape, spec["grid"]), dtype=torch.float64)
    if spec["kind"] == "qsm":  # the uploads re-standardize the already standardized signals
        return torch.tensor(hankel_upload(data.znorm_signal(x), t=.1), dtype=torch.complex128)
    return torch.tensor(x, dtype=torch.float64)


def five_task_model(spec, shape, n_classes, batched_qsm=False):
    if spec["kind"] == "angle":
        return AngleClassifier(8, n_reuploads=4, n_classes=n_classes, zz_readout=True)
    if spec["kind"] == "amplitude":
        return AmplitudeClassifier(int(np.prod(shape)), n_layers=3, n_classes=n_classes)
    return QSMClassifier(spec["n_qubits"], n_uploads=8, n_classes=n_classes, batched=batched_qsm)


def fit_five_task(x, y, shape, spec, seed):
    """Adam on cross-entropy for ``spec['epochs']`` epochs; the seed fixes initialization and minibatch order."""
    torch.manual_seed(seed)
    model = five_task_model(spec, shape, len(np.unique(y)))
    xx, yy = five_task_input(x, shape, spec), torch.tensor(y, dtype=torch.long)
    opt, loss_fn = torch.optim.Adam(model.parameters(), lr=spec["lr"]), torch.nn.CrossEntropyLoss()
    for _ in range(spec["epochs"]):
        permutation = torch.randperm(len(y))
        for start in range(0, len(y), spec["batch"]):
            ix = permutation[start:start + spec["batch"]]
            opt.zero_grad()
            loss_fn(model(xx[ix]), yy[ix]).backward()
            opt.step()
    return model


def predict_five_task(model, x, shape, spec):
    with torch.no_grad():
        return model(five_task_input(x, shape, spec)).argmax(1).numpy()


# ---------------------------------------------------------------- spatial pilot

SPATIAL_TASKS = ("PneumoniaMNIST", "NoduleMNIST3D")
SPATIAL_SEEDS = (62, 63, 64, 65, 66)


def spatial_grids(name):
    """Coarse and fine grids of block means: 2x4 and 4x4 images, 2^3 and 4^3 volumes."""
    return [(2, 4), (4, 4)] if name == "PneumoniaMNIST" else [(2, 2, 2), (4, 4, 4)]


def spatial_specs(name) -> list[dict]:
    """QSM on the native trajectory matrix and amplitude on each grid, and Angle-8 on the coarse grid.

    QSM registers are the larger of the native and the raster (1-D) lift, as in the
    pilot, where a raster control shared the register (five qubits for 4x4, seven for 4^3).
    """
    specs = []
    for grid in spatial_grids(name):
        d = int(np.prod(grid))
        specs.append(dict(candidate=f"native{d}", kind="qsm", grid=grid,
                          n_qubits=max(required_qubits(grid), required_qubits((d,))), depth=4, lr=.005, batch=32))
        specs.append(dict(candidate=f"amplitude{d}", kind="amplitude", grid=grid, n_qubits=(d - 1).bit_length(),
                          depth=3, lr=.05, batch=32))
    specs.append(dict(candidate="angle8", kind="angle", grid=spatial_grids(name)[0], n_qubits=8, depth=4, lr=.05,
                      batch=32))
    return specs


def spatial_input(x, shape, spec):
    z = features(x, shape, spec["grid"])
    if spec["kind"] != "qsm":
        return torch.tensor(z, dtype=torch.float64)
    return torch.tensor(block_upload(trajectory_matrix(z, spec["grid"]), time=2., n_qubits=spec["n_qubits"]),
                        dtype=torch.complex128)


def spatial_model(spec, seed):
    torch.manual_seed(seed)
    if spec["kind"] == "qsm":
        return QSMClassifier(spec["n_qubits"], n_uploads=4, batched=True)
    if spec["kind"] == "amplitude":
        return AmplitudeClassifier(int(np.prod(spec["grid"])), n_layers=3)
    return AngleClassifier(8, n_reuploads=4, zz_readout=True)
