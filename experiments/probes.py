"""Linear probes of initial and trained features (main text Section 3.3, SI S11).

For each of the 95 saved models (signals from the five-task study, seeds 52-56; images
and volumes from the spatial pilot, seeds 62-66) the initial model is rebuilt from its
seed and the trained one loaded from its checkpoint. Before any probe, the regenerated
data draw, splits and labels must be the saved ones, a saved initial checkpoint (spatial
QSM, kept in results/) must equal the rebuilt initialization, batched QSM must give the
features of the sequential circuit the signal models were trained with, and the trained
readout must reproduce the saved validation and test predictions and the recorded test
BA. The same standardized logistic regression (C = 1) is then fitted on the fitting split
to the initial features, the trained features and the encoding's input coordinates, and
scored once on the test split, whose predictions are saved.
Delta_learn = BA(trained) - BA(initial). Needs five_task.py and spatial_pilot.py first.

    uv run experiments/probes.py --list | --job ECG-52 | --all
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qenc.data import spatial_draw
from qenc.pooling import features
from qenc.protocols import (
    FIVE_TASK_SEEDS,
    SPATIAL_SEEDS,
    SPATIAL_TASKS,
    five_task_candidates,
    five_task_data,
    five_task_input,
    five_task_model,
    spatial_input,
    spatial_model,
    spatial_specs,
)
from qenc.utils import RESULTS, bundle, cli, digest, write_json

OUT = bundle("probes")
SIGNALS = ("ECG", "EEG", "MNIST1D")
JOBS = ([f"{d}-{s}" for d in SIGNALS for s in FIVE_TASK_SEEDS]
        + [f"{d}-{s}" for d in SPATIAL_TASKS for s in SPATIAL_SEEDS])


def saved_models(name, seed):
    """Data, source folder, candidate specs, input map and model builder of one job."""
    if name in SIGNALS:
        d = five_task_data(name, seed)
        n_classes = len(np.unique(d["y"]))

        def make(spec, init_seed, batched=True):
            torch.manual_seed(init_seed)  # QSM features are evaluated batched; the circuit is the trained one
            return five_task_model(spec, d["shape"], n_classes, batched_qsm=batched)

        return d, RESULTS / "five-task" / f"{name}-{seed}", five_task_candidates(d["shape"]), five_task_input, make
    d = spatial_draw(name, seed)
    return d, RESULTS / "spatial-pilot" / f"{name}-{seed}", spatial_specs(name), spatial_input, spatial_model


def check_draw(d, source, signal):
    """The regenerated inputs, splits and labels must be the saved ones."""
    saved = json.loads((source / "inputs.json").read_text())
    assert (digest(d["x"]), digest(d["xt"])) == (saved["train_sha256"], saved["test_sha256"]), f"{source}: draw"
    names = ({"fit": "fit", "val": "validation", "y": "train_labels", "yt": "test_labels"} if signal
             else {"fit": "fit", "val": "val", "y": "y", "yt": "yt"})
    with np.load(source / ("splits.npz" if signal else "split.npz")) as split:
        for key, stored in names.items():
            np.testing.assert_array_equal(d[key], split[stored])


def readout_features(model, x, chunk=16):
    with torch.no_grad():
        return torch.cat([model.features(part) for part in x.split(chunk)]).numpy()


def run(job):
    name, seed = job.rsplit("-", 1)
    seed, folder = int(seed), OUT / job
    if (folder / "complete.json").exists():
        return
    folder.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    d, source, specs, make_input, make_model = saved_models(name, seed)
    check_draw(d, source, name in SIGNALS)
    scores = pd.read_csv(source / ("quantum_scores.csv" if name in SIGNALS else "scores.csv"))
    rows, matched = [], 0
    for spec in specs:
        key = spec["candidate"]
        initial = make_model(spec, seed)
        if (source / f"{key}-initial.pt").exists():  # the original spatial pilot saved its QSM initializations
            saved = torch.load(source / f"{key}-initial.pt", weights_only=True)
            for param, value in initial.state_dict().items():
                torch.testing.assert_close(value, saved[param], rtol=0, atol=0)
            matched += 1
        trained = make_model(spec, seed)
        trained.load_state_dict(torch.load(source / f"{key}.pt", weights_only=True))
        x, xt = make_input(d["x"], d["shape"], spec), make_input(d["xt"], d["shape"], spec)
        if name in SIGNALS and spec["kind"] == "qsm":  # trained one example at a time, evaluated batched here
            sequential = make_model(spec, seed, batched=False)
            sequential.load_state_dict(trained.state_dict())
            np.testing.assert_allclose(readout_features(trained, xt[:3]), readout_features(sequential, xt[:3]),
                                       rtol=1e-10, atol=1e-10)
        trained_fit, trained_test = readout_features(trained, x), readout_features(trained, xt)
        predictions = {}
        for part, values in (("validation", trained_fit[d["val"]]), ("test", trained_test)):
            with torch.no_grad():
                logits = trained.readout(torch.tensor(values, dtype=trained.readout.weight.dtype))
            predictions[part] = logits.argmax(1).numpy()
            np.testing.assert_array_equal(predictions[part], np.load(source / f"{key}-{part}.npy"))
        end_to_end = float(scores.loc[scores.candidate == key, "test_bacc"].iloc[0])
        assert abs(balanced_accuracy_score(d["yt"], predictions["test"]) - end_to_end) < 1e-12, f"{key}: recorded BA"
        stages = {"initial": (readout_features(initial, x), readout_features(initial, xt)),
                  "trained": (trained_fit, trained_test)}
        stages = {k: (np.round(a, 10), np.round(b, 10)) for k, (a, b) in stages.items()}
        stages["input"] = (features(d["x"], d["shape"], spec["grid"]), features(d["xt"], d["shape"], spec["grid"]))
        for stage, (a, b) in stages.items():
            probe = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=5000, random_state=seed))
            probe.fit(a[d["fit"]], d["y"][d["fit"]])
            pred = probe.predict(b)
            np.save(folder / f"{key}-{stage}-probe-test.npy", pred)
            rows.append(dict(dataset=name, seed=seed, candidate=key, family=spec["kind"], stage=stage,
                             n_features=a.shape[1], test_bacc=balanced_accuracy_score(d["yt"], pred),
                             validation_bacc=balanced_accuracy_score(d["y"][d["val"]], probe.predict(a[d["val"]])),
                             end_to_end_bacc=end_to_end, parameters=sum(p.numel() for p in trained.parameters()),
                             n_qubits=spec["n_qubits"]))
        print(job, key, "checks passed, probes done", flush=True)
    pd.DataFrame(rows).to_csv(folder / "scores.csv", index=False)
    write_json(folder / "complete.json", dict(status="complete", saved_initializations_matched=matched))


if __name__ == "__main__":
    cli(JOBS, run, __doc__)
