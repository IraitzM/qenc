"""Accessibility ladder on the 95 saved models (main text Section 3, SI S4).

Six classifiers per model, all fitted on the fitting split with five-fold inner CV
(grouped by recording for EEG) and scored once on the test split:

  L0  best of logistic and RBF-SVM on the raw standardized input
  L1  the same on the coordinates the encoding receives
  L2  SVM on the fidelity kernel of the final state (initial and trained parameters), and its KTA
  L3  logistic on all one- and two-qubit Pauli expectations of that state
  L4  logistic on the model's own readout features (initial and trained)
  L5  L4 on the readout features of 20 random initializations (seeds 10000 + k)

Needs five_task.py and spatial_pilot.py first.

    uv run experiments/ladder.py --list | --job ECG-52 | --all
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from probes import JOBS, SIGNALS, saved_models

from qenc.measures import (
    alignment,
    fidelity,
    fit_classical,
    fit_kernel,
    fit_logistic,
    inner_cv,
    pauli_features,
    readout_features,
    states,
)
from qenc.pooling import features
from qenc.utils import bundle, cli, write_json

OUT = bundle("ladder")
N_RANDOM, RANDOM_SEED0 = 20, 10_000


def run(job):
    name, seed = job.rsplit("-", 1)
    seed, folder = int(seed), OUT / job
    if (folder / "complete.json").exists():
        return
    folder.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    d, source, specs, make_input, make_model = saved_models(name, seed)
    fit, y, yt = d["fit"], d["y"][d["fit"]], d["yt"]
    cv = inner_cv(y, d["groups"][fit] if d.get("groups") is not None else None, seed)
    scores = pd.read_csv(source / ("quantum_scores.csv" if name in SIGNALS else "scores.csv"))
    rows, predictions = [], {}

    def record(level, stage, candidate, family, result, n_features, **extra):
        test, cv_best, params, pred = result
        rows.append(dict(dataset=name, seed=seed, candidate=candidate, family=family, level=level, stage=stage,
                         test_bacc=test, cv_bacc=cv_best, params=params, n_features=n_features, **extra))
        predictions[f"{candidate}|{level}|{stage}"] = pred

    raw, raw_test = d["x"].reshape(len(d["x"]), -1), d["xt"].reshape(len(d["xt"]), -1)
    record("L0", "raw", "classical", "classical", fit_classical(raw[fit], raw_test, y, yt, cv, seed), raw.shape[1])
    for spec in specs:
        key, kind = spec["candidate"], spec["kind"]
        trained = make_model(spec, seed)
        trained.load_state_dict(torch.load(source / f"{key}.pt", weights_only=True))
        initial = make_model(spec, seed)
        xx, xt = make_input(d["x"], d["shape"], spec)[fit], make_input(d["xt"], d["shape"], spec)
        n, zz = trained.n_qubits, kind == "angle"
        coords, coords_test = (features(d[k], d["shape"], spec["grid"]) for k in ("x", "xt"))
        coords, coords_test = coords.reshape(len(coords), -1), coords_test.reshape(len(coords_test), -1)
        record("L1", "input", key, kind, fit_classical(coords[fit], coords_test, y, yt, cv, seed), coords.shape[1])
        end_to_end = scores.loc[scores.candidate == key, "test_bacc"].iloc[0]
        for stage, model in (("initial", initial), ("trained", trained)):
            s_fit, s_test = states(model, xx), states(model, xt)
            r_fit, r_test = readout_features(s_fit, n, zz), readout_features(s_test, n, zz)
            with torch.no_grad():
                own = torch.cat([model.features(part) for part in xt.split(32)]).numpy()
            np.testing.assert_allclose(r_test, own, atol=1e-6, err_msg=f"{job} {key} {stage}: state readout mismatch")
            if stage == "trained":
                with torch.no_grad():
                    logits = model.readout(torch.tensor(own, dtype=model.readout.weight.dtype))
                np.testing.assert_array_equal(logits.argmax(1).numpy(), np.load(source / f"{key}-test.npy"))
            k_fit, k_test = fidelity(s_fit, s_fit), fidelity(s_test, s_fit)
            record("L2", stage, key, kind, fit_kernel(k_fit, k_test, y, yt, cv), 2 ** n, kta=alignment(k_fit, y))
            p_fit, p_test = pauli_features(s_fit, n), pauli_features(s_test, n)
            record("L3", stage, key, kind, fit_logistic(p_fit, p_test, y, yt, cv, seed), p_fit.shape[1])
            record("L4", stage, key, kind, fit_logistic(r_fit, r_test, y, yt, cv, seed), r_fit.shape[1],
                   end_to_end_bacc=end_to_end)
        for k in range(N_RANDOM):
            model = make_model(spec, RANDOM_SEED0 + k)
            r_fit, r_test = (readout_features(states(model, a), n, zz) for a in (xx, xt))
            record("L5", f"random{k:02d}", key, kind, fit_logistic(r_fit, r_test, y, yt, cv, seed), r_fit.shape[1])
        print(job, key, "ladder done", flush=True)
    pd.DataFrame(rows).to_csv(folder / "scores.csv", index=False)
    np.savez_compressed(folder / "predictions.npz", yt=yt, **predictions)
    write_json(folder / "complete.json", dict(status="complete"))


if __name__ == "__main__":
    cli(JOBS, run, __doc__)
