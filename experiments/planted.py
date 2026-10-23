"""Labels planted on real images and volumes (main text Section 3.2, SI S9).

Real PneumoniaMNIST and NoduleMNIST3D inputs (seeds 80-84; 800 fit / 100 validation /
300 test, class labels unused), z-scored and averaged onto the coarse (8-value) or fine
(16-value) grid. With x = z / ||z|| and a, b, c, d the first four principal axes of the
fitting x, and t, t1, t2 fitting medians:

  F2  1[(a.x)^2 - (b.x)^2 > t]                                  in the one-copy amplitude class
  F4  1[((a.x)^2 - (b.x)^2 - t1) ((c.x)^2 - (d.x)^2 - t2) > 0]  needs two copies
  Kk  1[cos(k u) > 0], u uniform, planted as coarse coordinate 0  Angle-8 with four uploads
      is bounded by 1/2 + 4/(2k) for k > 4

Arms: one and two amplitude copies (F) or Angle-8 (K), trained for 20 epochs, the SVM on
the fidelity kernel of each trained final state (its ceiling), and classical references
(F: best of logistic/RBF on x and logistic on degree-2 and degree-4 monomials; K: best of
logistic/RBF on the eight coordinates and logistic on Fourier features of u of degree 4 and 16).
Every amplitude model is checked against its states: the explicit two-copy states give the
circuit's own readout, and the ceiling kernel of k copies is (x.x')^(2k).

    uv run experiments/planted.py --list | --job PneumoniaMNIST-80-F4-g8 | --all
"""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from sklearn.metrics import balanced_accuracy_score

from qenc.data import medmnist, znorm_flat
from qenc.measures import fidelity, fit_classical, fit_kernel, fit_logistic, inner_cv, states
from qenc.models import AmplitudeClassifier, AngleClassifier
from qenc.pooling import features
from qenc.protocols import spatial_grids
from qenc.utils import bundle, cli, write_json

OUT = bundle("planted")
SEEDS = (80, 81, 82, 83, 84)
N_SPLIT = {"fit": 800, "val": 100, "test": 300}
EPOCHS, BATCH, LR, UPLOADS = 20, 32, 0.05, 4
FREQS = (2, 4, 8, 16)
CELLS = (("PneumoniaMNIST", 8), ("PneumoniaMNIST", 16), ("NoduleMNIST3D", 8))
JOBS = ([f"{d}-{s}-K{k}" for d in ("NoduleMNIST3D", "PneumoniaMNIST") for s in SEEDS for k in FREQS]
        + [f"{d}-{s}-{label}-g{g}" for d, g in CELLS for s in SEEDS for label in ("F2", "F4")])


def base(dataset, seed):
    """Coarse and fine block means of z-scored inputs per split; the draw ignores class labels."""
    coarse, fine = spatial_grids(dataset)
    rng, out = np.random.default_rng(seed), {}
    with np.load(medmnist(dataset.lower())) as raw:
        train = rng.permutation(len(raw["train_images"]))[: N_SPLIT["fit"] + N_SPLIT["val"]]
        test = rng.permutation(len(raw["test_images"]))[: N_SPLIT["test"]]
        for split, source, ix in (("fit", "train", train[: N_SPLIT["fit"]]), ("val", "train", train[N_SPLIT["fit"]:]),
                                  ("test", "test", test)):
            images = raw[f"{source}_images"][np.sort(ix)]
            x = znorm_flat(images.astype(float) / 255.)
            out[split] = {8: features(x, images.shape[1:], coarse),
                          int(np.prod(fine)): features(x, images.shape[1:], fine)}
    return out


def unit(z):
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def function_labels(label, xs):
    """Labels per split from the fitting split's principal axes and medians."""
    centered = xs["fit"] - xs["fit"].mean(0)
    a, b, c, d = np.linalg.svd(centered, full_matrices=False)[2][:4]
    s1 = {s: (x @ a) ** 2 - (x @ b) ** 2 for s, x in xs.items()}
    t1 = np.median(s1["fit"])
    if label == "F2":
        return {s: (v > t1).astype(int) for s, v in s1.items()}
    s2 = {s: (x @ c) ** 2 - (x @ d) ** 2 for s, x in xs.items()}
    t2 = np.median(s2["fit"])
    return {s: ((s1[s] - t1) * (s2[s] - t2) > 0).astype(int) for s in xs}


def monomials(x, degree):
    index = list(combinations_with_replacement(range(x.shape[1]), degree))
    return np.stack([np.prod(x[:, list(i)], axis=1) for i in index], axis=1)


def trig(u, degree):
    return np.column_stack([f(j * u) for j in range(1, degree + 1) for f in (np.cos, np.sin)])


def build(arm, d, seed):
    torch.manual_seed(seed)
    if arm == "angle":
        return AngleClassifier(8, n_reuploads=UPLOADS, zz_readout=True)
    return AmplitudeClassifier(d, copies=2 if arm == "amp2" else 1, n_layers=3)


def train(model, x, y, seed):
    torch.manual_seed(seed)
    opt, yy = torch.optim.Adam(model.parameters(), lr=LR), torch.tensor(y, dtype=torch.long)
    for _ in range(EPOCHS):
        order = torch.randperm(len(y))
        for start in range(0, len(y), BATCH):
            ix = order[start:start + BATCH]
            opt.zero_grad()
            torch.nn.functional.cross_entropy(model(x[ix]), yy[ix]).backward()
            opt.step()
    return model


def predict(model, x):
    with torch.no_grad():
        return torch.cat([model(part) for part in x.split(64)]).argmax(1).numpy()


def two_copy_states(model, z):
    """V (|x> (x) |x>) of the two-copy circuit, formed explicitly (PennyLane cannot broadcast two embeddings)."""
    x = unit(z)
    x = np.pad(x, ((0, 0), (0, 2 ** model.n_per - x.shape[1])))
    product = np.einsum("bi,bj->bij", x, x).reshape(len(x), -1)
    v = qml.matrix(qml.StronglyEntanglingLayers(model.weights.detach().double().numpy(), wires=range(model.n_qubits)),
                   wire_order=range(model.n_qubits))
    return product @ v.T


def local_z(psi, n):
    """<Z_i> of statevectors whose first wire is the most significant bit."""
    probs = np.abs(psi.reshape(len(psi), *([2] * n))) ** 2
    return np.stack([probs.take(0, axis=i + 1).sum(axis=tuple(range(1, n)))
                     - probs.take(1, axis=i + 1).sum(axis=tuple(range(1, n))) for i in range(n)], axis=1)


def quantum_arm(arm, d, seed, x, y, yv, yt, cv):
    model = train(build(arm, d, seed), x["fit"], y, seed)
    pred_val, pred_test = predict(model, x["val"]), predict(model, x["test"])
    if arm == "amp2":
        s_fit, s_test = two_copy_states(model, x["fit"].numpy()), two_copy_states(model, x["test"].numpy())
        with torch.no_grad():
            own = model.features(x["fit"][:16]).numpy()
        np.testing.assert_allclose(local_z(s_fit[:16], model.n_qubits), own, atol=1e-6)
    else:
        s_fit, s_test = states(model, x["fit"]), states(model, x["test"])
    if arm != "angle":  # k amplitude copies have the ceiling kernel (x.x')^(2k), whatever the trained unitary
        u = unit(x["fit"][:32].numpy())
        exact = (u @ u.T) ** (4 if arm == "amp2" else 2)
        np.testing.assert_allclose(fidelity(s_fit[:32], s_fit[:32]), exact, atol=1e-8)
    ceiling = fit_kernel(fidelity(s_fit, s_fit), fidelity(s_test, s_fit), y, yt, cv)
    rows = [dict(arm=arm, test_bacc=balanced_accuracy_score(yt, pred_test),
                 val_bacc=balanced_accuracy_score(yv, pred_val), cv_bacc=np.nan, params=""),
            dict(arm=f"{arm}_ceiling", test_bacc=ceiling[0], val_bacc=np.nan, cv_bacc=ceiling[1], params=ceiling[2])]
    return rows, {arm: pred_test, f"{arm}_ceiling": ceiling[3]}


def classical_arm(name, result):
    test, cv_best, params, pred = result
    return [dict(arm=name, test_bacc=test, val_bacc=np.nan, cv_bacc=cv_best, params=params)], {name: pred}


def run(job):
    folder = OUT / job
    if (folder / "complete.json").exists():
        return
    folder.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    dataset, seed, label, *rest = job.split("-")
    seed, data = int(seed), base(dataset, int(seed))
    rows, preds = [], {}
    if label in ("F2", "F4"):
        d = int(rest[0][1:])
        z = {s: data[s][d] for s in N_SPLIT}
        x = {s: unit(v) for s, v in z.items()}
        ys = function_labels(label, x)
        y, yv, yt = ys["fit"], ys["val"], ys["test"]
        cv = inner_cv(y, None, seed)
        tensors = {s: torch.tensor(v, dtype=torch.float64) for s, v in z.items()}
        arms = [quantum_arm(arm, d, seed, tensors, y, yv, yt, cv) for arm in ("amp1", "amp2")]
        arms += [classical_arm("classical", fit_classical(x["fit"], x["test"], y, yt, cv, seed)),
                 classical_arm("surrogate2",
                               fit_logistic(monomials(x["fit"], 2), monomials(x["test"], 2), y, yt, cv, seed)),
                 classical_arm("surrogate4",
                               fit_logistic(monomials(x["fit"], 4), monomials(x["test"], 4), y, yt, cv, seed))]
    else:
        k = int(label[1:])
        rng = np.random.default_rng(seed + 3000)
        u = {s: rng.uniform(0, 2 * np.pi, N_SPLIT[s]) for s in ("fit", "val", "test")}
        z = {s: data[s][8].copy() for s in N_SPLIT}
        for s in z:
            z[s][:, 0] = u[s]
        ys = {s: (np.cos(k * u[s]) > 0).astype(int) for s in N_SPLIT}
        y, yv, yt = ys["fit"], ys["val"], ys["test"]
        cv = inner_cv(y, None, seed)
        tensors = {s: torch.tensor(v, dtype=torch.float64) for s, v in z.items()}
        arms = [quantum_arm("angle", 8, seed, tensors, y, yv, yt, cv),
                classical_arm("classical", fit_classical(z["fit"], z["test"], y, yt, cv, seed))]
        arms += [classical_arm(f"fourier{D}", fit_logistic(trig(u["fit"], D), trig(u["test"], D), y, yt, cv, seed))
                 for D in (UPLOADS, 16)]
    for r, p in arms:
        rows += r
        preds.update(p)
    print(job, {r["arm"]: round(r["test_bacc"], 3) for r in rows}, flush=True)
    pd.DataFrame(rows).assign(dataset=dataset, seed=seed, label=label, job=job,
                              class1_share=float(yt.mean())).to_csv(folder / "scores.csv", index=False)
    np.savez_compressed(folder / "predictions.npz", yt=yt, **preds)
    write_json(folder / "complete.json", dict(status="complete"))


if __name__ == "__main__":
    cli(JOBS, run, __doc__)
