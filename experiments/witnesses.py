"""Witnesses that input-only diagnostics cannot screen pooling (main text Section 3.1, SI S2, S5).

1. Phase: cos(2t) and sin(2t) images on a 28x28 grid have equal Fourier power, yet 14x7
   block averaging discards different shares of their energy.
2. Label rule: x = u c + 2 v h with c in the row space and h in the null space of the
   averaging, in 2-D (28x28 to 2x4) and 3-D (8^3 to 2^3), seeds 52-56. The loss is ~0.8 for
   every x, yet the compressed input keeps the label u and loses the label v (standardized
   logistic regression, C = 1).
3. One amplitude copy: on x(u) = (cos u, sin u) every quadratic score lies in span{1, cos 2u,
   sin 2u}, so cos 4u keeps a mean squared error of 1/2; two copies give it exactly.

    uv run experiments/witnesses.py --all
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qenc.pooling import box_features, box_projection, compression_loss
from qenc.utils import bundle, cli, write_json

OUT = bundle("witnesses")


def phase():
    t = 2 * np.pi * np.arange(28) / 28
    images = np.stack([np.broadcast_to(f(2 * t), (28, 28)) for f in (np.cos, np.sin)])
    result = compression_loss(images, (28, 28), (2, 4))
    return pd.DataFrame(dict(phase=["cosine", "sine"], spectral_tail=result["spectral_tail"],
                             projection_loss=result["projection_loss"]))


def label_rule():
    rows = []
    for shape, grid in [((28, 28), (2, 4)), ((8, 8, 8), (2, 2, 2))]:
        rng = np.random.default_rng(910)
        size = int(np.prod(shape))
        c = box_projection(rng.normal(size=(1, size)), shape, grid).ravel()
        c -= c.mean()
        c /= np.linalg.norm(c)
        h = rng.normal(size=(1, size))
        h = (h - box_projection(h, shape, grid)).ravel()
        h /= np.linalg.norm(h)
        for seed in range(52, 57):
            rng = np.random.default_rng(seed)

            def draw():
                # Every coarse amplitude occurs with both fine signs and vice versa.
                amplitudes = rng.uniform(.8, 1.2, size=(80, 2))
                pairs = np.array([(a * u, b * v) for u, v in amplitudes
                                  for a, b in [(-1, -1), (-1, 1), (1, -1), (1, 1)]])
                rng.shuffle(pairs)
                x = pairs[:, 0, None] * c + 2 * pairs[:, 1, None] * h
                return x, (pairs[:, 0] > 0).astype(int), (pairs[:, 1] > 0).astype(int)

            (x, yc, yf), (xt, ytc, ytf) = draw(), draw()
            loss = float(compression_loss(x, shape, grid)["projection_loss"].mean())
            for label, y, yt in [("coarse label", yc, ytc), ("fine label", yf, ytf)]:
                for representation in ["full", "compressed"]:
                    a, b = ((x, xt) if representation == "full"
                            else (box_features(torch.tensor(v), shape, grid).numpy() for v in (x, xt)))
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000)).fit(a, y)
                    rows.append(dict(dimensions=len(shape), seed=seed, label_rule=label, representation=representation,
                                     projection_loss=loss,
                                     test_bacc=float(balanced_accuracy_score(yt, model.predict(b)))))
    return pd.DataFrame(rows)


def single_copy(n_points=256):
    u = 2 * np.pi * np.arange(n_points) / n_points
    target = np.cos(4 * u)
    basis = np.column_stack([np.ones_like(u), np.cos(2 * u), np.sin(2 * u)])
    single = basis @ np.linalg.lstsq(basis, target, rcond=None)[0]
    states = np.column_stack([np.cos(u), np.sin(u)])
    observable = (np.kron(np.diag([1.0, -1.0]), np.diag([1.0, -1.0]))
                  - np.kron([[0.0, 1.0], [1.0, 0.0]], [[0.0, 1.0], [1.0, 0.0]]))
    copies = np.einsum("ni,nj->nij", states, states).reshape(n_points, 4)
    double = np.einsum("ni,ij,nj->n", copies, observable, copies)
    return {"one_copy_mse": float(np.mean((single - target) ** 2)),
            "two_copy_mse": float(np.mean((double - target) ** 2))}


def run(job):
    if (OUT / "label_witness.csv").exists():
        print(OUT / "label_witness.csv", "exists, skipping")
        return
    phase().to_csv(OUT / "phase_witness.csv", index=False)
    label_rule().to_csv(OUT / "label_witness.csv", index=False)
    write_json(OUT / "single_copy.json", single_copy())
    print(pd.read_csv(OUT / "phase_witness.csv").to_string(index=False))
    print(pd.read_csv(OUT / "label_witness.csv").groupby(["dimensions", "label_rule", "representation"])
          .test_bacc.mean())
    print(single_copy())


if __name__ == "__main__":
    cli(["witnesses"], run, __doc__)
