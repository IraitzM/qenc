"""Readout factorization on a zero-overlap checkerboard (SI S10).

The '2c' 4x4 checkerboard of 3-D ellipsoids (600 training, 300 test points), scaled to
[-pi, pi]. Three-qubit angle re-uploading (four uploads, 12 epochs) with an entangling or
a product ansatz, read out through local Z alone or with the neighbouring ZZ products.

    uv run experiments/checkerboard.py --all
"""
from __future__ import annotations

import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler

from qenc.data import checkerboard
from qenc.models import AngleClassifier
from qenc.utils import bundle, cli

OUT = bundle("checkerboard")
SEED = 42
ARMS = [("entangled ansatz + local readout", True, False), ("entangled ansatz + ZZ readout", True, True),
        ("product ansatz + local readout", False, False), ("product ansatz + ZZ readout", False, True)]


def fit(x, y, xt, yt, entangling, zz_readout):
    torch.manual_seed(SEED)
    model = AngleClassifier(3, n_reuploads=4, entangling=entangling, zz_readout=zz_readout)
    opt, loss_fn = torch.optim.Adam(model.parameters(), lr=0.05), torch.nn.CrossEntropyLoss()
    xx, yy = torch.tensor(x, dtype=torch.float64), torch.tensor(y, dtype=torch.long)
    for _ in range(12):
        perm = torch.randperm(len(xx))
        for start in range(0, len(xx), 16):
            ix = perm[start:start + 16]
            opt.zero_grad()
            loss_fn(model(xx[ix]), yy[ix]).backward()
            opt.step()
    with torch.no_grad():
        return balanced_accuracy_score(yt, model(torch.tensor(xt, dtype=torch.float64)).argmax(1).numpy())


def run(job):
    if (OUT / "scores.csv").exists():
        print(OUT / "scores.csv", "exists, skipping")
        return
    x, y, xt, yt = checkerboard(overlap=0.0, seed=SEED)
    scaler = MinMaxScaler(feature_range=(-torch.pi, torch.pi)).fit(x)
    x, xt = scaler.transform(x), scaler.transform(xt)
    rows = []
    for name, entangling, zz_readout in ARMS:
        rows.append(dict(model=name, test_bacc=fit(x, y, xt, yt, entangling, zz_readout)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(OUT / "scores.csv", index=False)


if __name__ == "__main__":
    cli(["2c"], run, __doc__)
