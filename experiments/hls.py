"""Heart against lung sounds (HLS-CMDS) on two input representations (main text Section 3.1, SI S8).

Five seeds (42-46), recording-level split, 40-value windows as raw waveforms or as
log-power spectral envelopes. Models: QSM (eight uploads, t = 0.1, z-scored input),
amplitude on min-max scaled input, Angle-8 on eight block means (local Z readout),
a tuned RBF-SVM and an MLP (64 hidden units) on standardized input.

    uv run experiments/hls.py --list | --job spectral-42 | --all
"""
from __future__ import annotations

import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from qenc.data import hls_cmds, znorm_signal
from qenc.measures import rbf_svm_scores
from qenc.models import AmplitudeClassifier, AngleClassifier, QSMClassifier, hankel_upload, to_nonneg_domain
from qenc.utils import bundle, cli, write_json

OUT = bundle("hls")
JOBS = [f"{rep}-{seed}" for rep in ("waveform", "spectral") for seed in (42, 43, 44, 45, 46)]
BATCH = 16


def train(model, x, y, xt, yt, epochs, lr, order_seed=None):
    """Adam on cross-entropy; ``order_seed`` reseeds the minibatch order (the QSM arm keeps the stream)."""
    if order_seed is not None:
        torch.manual_seed(order_seed)
    opt, loss_fn, yy = (torch.optim.Adam(model.parameters(), lr=lr), torch.nn.CrossEntropyLoss(),
                        torch.tensor(y, dtype=torch.long))
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        for start in range(0, len(x), BATCH):
            ix = perm[start:start + BATCH]
            opt.zero_grad()
            loss_fn(model(x[ix]), yy[ix]).backward()
            opt.step()
    with torch.no_grad():
        return balanced_accuracy_score(yt, model(xt).argmax(1).numpy())


def run(job):
    rep, seed = job.split("-")
    seed, folder = int(seed), OUT / job
    if (folder / "complete.json").exists():
        return
    folder.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    x, y, xt, yt = hls_cmds(seed, representation=rep)
    scores = {}
    torch.manual_seed(seed)
    model = QSMClassifier(QSMClassifier.n_qubits_for(x.shape[1]), n_uploads=8)
    u, ut = (torch.tensor(hankel_upload(znorm_signal(a), t=0.1), dtype=torch.complex128) for a in (x, xt))
    scores["QSM"] = train(model, u, y, ut, yt, epochs=20, lr=0.005)
    torch.manual_seed(seed)
    model = AmplitudeClassifier(x.shape[1], n_layers=3)
    a, at = (torch.tensor(to_nonneg_domain(v), dtype=torch.float64) for v in (x, xt))
    scores["amplitude"] = train(model, a, y, at, yt, epochs=15, lr=0.05, order_seed=0)
    torch.manual_seed(seed)
    model = AngleClassifier(8, n_reuploads=4, device_name="default.qubit").double()
    a, at = (torch.tensor(v, dtype=torch.float64) for v in (x, xt))
    scores["angle (pooled)"] = train(model, a, y, at, yt, epochs=15, lr=0.05, order_seed=0)
    scaler = StandardScaler().fit(x)
    xs, xts = scaler.transform(x), scaler.transform(xt)
    scores["RBF"] = rbf_svm_scores(xs, y, xts, yt, seed=seed)["test_bacc"]
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=1000, random_state=seed).fit(xs, y)
    scores["MLP"] = balanced_accuracy_score(yt, mlp.predict(xts))
    print(job, {k: round(v, 3) for k, v in scores.items()}, flush=True)
    pd.DataFrame([dict(representation=rep, seed=seed, model=k, test_bacc=float(v)) for k, v in scores.items()]).to_csv(
        folder / "scores.csv", index=False)
    write_json(folder / "complete.json", dict(status="complete"))


if __name__ == "__main__":
    cli(JOBS, run, __doc__)
