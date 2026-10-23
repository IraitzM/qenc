"""EEG preprocessing and split controls (main text Section 3.1, SI S8).

Five partitions (seeds 42-46), 200 training and 100 test windows per class:

  recording split, z-scored windows: QSM (eight uploads, t = 0.1) and a tuned RBF-SVM
  recording split, raw windows:      tuned RBF-SVM on the window, and on its log standard deviation alone
  random split, z-scored windows:    QSM again, to measure the effect of sibling windows of one
                                     recording falling on both sides (the shared share is recorded)

    uv run experiments/eeg_controls.py --list | --job 42 | --all
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score

from qenc.data import eeg_split
from qenc.measures import rbf_svm_scores
from qenc.models import QSMClassifier, hankel_upload
from qenc.utils import bundle, cli, write_json

OUT = bundle("eeg-controls")
JOBS = [str(seed) for seed in (42, 43, 44, 45, 46)]


def qsm(x_train, y_train, x_test, y_test, seed):
    torch.manual_seed(seed)
    u_train = torch.tensor(hankel_upload(x_train, t=0.1), dtype=torch.complex128)
    u_test = torch.tensor(hankel_upload(x_test, t=0.1), dtype=torch.complex128)
    model = QSMClassifier(QSMClassifier.n_qubits_for(x_train.shape[1]), n_uploads=8)
    opt, loss_fn = torch.optim.Adam(model.parameters(), lr=0.005), torch.nn.CrossEntropyLoss()
    yy = torch.tensor(y_train, dtype=torch.long)
    for _ in range(20):
        perm = torch.randperm(len(u_train))
        for start in range(0, len(u_train), 16):
            ix = perm[start:start + 16]
            opt.zero_grad()
            loss_fn(model(u_train[ix]), yy[ix]).backward()
            opt.step()
    with torch.no_grad():
        return balanced_accuracy_score(y_test, model(u_test).argmax(dim=1).numpy())


def run(job):
    seed, folder = int(job), OUT / job
    if (folder / "complete.json").exists():
        return
    folder.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    rows = []

    def add(split, features, model, bacc):
        rows.append(dict(seed=seed, split=split, features=features, model=model, test_bacc=float(bacc)))
        print(job, split, features, model, round(float(bacc), 4), flush=True)

    x, y, xt, yt, _, _ = eeg_split(seed)
    add("recording", "z-scored window", "RBF-SVM", rbf_svm_scores(x, y, xt, yt, seed=seed)["test_bacc"])
    add("recording", "z-scored window", "QSM", qsm(x, y, xt, yt, seed))
    x, y, xt, yt, _, _ = eeg_split(seed, znorm=False, shuffle=False)
    add("recording", "raw window", "RBF-SVM", rbf_svm_scores(x, y, xt, yt, seed=seed)["test_bacc"])
    log_std = [np.log(a.std(axis=1, keepdims=True) + 1e-8) for a in (x, xt)]
    add("recording", "log std", "RBF-SVM", rbf_svm_scores(log_std[0], y, log_std[1], yt, seed=seed)["test_bacc"])
    x, y, xt, yt, rec, rec_test = eeg_split(seed, split="random")
    add("random", "z-scored window", "QSM", qsm(x, y, xt, yt, seed))
    pd.DataFrame(rows).to_csv(folder / "scores.csv", index=False)
    write_json(folder / "complete.json", dict(status="complete",
                                              random_split_shared_recording=float(np.isin(rec_test, rec).mean())))


if __name__ == "__main__":
    cli(JOBS, run, __doc__)
