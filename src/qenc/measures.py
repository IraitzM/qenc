"""What the ladder and the planted labels measure on a model: final states, Pauli and
readout features, fidelity kernels and their alignment, and the classical procedures
fitted to them. Also the tuned RBF-SVM used by the preprocessing controls.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

LOGISTIC_C = [0.01, 0.1, 1.0, 10.0, 100.0]
SVC_C = [0.1, 1.0, 10.0, 100.0, 1000.0]
GAMMA_SCALE = [0.1, 1.0, 10.0]
PAULI = {"X": np.array([[0, 1], [1, 0]], complex), "Y": np.array([[0, -1j], [1j, 0]]),
         "Z": np.array([[1, 0], [0, -1]], complex)}


# ---------------------------------------------------------------- states and features

def states(model, x, batch=64) -> np.ndarray:
    """Final statevectors of the model's own circuit, with the measurements replaced by qml.state()."""
    n, device, out = model.n_qubits, qml.device("default.qubit", wires=model.n_qubits), []
    with torch.no_grad():
        for part in x.split(batch):
            tape = qml.workflow.construct_tape(model._circuit)(model.circuit_input(part), model.weights)
            psi = qml.execute([qml.tape.QuantumScript(tape.operations, [qml.state()])], device)[0]
            psi = psi.detach().numpy() if torch.is_tensor(psi) else np.asarray(psi)
            out.append(psi.reshape(-1, 2 ** n))
    return np.concatenate(out)


def reduced(psi, wires, n):
    t = psi.reshape(len(psi), *([2] * n))
    t = np.moveaxis(t, [w + 1 for w in wires], list(range(1, len(wires) + 1))).reshape(len(psi), 2 ** len(wires), -1)
    return np.einsum("bar,bcr->bac", t, t.conj())


def expect(rho, op):
    return np.real(np.einsum("bac,ca->b", rho, op))


def readout_features(psi, n, zz):
    """Local Z, plus the ring of neighbouring ZZ correlators when ``zz`` (the angle readout)."""
    z = [expect(reduced(psi, [i], n), PAULI["Z"]) for i in range(n)]
    if zz:
        z += [expect(reduced(psi, [i, (i + 1) % n], n), np.kron(PAULI["Z"], PAULI["Z"])) for i in range(n)]
    return np.stack(z, 1)


def pauli_features(psi, n):
    """All one-local (3n) and two-local (9 n (n - 1) / 2) Pauli expectations."""
    cols = []
    for i in range(n):
        cols += [expect(reduced(psi, [i], n), PAULI[p]) for p in "XYZ"]
    for i in range(n):
        for j in range(i + 1, n):
            rho = reduced(psi, [i, j], n)
            cols += [expect(rho, np.kron(PAULI[p], PAULI[q])) for p in "XYZ" for q in "XYZ"]
    return np.stack(cols, 1)


def fidelity(a, b):
    return np.abs(a.conj() @ b.T) ** 2


def alignment(k, y):
    """Centered kernel-target alignment with +-1 one-vs-rest targets."""
    labels = np.unique(y)
    t = np.stack([(y == c) * 2.0 - 1 for c in labels], 1) if len(labels) > 2 else ((y == labels[1]) * 2.0 - 1)[:, None]
    h = np.eye(len(k)) - 1 / len(k)
    kc, tc = h @ k @ h, h @ (t @ t.T) @ h
    return float(np.sum(kc * tc) / (np.linalg.norm(kc) * np.linalg.norm(tc)))


# ---------------------------------------------------------------- classical fits

def inner_cv(y, groups, seed):
    """Five stratified folds of the fitting split, grouped by recording when groups are given."""
    if groups is not None:
        return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(y, y, groups))
    return list(StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y))


def fit_classical(a, b, y, yt, cv, seed):
    """Best of logistic regression and RBF-SVM by inner CV. Returns test BA, CV BA, choice and predictions."""
    gamma = 1.0 / (a.shape[1] * a.var()) if a.var() > 0 else 1.0
    searches = [
        GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed)),
                     {"logisticregression__C": LOGISTIC_C}, scoring="balanced_accuracy", cv=cv),
        GridSearchCV(make_pipeline(StandardScaler(), SVC()),
                     {"svc__C": SVC_C, "svc__gamma": [g * gamma for g in GAMMA_SCALE]},
                     scoring="balanced_accuracy", cv=cv),
    ]
    best = max((s.fit(a, y) for s in searches), key=lambda s: s.best_score_)
    pred = best.predict(b)
    return balanced_accuracy_score(yt, pred), best.best_score_, str(best.best_params_), pred


def fit_logistic(a, b, y, yt, cv, seed):
    search = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed)),
                          {"logisticregression__C": LOGISTIC_C}, scoring="balanced_accuracy", cv=cv).fit(a, y)
    pred = search.predict(b)
    return balanced_accuracy_score(yt, pred), search.best_score_, str(search.best_params_), pred


def fit_kernel(k_fit, k_test, y, yt, cv):
    search = GridSearchCV(SVC(kernel="precomputed"), {"C": SVC_C}, scoring="balanced_accuracy", cv=cv).fit(k_fit, y)
    pred = search.predict(k_test)
    return balanced_accuracy_score(yt, pred), search.best_score_, str(search.best_params_), pred


def rbf_svm_scores(x_train, y_train, x_test, y_test, tune=True, seed=0):
    """RBF-SVC with gamma chosen on a stratified quarter of the training draw, then refit on all of it.

    The 11-point grid spans three decades either side of scikit-learn's ``scale`` value;
    ``tune=False`` uses ``scale`` itself.
    """
    gamma = "scale"
    if tune:
        x_fit, x_val, y_fit, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=seed,
                                                      stratify=y_train)
        var = float(np.asarray(x_train).var())
        centre = 1.0 / (x_train.shape[1] * var) if x_train.shape[1] and var > 0 else 1.0
        best = -np.inf
        for g in np.geomspace(centre * 1e-3, centre * 1e3, 11):
            score = balanced_accuracy_score(y_val, SVC(kernel="rbf", gamma=g).fit(x_fit, y_fit).predict(x_val))
            if score > best:
                gamma, best = float(g), float(score)
    clf = SVC(kernel="rbf", gamma=gamma).fit(x_train, y_train)
    return {"test_bacc": balanced_accuracy_score(y_test, clf.predict(x_test)), "gamma": float(clf._gamma)}
