"""Controlled-rank regressions with a two-qubit entangler ladder (SI Section S10).

A two-qubit circuit re-uploads x on wire 0 and y on wire 1; ``k`` CRZ entanglers
placed in interior layers raise the rank of the 2-D Fourier coefficient matrix it
can reach, min(2L + 1, 4 * 3**(k - 1)), while the frequency support stays the
(2L + 1)^2 box. Targets of exact rank and fixed bandwidth isolate rank from bandwidth.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from torch import nn


def reachable_rank_law(k: int, degree: int) -> int:
    return int(min(2 * degree + 1, 1 if k == 0 else 4 * 3 ** (k - 1)))


def recommended_placement(k: int, degree: int) -> tuple[int, ...]:
    """``k`` entanglers spread evenly over the interior layers (layer 0 is the weakest slot)."""
    if k <= 0:
        return ()
    if k >= degree:
        return tuple(range(degree))
    interior = list(range(1, degree)) or [0]
    return tuple(sorted({interior[int(round(i))] for i in np.linspace(0, len(interior) - 1, k)}))


def controlled_rank_target(rank: int, bandwidth: int, n_grid: int, rng) -> np.ndarray:
    """U diag(1) V^T with orthonormal frames inside the band-limited subspace, scaled to unit SD.

    The algebraic and the effective rank both equal ``rank``; the mean is kept,
    since centering would add a rank-one constant matrix.
    """
    axis = np.arange(n_grid) * 2 * np.pi / n_grid
    cols = [np.ones_like(axis)]
    for m in range(1, bandwidth + 1):
        cols += [np.cos(m * axis), np.sin(m * axis)]
    basis = np.stack(cols, axis=1)
    basis /= np.linalg.norm(basis, axis=0, keepdims=True)
    box = 2 * bandwidth + 1

    def frame():
        q, _ = np.linalg.qr(rng.standard_normal((box, rank)))
        return basis @ q

    out = (frame() * np.ones(rank)) @ frame().T
    return out / (out.std() + 1e-12)


def truncated_svd_r2(target: np.ndarray, rank: int) -> float:
    u, s, vt = np.linalg.svd(target, full_matrices=False)
    recon = (u[:, :rank] * s[:rank]) @ vt[:rank, :]
    ss_tot = float(np.sum((target - target.mean()) ** 2)) + 1e-12
    return float(np.clip(1 - float(np.sum((target - recon) ** 2)) / ss_tot, 0.0, 1.0))


def separable_plus_const_r2(img: np.ndarray, n_coarse: int = 401, n_fine: int = 401) -> float:
    """Best R^2 of s g(x) h(y) + b, the product-ansatz class: rank-one SVD of img - b, minimized over b."""

    def r2(b):
        u, s, vt = np.linalg.svd(img - b, full_matrices=False)
        recon = np.outer(u[:, 0] * s[0], vt[0, :]) + b
        return float(1 - np.sum((img - recon) ** 2) / (np.sum((img - img.mean()) ** 2) + 1e-12))

    span = 3.0 * img.std() + 1e-12
    coarse = np.linspace(-span, span, n_coarse)
    scores = np.array([r2(b) for b in coarse])
    best = int(np.argmax(scores))
    step = coarse[1] - coarse[0]
    fine = np.linspace(coarse[best] - step, coarse[best] + step, n_fine)
    return float(np.clip(max(scores[best], max(r2(b) for b in fine)), 0.0, 1.0))


def model_ceiling_r2(target: np.ndarray, k: int, degree: int) -> float:
    """Separable-plus-constant ceiling at k = 0; truncated SVD at the model's rank (law + 1 for the bias) above."""
    if k == 0:
        return separable_plus_const_r2(target)
    return truncated_svd_r2(target, int(min(reachable_rank_law(k, degree) + 1, 2 * degree + 1)))


class EntanglerLadder(nn.Module):
    """scale * <Z0 Z1> + bias of a two-qubit re-uploading circuit with CRZ entanglers in ``ent_layers``.

    A CRZ at angle zero is the identity, so every rung contains the one below it.
    """

    def __init__(self, degree: int, ent_layers=()):
        super().__init__()
        self.rot_weights = nn.Parameter(torch.randn(degree + 1, 2, 3, dtype=torch.float64))
        self.ent_weights = nn.Parameter(torch.randn(max(degree, 1), dtype=torch.float64))
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
        layers = set(ent_layers)

        @qml.qnode(qml.device("default.qubit", wires=2), interface="torch", diff_method="backprop")
        def circuit(x, y, rot_w, ent_w):
            qml.Rot(*rot_w[0, 0], wires=0)
            qml.Rot(*rot_w[0, 1], wires=1)
            for layer in range(degree):
                qml.RZ(x, wires=0)
                qml.RZ(y, wires=1)
                if layer in layers:
                    qml.CRZ(ent_w[layer], wires=[0, 1])
                qml.Rot(*rot_w[layer + 1, 0], wires=0)
                qml.Rot(*rot_w[layer + 1, 1], wires=1)
            return qml.expval(qml.PauliZ(0) @ qml.PauliZ(1))

        self._circuit = circuit

    def forward(self, xs, ys):
        return self.scale * self._circuit(xs, ys, self.rot_weights, self.ent_weights) + self.bias
