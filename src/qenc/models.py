"""The three encoding families as PennyLane classifiers, and the QSM upload unitaries.

Each model has trainable ``weights`` followed by an affine ``readout``. Parameter
names, shapes and initialization order match the checkpoints in results/, so
``load_state_dict`` restores any saved model and ``torch.manual_seed`` before the
constructor reproduces its initialization. Every model exposes ``features(x)``,
the measured expectation values fed to the readout, and ``circuit_input(x)``,
what its circuit receives.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import torch
from torch import nn

from qenc.pooling import pool_1d


class AngleClassifier(nn.Module):
    """Angle encoding with data re-uploading (Schuld, Sweke and Meyer, 2021).

    Each of ``n_reuploads`` blocks applies RY(u_i) on wire i and one layer of
    StronglyEntanglingLayers (rotations and a CNOT ring); ``entangling=False``
    removes the CNOTs at equal parameter count. The input is first averaged into
    ``n_qubits`` consecutive block means, which leaves an input of length
    ``n_qubits`` unchanged. The readout sees local Z and, with ``zz_readout``,
    the ring of neighbouring ZZ correlators.
    """

    def __init__(self, n_qubits: int = 8, n_reuploads: int = 4, n_classes: int = 2, entangling: bool = True,
                 zz_readout: bool = False, device_name: str = "lightning.qubit"):
        super().__init__()
        self.n_qubits = n_qubits
        self.dev = qml.device(device_name, wires=n_qubits)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=n_qubits)
        self.weights = nn.Parameter(0.1 * torch.randn(n_reuploads, *shape))

        def block(w):
            if entangling:
                qml.StronglyEntanglingLayers(w, wires=range(n_qubits))
            else:
                for wire in range(n_qubits):
                    qml.Rot(*w[0, wire], wires=wire)

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(u, weights):
            for r in range(n_reuploads):
                qml.AngleEmbedding(u, wires=range(n_qubits), rotation="Y")
                block(weights[r])
            observables = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            if zz_readout:
                observables += [qml.expval(qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits)) for i in range(n_qubits)]
            return observables

        self._circuit = circuit
        self.readout = nn.Linear(n_qubits * (2 if zz_readout else 1), n_classes)

    def circuit_input(self, x):
        return pool_1d(x, self.n_qubits)

    def features(self, x):
        u = self.circuit_input(x)
        return torch.stack([torch.stack(self._circuit(u[i], self.weights)).reshape(-1) for i in range(len(u))])

    def forward(self, x):
        return self.readout(self.features(x))


class AmplitudeClassifier(nn.Module):
    """Amplitude embedding of the full input on ceil(log2 d) qubits, then StronglyEntanglingLayers.

    With ``copies=k`` the normalized input is prepared on k registers, so the
    measured scores become degree-2k forms of the normalized input instead of
    quadratic ones. The readout sees local Z on every wire.
    """

    def __init__(self, input_dim: int, copies: int = 1, n_layers: int = 3, n_classes: int = 2,
                 device_name: str = "lightning.qubit"):
        super().__init__()
        self.n_per = max(1, int(np.ceil(np.log2(input_dim))))
        self.n_qubits = n_qubits = copies * self.n_per
        self.dev = qml.device(device_name, wires=n_qubits)
        self.weights = nn.Parameter(0.1 * torch.randn(*qml.StronglyEntanglingLayers.shape(n_layers, n_qubits)))
        n_per = self.n_per

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def circuit(x, weights):
            for c in range(copies):
                qml.AmplitudeEmbedding(x, wires=range(c * n_per, (c + 1) * n_per), normalize=True, pad_with=0.0)
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = circuit
        self.readout = nn.Linear(n_qubits, n_classes)

    def circuit_input(self, x):
        return x

    def features(self, x):
        return torch.stack([torch.stack(self._circuit(x[i], self.weights)).reshape(-1) for i in range(len(x))])

    def forward(self, x):
        return self.readout(self.features(x))


class QSMClassifier(nn.Module):
    """Quantum spectral model (Wang et al.) with a fixed upload time.

    The input is the precomputed upload unitary U(x) of :func:`hankel_upload` or
    :func:`block_upload`. Each of ``n_uploads`` layers applies U(x) and a brick
    wall of trainable SU(4) gates on neighbouring wires (even bonds, then odd);
    local Z expectations feed the readout. ``batched`` evaluates a batch of
    unitaries in one broadcast circuit call instead of one call per example.
    """

    def __init__(self, n_qubits: int, n_uploads: int = 8, n_classes: int = 2, batched: bool = False,
                 device_name: str = "default.qubit"):
        super().__init__()
        self.n_qubits, self.batched = n_qubits, batched
        self.dev = qml.device(device_name, wires=n_qubits)
        self.weights = nn.Parameter(0.1 * torch.randn(n_uploads, n_qubits - 1, 15, dtype=torch.float64))
        bonds = list(range(0, n_qubits - 1, 2)) + list(range(1, n_qubits - 1, 2))

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(u, weights):
            for r in range(n_uploads):
                qml.QubitUnitary(u, wires=range(n_qubits))
                for pair, w in zip(bonds, weights[r]):
                    qml.SpecialUnitary(w, wires=[pair, pair + 1])
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self._circuit = circuit
        self.readout = nn.Linear(n_qubits, n_classes, dtype=torch.float64)

    @staticmethod
    def n_qubits_for(length: int) -> int:
        """Register of a length-d signal: the block generator has dimension d + 1."""
        return int(np.ceil(np.log2(length + 1)))

    def circuit_input(self, u):
        return u

    def features(self, u):
        if self.batched:
            return torch.stack(self._circuit(u, self.weights), dim=-1)
        return torch.stack([torch.stack(self._circuit(u[i], self.weights)).reshape(-1) for i in range(len(u))])

    def forward(self, u):
        return self.readout(self.features(u))


# ---------------------------------------------------------------- QSM uploads

def hankel_upload(x: np.ndarray, t: float = 0.1) -> np.ndarray:
    """exp(-i t B / 2) for a batch of 1-D signals, unnormalized (signal QSM).

    H[i, j] = x[i + j] with window d // 2, B = [[0, H], [H^T, 0]] zero-padded to
    the next power of two (B + 0, hence U + I on unused levels).
    """
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    window = d // 2
    k_cols = d - window + 1
    traj = np.lib.stride_tricks.as_strided(x, shape=(n, window, k_cols),
                                           strides=(x.strides[0], x.strides[1], x.strides[1]))
    p = window + k_cols
    dim = int(2 ** np.ceil(np.log2(p)))
    b = np.zeros((n, dim, dim), dtype=np.float64)
    b[:, :window, window:p] = traj
    b[:, window:p, :window] = np.swapaxes(traj, 1, 2)
    values, vectors = np.linalg.eigh(b)
    u = vectors * np.exp(-0.5j * t * values)[:, None, :]
    return u @ np.swapaxes(vectors.conj(), 1, 2)


def _geometry(shape):
    window = tuple(max(1, n // 2) for n in shape)
    return window, tuple(n - w + 1 for n, w in zip(shape, window))


def trajectory_matrix(x: np.ndarray, shape) -> np.ndarray:
    """Multilevel trajectory matrices H[a, b] = X[a + b] of a batch; columns are overlapping patches."""
    shape = tuple(shape)
    window, origins = _geometry(shape)
    native = np.asarray(x, dtype=float).reshape(-1, *shape)
    patches = np.lib.stride_tricks.sliding_window_view(native, window, axis=tuple(range(1, len(shape) + 1)))
    return patches.reshape(len(native), int(np.prod(origins)), int(np.prod(window))).transpose(0, 2, 1).copy()


def required_qubits(shape) -> int:
    window, origins = _geometry(tuple(shape))
    return max(1, (int(np.prod(window)) + int(np.prod(origins)) - 1).bit_length())


def block_upload(h: np.ndarray, time: float = 2.0, n_qubits: int | None = None) -> np.ndarray:
    """exp(-i time B / 2) of a batch of trajectory matrices divided by their Frobenius norm (spatial QSM)."""
    h = np.asarray(h, dtype=float)
    m, n = h.shape[1:]
    nq = max(1, (m + n - 1).bit_length()) if n_qubits is None else n_qubits
    norm = np.linalg.norm(h, axis=(1, 2))
    matrix = h / np.where(norm > 0, norm, 1)[:, None, None]
    dim = 2 ** nq
    b = np.zeros((len(h), dim, dim))
    b[:, :m, m:m + n] = matrix
    b[:, m:m + n, :m] = matrix.transpose(0, 2, 1)
    values, vectors = np.linalg.eigh(b)
    return (vectors * np.exp(-.5j * time * values)[:, None, :]) @ vectors.transpose(0, 2, 1)


def to_nonneg_domain(x: np.ndarray) -> np.ndarray:
    """Per-sample min-max scaling to [0, 1] before amplitude embedding (HLS control)."""
    lo, hi = x.min(axis=-1, keepdims=True), x.max(axis=-1, keepdims=True)
    return (x - lo) / (hi - lo + 1e-8)
