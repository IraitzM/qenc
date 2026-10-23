"""Torch models matching the PennyLane classifiers gate for gate.

| Model | PennyLane counterpart |
|---|---|
| AngleReupload | qenc.models.AngleClassifier (1-D pooling) |
| Amplitude | qenc.models.AmplitudeClassifier |
| QSM | qenc.models.QSMClassifier |

Every model exposes ``state(x)`` (final statevector), ``features(x)`` (measured
features) and ``forward(x)`` (logits), with ``forward(x, output="features")`` for
``torch.func`` transforms. Readouts: "z" (local Z) and "z+zz" (local Z and the ring
of neighbouring ZZ, the angle readout).
"""
from __future__ import annotations

import math

import torch
from torch import nn

from qenc.fastsim import sim


def n_features(readout: str, n: int) -> int:
    return {"z": n, "z+zz": 2 * n}[readout]


def measure(psi: torch.Tensor, n: int, readout: str) -> torch.Tensor:
    if readout == "z":
        return sim.z_expectations(psi, n)
    return torch.cat([sim.z_expectations(psi, n), sim.zz_expectations(psi, n, sim.ring_pairs(n))], 1)


def block_means(x: torch.Tensor, n: int) -> torch.Tensor:
    """Trim, then means of ``n`` consecutive blocks (inputs of length n pass unchanged)."""
    width = x.shape[-1] // n
    return x[..., :width * n].reshape(*x.shape[:-1], n, width).mean(-1)


class _Base(nn.Module):
    n_qubits: int
    readout_kind: str

    def features(self, x):
        return measure(self.state(x), self.n_qubits, self.readout_kind)

    def forward(self, x, output: str = "logits"):
        f = self.features(x)
        return f if output == "features" else self.readout(f)


class AngleReupload(_Base):
    """(RY(u) on every wire, then StronglyEntanglingLayers) x uploads, on ``n_qubits`` block means."""

    def __init__(self, n_qubits: int, n_reuploads: int = 4, n_classes: int = 2, readout: str = "z+zz",
                 init_scale: float = 0.1, dtype=torch.float32):
        super().__init__()
        self.n_qubits, self.n_reuploads, self.readout_kind = n_qubits, n_reuploads, readout
        self.weights = nn.Parameter(init_scale * torch.randn(n_reuploads, 1, n_qubits, 3, dtype=dtype))
        self.readout = nn.Linear(n_features(readout, n_qubits), n_classes, dtype=dtype)

    def state(self, x):
        u = block_means(x, self.n_qubits).to(self.weights.dtype)
        psi = sim.zero_state(len(x), self.n_qubits, sim.complex_dtype(self.weights.dtype), x.device)
        for r in range(self.n_reuploads):
            psi = sim.strongly_entangling(sim.angle_embedding_y(psi, u, self.n_qubits), self.weights[r], self.n_qubits)
        return psi


class Amplitude(_Base):
    """Amplitude embedding, then StronglyEntanglingLayers on all wires."""

    def __init__(self, input_dim: int, n_layers: int = 3, n_classes: int = 2, readout: str = "z",
                 init_scale: float = 0.1, dtype=torch.float32):
        super().__init__()
        self.readout_kind = readout
        self.n_qubits = max(1, math.ceil(math.log2(input_dim)))
        self.weights = nn.Parameter(init_scale * torch.randn(n_layers, self.n_qubits, 3, dtype=dtype))
        self.readout = nn.Linear(n_features(readout, self.n_qubits), n_classes, dtype=dtype)

    def state(self, x):
        psi = sim.amplitude_state(x.to(self.weights.dtype), self.n_qubits)
        n = self.n_qubits
        # Layer by layer, with StronglyEntanglingLayers' default ranges.
        ranges = [(l % (n - 1)) + 1 for l in range(len(self.weights))] if n > 1 else [0] * len(self.weights)
        for w, r in zip(self.weights, ranges):
            psi = sim.strongly_entangling(psi, w[None], n, ranges=[r])
        return psi


class QSM(_Base):
    """Fixed-time spectral uploads interleaved with brick-wall SU(4) mixers (even bonds, then odd).

    ``forward`` takes the precomputed upload unitaries (batch, 2**n, 2**n); see ``sim.block_unitary``.
    """

    def __init__(self, n_qubits: int, n_uploads: int = 4, n_classes: int = 2, readout: str = "z",
                 init_scale: float = 0.1, dtype=torch.float64):
        super().__init__()
        self.n_qubits, self.n_uploads, self.readout_kind = n_qubits, n_uploads, readout
        self.weights = nn.Parameter(init_scale * torch.randn(n_uploads, n_qubits - 1, 15, dtype=dtype))
        self.bonds = list(range(0, n_qubits - 1, 2)) + list(range(1, n_qubits - 1, 2))
        self.readout = nn.Linear(n_features(readout, n_qubits), n_classes, dtype=dtype)

    def state(self, u):
        psi = sim.zero_state(len(u), self.n_qubits, u.dtype, u.device)
        for r in range(self.n_uploads):
            psi = sim.apply_unitary(psi, u)
            for pair, gate in zip(self.bonds, sim.special_unitary(self.weights[r].to(sim.real_dtype(u.dtype)))):
                psi = sim.apply_2q_adjacent(psi, gate, pair, self.n_qubits)
        return psi
