"""Batched statevector primitives in torch, gate for gate equal to PennyLane.

A batch of states has shape (batch, 2**n). Wire 0 is the most significant bit
of the basis index, as in PennyLane, so the state of wires (0, 1) is the
Kronecker product with wire 0 on the left. Every function is differentiable
with autograd and works unchanged on CUDA. Gates are either shared by the
whole batch, with shape (2, 2) or (4, 4), or given per sample, with a leading
batch dimension.
"""
from __future__ import annotations

from itertools import product

import torch


def complex_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.complex128 if dtype in (torch.float64, torch.complex128) else torch.complex64


def real_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float64 if dtype in (torch.float64, torch.complex128) else torch.float32


def cis(angle: torch.Tensor) -> torch.Tensor:
    """exp(i angle), differentiable and dtype-preserving."""
    return torch.polar(torch.ones_like(angle), angle)


# ---------------------------------------------------------------- matrices

def ry_matrix(angle: torch.Tensor) -> torch.Tensor:
    """RY for each entry of ``angle``: shape angle.shape + (2, 2)."""
    c, s = torch.cos(angle / 2), torch.sin(angle / 2)
    m = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
    return m.to(complex_dtype(angle.dtype))


def rot_matrix(phi: torch.Tensor, theta: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
    """PennyLane Rot(phi, theta, omega) = RZ(omega) RY(theta) RZ(phi)."""
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    plus, minus = (phi + omega) / 2, (phi - omega) / 2
    cd = complex_dtype(theta.dtype)
    a = cis(-plus) * c.to(cd)
    b = -cis(minus) * s.to(cd)
    d = cis(-minus) * s.to(cd)
    e = cis(plus) * c.to(cd)
    return torch.stack([torch.stack([a, b], -1), torch.stack([d, e], -1)], -2)


def pauli(label: str, dtype=torch.complex128, device=None) -> torch.Tensor:
    table = {"I": [[1, 0], [0, 1]], "X": [[0, 1], [1, 0]], "Y": [[0, -1j], [1j, 0]], "Z": [[1, 0], [0, -1]]}
    return torch.tensor(table[label], dtype=dtype, device=device)


def two_qubit_paulis(dtype=torch.complex128, device=None) -> torch.Tensor:
    """The 15 non-identity two-qubit Pauli words in PennyLane's SpecialUnitary order."""
    words = [a + b for a, b in product("IXYZ", repeat=2)][1:]
    return torch.stack([torch.kron(pauli(w[0], dtype, device), pauli(w[1], dtype, device)) for w in words])


def special_unitary(theta: torch.Tensor) -> torch.Tensor:
    """PennyLane SpecialUnitary on two wires: exp(+i sum_k theta_k P_k), theta.shape[-1] == 15."""
    basis = two_qubit_paulis(complex_dtype(theta.dtype), theta.device)
    generator = torch.einsum("...k,kij->...ij", theta.to(basis.dtype), basis)
    return torch.linalg.matrix_exp(1j * generator)


# ---------------------------------------------------------------- states and gates

def zero_state(batch: int, n: int, dtype=torch.complex64, device=None) -> torch.Tensor:
    psi = torch.zeros(batch, 2 ** n, dtype=dtype, device=device)
    psi[:, 0] = 1
    return psi


def amplitude_state(x: torch.Tensor, n_qubits: int, copies: int = 1) -> torch.Tensor:
    """Normalized amplitude embedding, zero-padded to 2**n_qubits, on ``copies`` registers.

    PennyLane AmplitudeEmbedding(normalize=True, pad_with=0) on each register,
    the first register holding the most significant wires.
    """
    width = 2 ** n_qubits
    if x.shape[1] > width:
        raise ValueError(f"{x.shape[1]} features do not fit {n_qubits} qubits")
    padded = torch.nn.functional.pad(x, (0, width - x.shape[1]))
    one = padded / torch.linalg.vector_norm(padded, dim=1, keepdim=True)
    one = one.to(complex_dtype(x.dtype))
    state = one
    for _ in range(copies - 1):
        state = torch.einsum("bi,bj->bij", state, one).reshape(len(x), -1)
    return state


def apply_1q(psi: torch.Tensor, gate: torch.Tensor, wire: int, n: int) -> torch.Tensor:
    s = psi.reshape(psi.shape[0], 2 ** wire, 2, 2 ** (n - wire - 1))
    spec = "ij,bajc->baic" if gate.dim() == 2 else "bij,bajc->baic"
    return torch.einsum(spec, gate, s).reshape(psi.shape)


def apply_2q_adjacent(psi: torch.Tensor, gate: torch.Tensor, wire: int, n: int) -> torch.Tensor:
    """A 4x4 gate on wires (wire, wire + 1), wire the more significant one."""
    s = psi.reshape(psi.shape[0], 2 ** wire, 4, 2 ** (n - wire - 2))
    spec = "ij,bajc->baic" if gate.dim() == 2 else "bij,bajc->baic"
    return torch.einsum(spec, gate, s).reshape(psi.shape)


def apply_cnot(psi: torch.Tensor, control: int, target: int, n: int) -> torch.Tensor:
    s = psi.reshape(psi.shape[0], *([2] * n))
    off, on = s.select(control + 1, 0), s.select(control + 1, 1)
    axis = target + 1 if target < control else target  # the control axis is gone in ``on``
    return torch.stack([off, on.flip(axis)], dim=control + 1).reshape(psi.shape)


def apply_unitary(psi: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """A unitary on all wires: shared (dim, dim) or per sample (batch, dim, dim)."""
    if u.dim() == 2:
        return psi @ u.transpose(0, 1)
    return torch.einsum("bij,bj->bi", u, psi)


def strongly_entangling(psi: torch.Tensor, weights: torch.Tensor, n: int, ranges=None) -> torch.Tensor:
    """PennyLane StronglyEntanglingLayers with CNOT imprimitives; weights (layers, n, 3)."""
    layers = weights.shape[0]
    if ranges is None:
        ranges = [(l % (n - 1)) + 1 for l in range(layers)] if n > 1 else [0] * layers
    for l in range(layers):
        for i in range(n):
            psi = apply_1q(psi, rot_matrix(weights[l, i, 0], weights[l, i, 1], weights[l, i, 2]), i, n)
        if n > 1:
            for i in range(n):
                psi = apply_cnot(psi, i, (i + ranges[l]) % n, n)
    return psi


def angle_embedding_y(psi: torch.Tensor, features: torch.Tensor, n: int) -> torch.Tensor:
    """PennyLane AngleEmbedding(rotation="Y"): RY(features[:, i]) on wire i."""
    gates = ry_matrix(features)  # (batch, n, 2, 2)
    for i in range(n):
        psi = apply_1q(psi, gates[:, i], i, n)
    return psi


# ---------------------------------------------------------------- measurements

def probabilities(psi: torch.Tensor, n: int) -> torch.Tensor:
    return (psi.real ** 2 + psi.imag ** 2).reshape(psi.shape[0], *([2] * n))


def z_expectations(psi: torch.Tensor, n: int) -> torch.Tensor:
    p = psi.real ** 2 + psi.imag ** 2
    out = []
    for i in range(n):
        m = p.reshape(p.shape[0], 2 ** i, 2, 2 ** (n - i - 1)).sum(dim=(1, 3))
        out.append(m[:, 0] - m[:, 1])
    return torch.stack(out, 1)


def zz_expectations(psi: torch.Tensor, n: int, pairs) -> torch.Tensor:
    probs = probabilities(psi, n)
    out = []
    for i, j in pairs:
        others = [d for d in range(1, n + 1) if d not in (i + 1, j + 1)]
        m = probs.sum(dim=others) if others else probs
        if i > j:
            m = m.transpose(1, 2)
        out.append(m[:, 0, 0] + m[:, 1, 1] - m[:, 0, 1] - m[:, 1, 0])
    return torch.stack(out, 1)


def ring_pairs(n: int):
    """The neighbouring pairs (i, i + 1 mod n) of AngleEncodingClassifier's ZZ readout."""
    return [(i, (i + 1) % n) for i in range(n)]


def rdm_1q(psi: torch.Tensor, n: int) -> torch.Tensor:
    """One-qubit reduced density matrices, shape (batch, n, 2, 2)."""
    out = []
    for w in range(n):
        s = psi.reshape(psi.shape[0], 2 ** w, 2, 2 ** (n - w - 1))
        out.append(torch.einsum("baic,bajc->bij", s, s.conj()))
    return torch.stack(out, 1)


# ---------------------------------------------------------------- QSM uploads

def symmetric_eigh(b: torch.Tensor, chunk: int = 32, backend: str = "auto"):
    """Eigendecomposition of real symmetric matrices in float64, robust on CUDA.

    Batched cuSOLVER eigh can fail on large blocks (CUSOLVER_STATUS_INVALID_VALUE
    on 256 matrices of size 1024 on an A10), and float32 leaves the upload
    unitary only unitary to about 1e-5. So the decomposition always runs in
    float64, in chunks, and with backend="auto" a chunk that the device solver
    rejects is redone on the CPU with LAPACK. backend="cpu" always uses LAPACK.
    Results come back on b's device in float64.
    """
    values, vectors = [], []
    for part in b.to(torch.float64).split(chunk):
        if backend == "cpu" or part.device.type == "cpu":
            v, w = torch.linalg.eigh(part.cpu())
        else:
            try:
                v, w = torch.linalg.eigh(part)
            except RuntimeError:  # torch.linalg.LinAlgError and cuSOLVER status errors
                v, w = torch.linalg.eigh(part.cpu())
        values.append(v.to(b.device))
        vectors.append(w.to(b.device))
    return torch.cat(values), torch.cat(vectors)


def block_unitary(h: torch.Tensor, time: float, n_qubits: int, normalize: bool = True,
                  chunk: int = 32, backend: str = "auto") -> torch.Tensor:
    """exp(-i time B / 2) for the block Hamiltonian B = [[0, H], [H^T, 0]], zero-padded.

    Mirrors ``qenc.models.block_upload`` (normalize=True, Frobenius). h: (batch, m, k) real.
    The eigendecomposition and the unitary are formed in double precision
    (see ``symmetric_eigh``) and returned in the complex type matching h.
    """
    batch, m, k = h.shape
    dim = 2 ** n_qubits
    if m + k > dim:
        raise ValueError("Block does not fit the qubit register")
    out_dtype = complex_dtype(h.dtype)
    h = h.to(torch.float64)
    if normalize:
        norm = torch.linalg.matrix_norm(h)
        h = h / torch.where(norm > 0, norm, torch.ones_like(norm))[:, None, None]
    b = torch.zeros(batch, dim, dim, dtype=torch.float64, device=h.device)
    b[:, :m, m:m + k] = h
    b[:, m:m + k, :m] = h.transpose(1, 2)
    values, vectors = symmetric_eigh(b, chunk=chunk, backend=backend)
    phases = cis(-0.5 * time * values)
    v = vectors.to(torch.complex128)
    return ((v * phases[:, None, :]) @ v.transpose(1, 2)).to(out_dtype)
