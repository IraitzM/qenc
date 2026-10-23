"""Representation measurements of the regime map: fidelity and projected kernels, the
empirical tangent kernel, kernel-target alignment, linear CKA and the linearized model.

Kernels are computed from final states or reduced states of a whole batch at
once. The tangent kernel and the linearized model use ``torch.func`` and the
models' ``forward(x, output=...)`` hook.
"""
from __future__ import annotations

import torch
from torch.func import functional_call, jacrev, jvp, vmap


def fidelity_kernel(a: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
    """|<psi(x)|psi(x')>|^2 for state batches a (Na, D) and b (Nb, D)."""
    b = a if b is None else b
    return (a @ b.conj().transpose(0, 1)).abs() ** 2


def projected_kernel(ra: torch.Tensor, rb: torch.Tensor | None = None, gamma: float = 1.0) -> torch.Tensor:
    """exp(-gamma sum_q ||rho_q(x) - rho_q(x')||_F^2) from one-qubit reduced states (N, n, 2, 2)."""
    rb = ra if rb is None else rb
    va, vb = ra.reshape(len(ra), -1), rb.reshape(len(rb), -1)
    sq = (va.abs() ** 2).sum(1)[:, None] + (vb.abs() ** 2).sum(1)[None, :] - 2 * (va @ vb.conj().T).real
    return torch.exp(-gamma * sq.clamp_min(0))


def kta(k: torch.Tensor, y: torch.Tensor) -> float:
    """Centered kernel-target alignment, +-1 one-vs-rest targets (the ladder pilot's definition)."""
    labels = torch.unique(y)
    if len(labels) > 2:
        t = torch.stack([(y == c).to(k.dtype) * 2 - 1 for c in labels], 1)
    else:
        t = ((y == labels[1]).to(k.dtype) * 2 - 1)[:, None]
    n = len(k)
    h = torch.eye(n, dtype=k.dtype, device=k.device) - 1.0 / n
    kc, tc = h @ k @ h, h @ (t @ t.T) @ h
    return float((kc * tc).sum() / (torch.linalg.matrix_norm(kc) * torch.linalg.matrix_norm(tc)))


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    """Linear centered kernel alignment between two feature matrices with the same rows."""
    x = x - x.mean(0)
    y = y - y.mean(0)
    cross = torch.linalg.matrix_norm(x.T @ y) ** 2
    return float(cross / (torch.linalg.matrix_norm(x.T @ x) * torch.linalg.matrix_norm(y.T @ y)))


def relative_change(k0: torch.Tensor, k1: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(k1 - k0) / torch.linalg.matrix_norm(k0))


def _split_params(model, names=None):
    params = {k: v.detach() for k, v in model.named_parameters()}
    names = list(params) if names is None else list(names)
    return {k: params[k] for k in names}, {k: v for k, v in params.items() if k not in names}


def jacobian(model, x: torch.Tensor, output: str = "features", names=None, chunk: int | None = None) -> torch.Tensor:
    """Per-sample Jacobian of the model output with respect to the chosen parameters.

    Returns (N, outputs, P), the parameters flattened in ``names`` order
    (default: all parameters, ``weights`` first for the quantum part).
    """
    train, frozen = _split_params(model, names)
    # jacrev rejects complex arguments even when it does not differentiate them,
    # so complex inputs (QSM upload unitaries) travel as a real view.
    complex_input = x.is_complex()

    def single(p, xi):
        if complex_input:
            xi = torch.view_as_complex(xi)
        out = functional_call(model, {**p, **frozen}, (xi[None],), {"output": output})
        return out[0]

    parts = []
    for xs in (x.split(chunk) if chunk else (x,)):
        if complex_input:
            xs = torch.view_as_real(xs)
        j = vmap(jacrev(single), in_dims=(None, 0))(train, xs)
        parts.append(torch.cat([j[k].reshape(len(xs), j[k].shape[1], -1) for k in train], 2))
    return torch.cat(parts, 0)


def tangent_kernel(model, xa: torch.Tensor, xb: torch.Tensor | None = None, output: str = "features",
                   names=None, chunk: int | None = None) -> torch.Tensor:
    """Empirical tangent kernel summed over outputs: K(x, x') = sum_o <J_o(x), J_o(x')>."""
    ja = jacobian(model, xa, output, names, chunk)
    jb = ja if xb is None else jacobian(model, xb, output, names, chunk)
    return torch.einsum("aop,bop->ab", ja, jb)


class Linearized(torch.nn.Module):
    """First-order expansion of a model around its parameters at construction.

    f_lin(x; theta) = f(x; theta0) + J(x; theta0) (theta - theta0), trained like
    the model itself. Comparing a trained model with its trained linearization
    measures how far training leaves the tangent-kernel (lazy) regime.
    """

    def __init__(self, model, output: str = "logits"):
        super().__init__()
        self.model, self.output = model, output
        self.theta0 = {k: v.detach().clone() for k, v in model.named_parameters()}
        self.delta = torch.nn.ParameterDict({k.replace(".", "__"): torch.nn.Parameter(torch.zeros_like(v))
                                             for k, v in self.theta0.items()})

    def forward(self, x):
        def f(p):
            return functional_call(self.model, p, (x,), {"output": self.output})

        tangents = {k: self.delta[k.replace(".", "__")] for k in self.theta0}
        out0, dout = jvp(f, (self.theta0,), (tangents,))
        return out0 + dout
