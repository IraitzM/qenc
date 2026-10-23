"""Inputs of the regime map: one class-balanced PneumoniaMNIST draw at three MedMNIST+
resolutions, and the representation each family receives.

Per-image z-scoring, block means and the multilevel trajectory matrix are those of
the spatial pilot, ported to torch.
"""
from __future__ import annotations

import numpy as np
import torch

from qenc.data import medmnist
from qenc.fastsim import sim


def znorm(x: torch.Tensor) -> torch.Tensor:
    """Per-sample z-score over all pixels (population SD)."""
    flat = x.reshape(len(x), -1)
    sd = flat.std(1, keepdim=True, unbiased=False)
    return ((flat - flat.mean(1, keepdim=True)) / torch.where(sd > 1e-12, sd, torch.ones_like(sd))).reshape(x.shape)


def block_means(images: torch.Tensor, grid) -> torch.Tensor:
    """Means of disjoint blocks on a grid that divides the image, flattened (row-major)."""
    n, a, b = images.shape
    ga, gb = grid
    return images.reshape(n, ga, a // ga, gb, b // gb).mean(dim=(2, 4)).reshape(n, ga * gb)


def trajectory_matrix(grids: torch.Tensor, shape) -> torch.Tensor:
    """H[i, j] = X[a_i + b_j], rows over patch offsets a (window floor(shape / 2)), columns over origins b."""
    a, b = shape
    wa, wb = max(1, a // 2), max(1, b // 2)
    patches = grids.reshape(len(grids), a, b).unfold(1, wa, 1).unfold(2, wb, 1)
    return patches.reshape(len(grids), (a - wa + 1) * (b - wb + 1), wa * wb).transpose(1, 2)


def qsm_qubits(shape) -> int:
    a, b = shape
    wa, wb = max(1, a // 2), max(1, b // 2)
    return max(1, (wa * wb + (a - wa + 1) * (b - wb + 1) - 1).bit_length())


def archive(size: int):
    return medmnist("pneumoniamnist" if size == 28 else f"pneumoniamnist_{size}")


def draw_indices(seed: int, n_fit: int, n_val: int):
    """Class-balanced fit and validation indices from the training split, and the full test split.

    Every MedMNIST+ size holds the same images in the same order, so one draw serves all resolutions.
    """
    with np.load(archive(28)) as raw:
        y = raw["train_labels"].ravel()
        n_test = len(raw["test_labels"])
    rng = np.random.default_rng(seed)
    fit, val = [], []
    for c in (0, 1):
        ix = rng.permutation(np.flatnonzero(y == c))
        fit.append(ix[: n_fit // 2])
        val.append(ix[n_fit // 2: n_fit // 2 + n_val // 2])
    return {"fit": np.sort(np.concatenate(fit)), "val": np.sort(np.concatenate(val)), "test": np.arange(n_test)}


def load_images(size: int, indices) -> dict:
    """z-scored float32 images and labels per split."""
    out = {}
    with np.load(archive(size)) as raw:
        for split, source in (("fit", "train"), ("val", "train"), ("test", "test")):
            ix = indices[split]
            images = torch.tensor(raw[f"{source}_images"][ix], dtype=torch.float32) / 255.0
            if images.ndim == 4:  # some sizes carry a channel axis
                images = images[..., 0]
            out[split] = (znorm(images.double()).float(), torch.tensor(raw[f"{source}_labels"][ix].ravel()).long())
    return out


def representation(family: str, images: torch.Tensor, grid, device="cpu", time: float = 2.0):
    """angle and amplitude: block means on ``grid`` (the native image when the grid equals its size).
    qsm: fixed-time uploads of the Frobenius-normalized trajectory matrix of those means, in double precision.
    """
    grid = tuple(grid)
    means = images.reshape(len(images), -1) if tuple(images.shape[1:]) == grid else block_means(images, grid)
    if family in ("angle", "amplitude"):
        return means
    n = qsm_qubits(grid)
    parts = []
    for chunk in means.split(64):
        h = trajectory_matrix(chunk.to(device, torch.float64), grid)
        parts.append(sim.block_unitary(h, time, n, normalize=True).to(torch.complex64).cpu())
    return torch.cat(parts)
