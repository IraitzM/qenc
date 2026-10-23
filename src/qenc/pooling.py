"""Fixed block averaging A, its projection A^+A and the share of energy it discards.

Blocks are disjoint and tile the input (kernel = stride, no padding). In 1-D a
remainder that does not fill a block is dropped, which is how 178-sample EEG
windows give eight means of 22 samples.
"""
from __future__ import annotations

from math import prod

import numpy as np
import torch
import torch.nn.functional as F


def block_geometry(shape, grid):
    shape, grid = tuple(shape), tuple(grid)
    if len(shape) not in (1, 2, 3) or len(shape) != len(grid):
        raise ValueError("Provide matching 1D, 2D or 3D shape and output grid")
    kernel = tuple(n // g for n, g in zip(shape, grid))
    if len(shape) > 1 and any(n % g for n, g in zip(shape, grid)):
        raise ValueError("Spatial grids must tile every pixel or voxel")
    return shape, grid, kernel


def box_features(x: torch.Tensor, shape, grid) -> torch.Tensor:
    """Block means of a batch of flat (or native) inputs, flattened row-major."""
    shape, grid, kernel = block_geometry(shape, grid)
    native = x.reshape(-1, *shape)
    weight = x.new_full((1, 1, *kernel), 1.0 / prod(kernel))
    convolution = (F.conv1d, F.conv2d, F.conv3d)[len(shape) - 1]
    return convolution(native[:, None], weight, stride=kernel).flatten(1)


def features(x: np.ndarray, shape, grid) -> np.ndarray:
    """NumPy front end of :func:`box_features`; ``grid=None`` returns the input unchanged."""
    if grid is None:
        return x
    return box_features(torch.tensor(x, dtype=torch.float64), shape, grid).numpy()


def box_projection(x: np.ndarray, shape, grid) -> np.ndarray:
    """A^+ A x: every block replaced by its mean (a 1-D remainder is set to zero)."""
    shape, grid, kernel = block_geometry(shape, grid)
    native = np.asarray(x, dtype=float).reshape(-1, *shape)
    slices = (slice(None),) + tuple(slice(0, g * k) for g, k in zip(grid, kernel))
    interleaved = tuple(v for pair in zip(grid, kernel) for v in pair)
    lifted = native[slices].reshape(-1, *interleaved).mean(axis=tuple(range(2, 2 * len(shape) + 1, 2)))
    for axis, k in enumerate(kernel, 1):
        lifted = np.repeat(lifted, k, axis=axis)
    projected = np.zeros_like(native)
    projected[slices] = lifted
    return projected.reshape(np.asarray(x).shape)


def compression_loss(x: np.ndarray, shape, grid) -> dict:
    """E_A = ||(I - A^+A) z||^2 / ||z||^2 of centered inputs, and the Fourier power outside the grid's band.

    The band is the nominal spectral rectangle |k_i| <= grid_i // 2 that a Fourier cutoff would keep.
    """
    shape, grid, _ = block_geometry(shape, grid)
    flat = np.asarray(x, dtype=float).reshape(-1, prod(shape))
    z = flat - flat.mean(axis=1, keepdims=True)
    loss = np.sum((z - box_projection(z, shape, grid)) ** 2, axis=1) / np.sum(z * z, axis=1)
    power = np.abs(np.fft.fftn(z.reshape(-1, *shape), axes=tuple(range(1, len(shape) + 1)))) ** 2
    power = power.reshape(len(z), -1)
    power /= power.sum(axis=1, keepdims=True)
    inside = np.ones(shape, dtype=bool)
    for f, g in zip(np.meshgrid(*[np.fft.fftfreq(n) * n for n in shape], indexing="ij"), grid):
        inside &= abs(f) <= g // 2 + 1e-10
    return {"projection_loss": loss, "spectral_tail": power[:, ~inside.ravel()].sum(axis=1)}


def pool_1d(x: torch.Tensor, n: int) -> torch.Tensor:
    """Means of ``n`` consecutive blocks of the last axis, dropping the remainder (identity when len == n)."""
    width = x.shape[-1] // n
    if width < 1:
        raise ValueError("Input must have at least n entries")
    return x[..., :width * n].reshape(*x.shape[:-1], n, width).mean(dim=-1)


def expanded_coarse(fine: np.ndarray, fine_grid, coarse_grid) -> np.ndarray:
    """P F: average the fine grid onto the coarse one and repeat the means back to the fine shape."""
    factors = tuple(f // c for f, c in zip(fine_grid, coarse_grid))
    interleaved = tuple(v for pair in zip(coarse_grid, factors) for v in pair)
    restored = fine.reshape(len(fine), *interleaved).mean(axis=tuple(range(2, 2 * len(fine_grid) + 1, 2)))
    for axis, factor in enumerate(factors, 1):
        restored = np.repeat(restored, factor, axis=axis)
    return restored
