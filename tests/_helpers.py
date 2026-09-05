"""Shared test helpers. Not part of the public package."""

from __future__ import annotations

import numpy as np


def align_rows(reference: np.ndarray, estimated: np.ndarray) -> np.ndarray:
    aligned = estimated.copy()
    for index in range(estimated.shape[0]):
        if np.dot(reference[index], estimated[index]) < 0:
            aligned[index] *= -1
    return aligned


def structured_problem(seed: int = 0, *, n_channels: int = 7, n_times: int = 36):
    rng = np.random.default_rng(seed)
    n_conditions = 5
    mixing = rng.standard_normal((n_channels, 3))
    sources = rng.standard_normal((3, n_times))
    sources = sources + np.array([[1.5], [-0.8], [0.4]])
    amplitudes = rng.standard_normal((n_conditions, 3))
    X = np.zeros((n_conditions, n_channels, n_times))
    for condition in range(n_conditions):
        X[condition] = mixing @ (amplitudes[condition, :, None] * sources)
        X[condition] += 0.03 * rng.standard_normal((n_channels, n_times))
        X[condition] += 0.2 * (condition + 1)
    y = np.abs(amplitudes @ amplitudes.T)
    y = 0.5 * (y + y.T)
    np.fill_diagonal(y, 0.0)
    return X, y


def snapshot_problem():
    """Fixed inputs used to lock default ``ReDisCA()`` numerics from main."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 8, 50))
    y = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    return X, y
