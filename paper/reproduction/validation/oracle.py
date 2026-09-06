"""Independent AIRI → stock-SPoC reference calculation.

This module is a validation oracle only. Experiment runners must not import
it. Production fits use ``redisca.ReDisCA`` via ``common.method.make_redisca``.

Formulas follow pinned sources:

- AIRI ``Redisca_tools_faces_3_random_norm_correct.m`` @ 15bc19c
- stock SPoC ``spoc.m``, ``whiten_data.m``, ``create_Cxxz.m`` @ 18e4754

NumPy ``eigh`` / RNG are not MATLAB ``eig`` / ``rand`` bitwise parity.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def directed_pairs(n_conditions: int) -> list[tuple[int, int]]:
    return [
        (i, j)
        for i in range(n_conditions)
        for j in range(n_conditions)
        if i != j
    ]


def matlab_cov(delta_channels_by_time: NDArray[np.floating]) -> NDArray[np.float64]:
    """MATLAB ``cov`` of a ``(T, n_channels)`` epoch stored as ``(n_channels, T)``."""
    delta = np.asarray(delta_channels_by_time, dtype=np.float64)
    n_times = delta.shape[-1]
    if n_times < 2:
        raise ValueError("MATLAB cov requires at least two time samples.")
    centered = delta - delta.mean(axis=-1, keepdims=True)
    return (centered @ centered.T) / (n_times - 1)


def matlab_zscore(values: NDArray[np.floating]) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=np.float64).ravel()
    centered = values - np.mean(values)
    scale = float(np.sqrt(np.sum(centered * centered) / (values.size - 1)))
    if scale == 0.0:
        raise ValueError("SPoC target vector has zero sample standard deviation.")
    return centered / scale


def independent_airi_spoc(
    X: NDArray[np.floating],
    y: NDArray[np.floating],
    *,
    rank_tol: float = 1e-6,
) -> dict[str, NDArray[np.float64]]:
    """Directed-pair MATLAB-cov + explicit-whitening SPoC.

    ``X`` is ``(n_conditions, n_channels, n_times)``.
    ``y`` is a symmetric zero-diagonal RDM.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n_conditions, n_channels, _n_times = X.shape
    pairs = directed_pairs(n_conditions)
    covariances = np.stack([matlab_cov(X[i] - X[j]) for i, j in pairs], axis=0)
    cxx = covariances.mean(axis=0)
    values = np.array([y[i, j] for i, j in pairs], dtype=np.float64)
    z = matlab_zscore(values)
    cxxz = np.mean(z[:, None, None] * (covariances - cxx), axis=0)

    evals, evecs = np.linalg.eigh(0.5 * (cxx + cxx.T))
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    used_rank = int(np.sum(evals > rank_tol * evals[0]))
    basis = evecs[:, :used_rank]
    metric = evals[:used_rank]
    whitener = (metric ** -0.5)[:, None] * basis.T
    whitened = whitener @ cxxz @ whitener.T
    whitened = 0.5 * (whitened + whitened.T)
    lambdas, white_filters = np.linalg.eigh(whitened)
    order = np.argsort(lambdas)[::-1]
    lambdas = lambdas[order]
    filters = whitener.T @ white_filters[:, order]
    for index in range(filters.shape[1]):
        filters[:, index] /= np.sqrt(filters[:, index] @ cxx @ filters[:, index])
    patterns = cxx @ np.linalg.solve(filters.T @ cxx @ filters, filters.T).T
    return {
        "filters": np.asarray(filters.T, dtype=np.float64),
        "patterns": np.asarray(patterns.T, dtype=np.float64),
        "eigenvalues": np.asarray(lambdas, dtype=np.float64),
        "z": z,
        "cxx": np.asarray(cxx, dtype=np.float64),
        "cxxz": np.asarray(cxxz, dtype=np.float64),
        "rank": np.asarray([used_rank], dtype=np.int64),
    }


def align_rows(reference: NDArray[np.floating], estimated: NDArray[np.floating]) -> NDArray[np.float64]:
    aligned = np.array(estimated, dtype=np.float64, copy=True)
    for index in range(min(reference.shape[0], aligned.shape[0])):
        if float(np.dot(reference[index], aligned[index])) < 0.0:
            aligned[index] *= -1.0
    return aligned
