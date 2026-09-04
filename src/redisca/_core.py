"""Deterministic numerical primitives for ReDisCA.

This module implements the scientific core only: pair construction, pair
matrices, target standardization, mean aggregations, rank/whitening, the
generalized eigenproblem, filter metric-normalization, and Haufe/SPoC
patterns. Inputs are assumed already validated.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh


def pair_indices(n_conditions: int) -> list[tuple[int, int]]:
    """Return unique unordered condition pairs ``(i, j)`` with ``i < j``."""
    return [
        (i, j)
        for i in range(n_conditions)
        for j in range(i + 1, n_conditions)
    ]


def pair_matrix(
    x_i: NDArray[np.floating],
    x_j: NDArray[np.floating],
    *,
    demean_time: bool,
) -> NDArray[np.floating]:
    """Form the symmetric quadratic pair matrix from two condition averages.

    ``demean_time`` controls only per-channel temporal centering. The quadratic
    itself is the unscaled Gram ``delta @ delta.T`` on both paths: MATLAB
    ``cov``'s ``1/(T-1)`` is a global scalar and is omitted by design.
    """
    delta = x_i - x_j
    if demean_time:
        delta = delta - delta.mean(axis=-1, keepdims=True)
    return delta @ delta.T


def pair_matrices(
    X: NDArray[np.floating],
    pairs: list[tuple[int, int]],
    *,
    demean_time: bool,
) -> NDArray[np.floating]:
    """Stack pair matrices for ``pairs`` into an array of shape ``(P, N, N)``."""
    index_i = np.fromiter((i for i, _ in pairs), dtype=np.intp, count=len(pairs))
    index_j = np.fromiter((j for _, j in pairs), dtype=np.intp, count=len(pairs))
    delta = X[index_i] - X[index_j]
    if demean_time:
        delta = delta - delta.mean(axis=-1, keepdims=True)
    return np.matmul(delta, np.swapaxes(delta, -1, -2))


def vectorize_rdm(
    y: NDArray[np.floating],
    pairs: list[tuple[int, int]],
) -> NDArray[np.floating]:
    """Extract target RDM entries in the same ``i < j`` order as the pair matrices."""
    return np.fromiter(
        (y[i, j] for i, j in pairs),
        dtype=np.float64,
        count=len(pairs),
    )


def standardize_target(values: NDArray[np.floating]) -> NDArray[np.floating]:
    """Z-score a pair vector with the MATLAB/SPoC sample standard deviation.

    Uses ``ddof=1``. The implementation is scale-free: for any finite
    non-constant vector ``v`` and positive finite scalar ``c``,
    ``standardize_target(c * v)`` matches ``standardize_target(v)`` up to
    floating-point error. A genuinely constant vector raises.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        raise ValueError("Target RDM pair vector must not be empty.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Target RDM pair vector contains non-finite values.")

    centered = values - np.mean(values)
    if not np.all(np.isfinite(centered)):
        raise ValueError("Target RDM pair vector is not finite after centering.")

    amplitude = float(np.max(np.abs(centered)))
    if not np.isfinite(amplitude):
        raise ValueError("Target RDM pair vector amplitude is not finite.")
    if amplitude == 0.0:
        raise ValueError(
            "Standard deviation of the target RDM pair vector is close to zero. "
            "The target RDM is uninformative (all unique pair entries are "
            "nearly equal)."
        )

    scaled = centered / amplitude
    scale = float(np.std(scaled, ddof=1))
    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError(
            "Standard deviation of the target RDM pair vector is close to zero. "
            "The target RDM is uninformative (all unique pair entries are "
            "nearly equal)."
        )
    return scaled / scale


def mean_pair_matrix(pair_stack: NDArray[np.floating]) -> NDArray[np.floating]:
    """Return ``R_bar = mean_k(R_k)``."""
    return np.mean(pair_stack, axis=0)


def weighted_centered_mean(
    pair_stack: NDArray[np.floating],
    r_bar: NDArray[np.floating],
    z: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Return ``R_bar_d = mean_k(z_k * (R_k - R_bar))``."""
    z = np.asarray(z, dtype=np.float64)
    centered = pair_stack - r_bar
    return np.mean(z[:, np.newaxis, np.newaxis] * centered, axis=0)


def symmetrize_matrix(
    matrix: NDArray[np.floating],
    *,
    name: str,
    relative_tol: float = 1e-8,
    eps_factor: float = 100.0,
) -> NDArray[np.floating]:
    """Return the symmetric part of ``matrix``, or raise if asymmetry is material.

    The test is relative Frobenius: the antisymmetric part is compared to the
    symmetric part, with a scale-aware floor so an exact zero is not treated as
    a failure.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    symmetric = 0.5 * (matrix + matrix.T)
    antisymmetric = 0.5 * (matrix - matrix.T)
    anti_norm = float(np.linalg.norm(antisymmetric, ord="fro"))
    matrix_norm = float(np.linalg.norm(matrix, ord="fro"))
    symmetric_norm = float(np.linalg.norm(symmetric, ord="fro"))
    floor = max(
        eps_factor * float(np.finfo(np.float64).eps) * max(matrix_norm, 0.0),
        relative_tol * max(symmetric_norm, 0.0),
    )
    if anti_norm > floor:
        rel = anti_norm / max(symmetric_norm, np.finfo(np.float64).tiny)
        raise ValueError(
            f"{name} is not symmetric within tolerance: "
            f"||anti||_F={anti_norm:.3e}, rel={rel:.3e}. "
            "This indicates an upstream construction bug."
        )
    return symmetric


def normalize_filters(
    filters: NDArray[np.floating],
    r_bar: NDArray[np.floating],
    *,
    min_norm_sq: float = 1e-12,
) -> NDArray[np.floating]:
    """Normalize columns of ``filters`` so that ``w.T @ R_bar @ w = 1``."""
    filters = np.array(filters, dtype=np.float64, copy=True)
    for index in range(filters.shape[1]):
        weight = filters[:, index]
        norm_sq = float(weight @ r_bar @ weight)
        if not np.isfinite(norm_sq):
            raise RuntimeError(
                f"Failed to normalize filter {index}: "
                f"w.T @ R_bar @ w is not finite ({norm_sq})."
            )
        if norm_sq <= min_norm_sq:
            raise RuntimeError(
                f"Failed to normalize filter {index}: "
                f"w.T @ R_bar @ w = {norm_sq:.3e} <= {min_norm_sq:.3e}."
            )
        filters[:, index] = weight / np.sqrt(norm_sq)
    return filters


def solve_generalized_eigenproblem(
    r_bar_d: NDArray[np.floating],
    r_bar: NDArray[np.floating],
    *,
    rank: int | None = None,
    rank_tol: float = 1e-6,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Solve ``R_bar_d w = lambda R_bar w`` in the principal subspace of ``R_bar``.

    Eigenvalues are returned in signed descending order. Internal filters have
    shape ``(n_channels, rank_used)`` and satisfy ``w.T @ R_bar @ w = 1``.
    """
    r_bar = symmetrize_matrix(r_bar, name="R_bar")
    r_bar_d = symmetrize_matrix(r_bar_d, name="R_bar_d")

    eigenvalues, eigenvectors = eigh(r_bar)
    max_eig = float(np.max(eigenvalues))
    if not np.isfinite(max_eig) or max_eig <= 0.0:
        raise ValueError(
            "No positive eigenvalues found in R_bar. "
            "The condition-average data are uninformative."
        )

    eig_tol = float(rank_tol) * max_eig
    effective_rank = int(np.sum(eigenvalues > eig_tol))
    if effective_rank == 0:
        raise ValueError(
            "No eigenvalues of R_bar exceed rank_tol * max_eigval. "
            "The condition-average data are uninformative."
        )

    if rank is None:
        used_rank = effective_rank
    elif rank > effective_rank:
        raise ValueError(
            f"Requested rank={rank} exceeds the effective numerical rank "
            f"{effective_rank} of R_bar "
            f"(threshold rank_tol * max_eigval = {eig_tol:.3e})."
        )
    else:
        used_rank = rank

    # Leading principal directions: largest algebraic eigenvalues of R_bar.
    principal = np.argsort(eigenvalues)[::-1][:used_rank]
    basis = eigenvectors[:, principal]
    reduced_metric = eigenvalues[principal]
    if np.any(reduced_metric <= eig_tol):
        raise RuntimeError(
            "Internal error: selected R_bar eigenvalues are not above the "
            "rank threshold despite effective-rank filtering."
        )

    reduced_target = symmetrize_matrix(
        basis.T @ r_bar_d @ basis,
        name="R_bar_d_principal",
    )
    reduced_evals, reduced_filters = eigh(
        reduced_target,
        np.diag(reduced_metric),
    )
    order = np.argsort(reduced_evals)[::-1]
    filters = basis @ reduced_filters[:, order]
    filters = normalize_filters(filters, r_bar)

    lambdas = np.einsum("ij,ji->i", filters.T, r_bar_d @ filters)
    order = np.argsort(lambdas)[::-1]
    return filters[:, order], lambdas[order]


def compute_patterns(
    filters: NDArray[np.floating],
    r_bar: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Compute Haufe/SPoC patterns ``A = R_bar @ W @ inv(W.T @ R_bar @ W)``.

    ``filters`` is column-oriented, shape ``(n_channels, rank)``. The formula
    is valid for rectangular ``W``.
    """
    metric = filters.T @ r_bar @ filters
    projected = r_bar @ filters
    try:
        patterns = np.linalg.solve(metric, projected.T).T
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "Failed to compute patterns: W.T @ R_bar @ W is singular."
        ) from exc
    return patterns
