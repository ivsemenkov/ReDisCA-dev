"""Deterministic numerical primitives for ReDisCA.

This module implements the scientific core only: pair construction, pair
matrices, target standardization, pair aggregation, rank/whitening, the
generalized and explicit-whitening eigenproblems, filter metric-normalization,
and Haufe/SPoC patterns. Inputs are assumed already validated.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh

Aggregation = Literal["mean", "sum"]
SolverName = Literal["generalized", "whitening"]


class MetricSubspace(NamedTuple):
    """Principal subspace of ``R_bar`` / ``Cxx`` used by both solvers."""

    r_bar: NDArray[np.floating]
    eigenvalues: NDArray[np.floating]
    basis: NDArray[np.floating]
    eig_tol: float
    used_rank: int
    effective_rank: int


def pair_indices(
    n_conditions: int,
    *,
    directed: bool = False,
) -> list[tuple[int, int]]:
    """Return condition pairs in a deterministic nested-loop order.

    ``directed=False`` (default) yields unique unordered pairs ``(i, j)``
    with ``i < j``. ``directed=True`` yields every ``i != j`` in the AIRI
    nested-loop order: outer ``i``, inner ``j``, skip ``i == j``.
    """
    if directed:
        return [
            (i, j)
            for i in range(n_conditions)
            for j in range(n_conditions)
            if i != j
        ]
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
    divide_by_t_minus_1: bool = False,
) -> NDArray[np.floating]:
    """Form the symmetric quadratic pair matrix from two condition averages.

    ``demean_time`` controls only per-channel temporal centering.
    ``divide_by_t_minus_1`` applies MATLAB ``cov``'s ``1/(T-1)`` scale.
    The two switches are independent.
    """
    delta = x_i - x_j
    if demean_time:
        delta = delta - delta.mean(axis=-1, keepdims=True)
    gram = delta @ delta.T
    if divide_by_t_minus_1:
        n_times = int(delta.shape[-1])
        if n_times < 2:
            raise ValueError(
                "divide_by_t_minus_1=True requires at least 2 time samples."
            )
        gram = gram / (n_times - 1)
    return gram


def pair_matrices(
    X: NDArray[np.floating],
    pairs: list[tuple[int, int]],
    *,
    demean_time: bool,
    divide_by_t_minus_1: bool = False,
) -> NDArray[np.floating]:
    """Stack pair matrices for ``pairs`` into an array of shape ``(P, N, N)``."""
    index_i = np.fromiter((i for i, _ in pairs), dtype=np.intp, count=len(pairs))
    index_j = np.fromiter((j for _, j in pairs), dtype=np.intp, count=len(pairs))
    delta = X[index_i] - X[index_j]
    if demean_time:
        delta = delta - delta.mean(axis=-1, keepdims=True)
    grams = np.matmul(delta, np.swapaxes(delta, -1, -2))
    if divide_by_t_minus_1:
        n_times = int(X.shape[-1])
        if n_times < 2:
            raise ValueError(
                "divide_by_t_minus_1=True requires at least 2 time samples."
            )
        grams = grams / (n_times - 1)
    return grams


def vectorize_rdm(
    y: NDArray[np.floating],
    pairs: list[tuple[int, int]],
) -> NDArray[np.floating]:
    """Extract target RDM entries in the same pair order as the pair matrices."""
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


def weighted_aggregate(
    centered_stack: NDArray[np.floating],
    z: NDArray[np.floating],
    *,
    aggregation: Aggregation = "mean",
) -> NDArray[np.floating]:
    """Return the z-weighted mean or sum of already-centered pair matrices."""
    z = np.asarray(z, dtype=np.float64)
    weighted = z[:, np.newaxis, np.newaxis] * centered_stack
    if aggregation == "mean":
        return np.mean(weighted, axis=0)
    if aggregation == "sum":
        return np.sum(weighted, axis=0)
    raise ValueError(
        "aggregation must be 'mean' or 'sum', "
        f"got {aggregation!r}."
    )


def weighted_centered_mean(
    pair_stack: NDArray[np.floating],
    r_bar: NDArray[np.floating],
    z: NDArray[np.floating],
    *,
    aggregation: Aggregation = "mean",
) -> NDArray[np.floating]:
    """Return the z-weighted centered pair aggregate.

    Default ``aggregation='mean'`` is ``mean_k(z_k * (R_k - R_bar))``.
    ``aggregation='sum'`` is the paper Eq. 7 sum of the same terms.
    """
    centered = pair_stack - r_bar
    return weighted_aggregate(centered, z, aggregation=aggregation)


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


def metric_subspace(
    r_bar: NDArray[np.floating],
    *,
    rank: int | None = None,
    rank_tol: float = 1e-6,
) -> MetricSubspace:
    """Eigendecompose ``R_bar``, sort descending, and apply the rank threshold.

    Directions with ``eigval > rank_tol * max_eigval`` define the effective
    numerical rank. This is the shared SPoC/library rank rule.
    """
    r_bar = symmetrize_matrix(r_bar, name="R_bar")
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

    principal = np.argsort(eigenvalues)[::-1][:used_rank]
    basis = eigenvectors[:, principal]
    reduced_metric = eigenvalues[principal]
    if np.any(reduced_metric <= eig_tol):
        raise RuntimeError(
            "Internal error: selected R_bar eigenvalues are not above the "
            "rank threshold despite effective-rank filtering."
        )
    return MetricSubspace(
        r_bar=r_bar,
        eigenvalues=np.asarray(reduced_metric, dtype=np.float64),
        basis=np.asarray(basis, dtype=np.float64),
        eig_tol=eig_tol,
        used_rank=used_rank,
        effective_rank=effective_rank,
    )


def whitening_matrix(subspace: MetricSubspace) -> NDArray[np.float64]:
    """Return the stock-SPoC whitening matrix with filters in the rows."""
    return (subspace.eigenvalues ** -0.5)[:, np.newaxis] * subspace.basis.T


def _normalized_spectrum(
    filters: NDArray[np.floating],
    r_bar_d: NDArray[np.floating],
    r_bar: NDArray[np.floating],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    filters = normalize_filters(filters, r_bar)
    lambdas = np.einsum("ij,ji->i", filters.T, r_bar_d @ filters)
    order = np.argsort(lambdas)[::-1]
    return filters[:, order], lambdas[order]


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
    r_bar_d = symmetrize_matrix(r_bar_d, name="R_bar_d")
    subspace = metric_subspace(r_bar, rank=rank, rank_tol=rank_tol)
    reduced_target = symmetrize_matrix(
        subspace.basis.T @ r_bar_d @ subspace.basis,
        name="R_bar_d_principal",
    )
    _reduced_evals, reduced_filters = eigh(
        reduced_target,
        np.diag(subspace.eigenvalues),
    )
    order = np.argsort(_reduced_evals)[::-1]
    filters = subspace.basis @ reduced_filters[:, order]
    return _normalized_spectrum(filters, r_bar_d, subspace.r_bar)


def solve_whitening_eigenproblem(
    r_bar_d: NDArray[np.floating],
    r_bar: NDArray[np.floating],
    *,
    rank: int | None = None,
    rank_tol: float = 1e-6,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Solve the stock-SPoC explicit-whitening form of the same eigenproblem.

    Steps: eigendecompose ``Cxx``, sort descending, keep
    ``eig > max_eig * rank_tol``, form the whitening matrix, ordinary eig of
    the whitened ``Cxxz``, map filters back to sensor space, metric-normalize,
    and use Haufe/SPoC patterns downstream.
    """
    r_bar_d = symmetrize_matrix(r_bar_d, name="R_bar_d")
    subspace = metric_subspace(r_bar, rank=rank, rank_tol=rank_tol)
    whitener = whitening_matrix(subspace)
    whitened = symmetrize_matrix(
        whitener @ r_bar_d @ whitener.T,
        name="Cxxz_white",
    )
    _white_evals, white_filters = eigh(whitened)
    order = np.argsort(_white_evals)[::-1]
    filters = whitener.T @ white_filters[:, order]
    return _normalized_spectrum(filters, r_bar_d, subspace.r_bar)


def solve_eigenproblem(
    r_bar_d: NDArray[np.floating],
    r_bar: NDArray[np.floating],
    *,
    solver: SolverName = "generalized",
    rank: int | None = None,
    rank_tol: float = 1e-6,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Dispatch to the generalized or explicit-whitening solver."""
    if solver == "generalized":
        return solve_generalized_eigenproblem(
            r_bar_d, r_bar, rank=rank, rank_tol=rank_tol
        )
    if solver == "whitening":
        return solve_whitening_eigenproblem(
            r_bar_d, r_bar, rank=rank, rank_tol=rank_tol
        )
    raise ValueError(
        "solver must be 'generalized' or 'whitening', "
        f"got {solver!r}."
    )


def subspace_eigenvalues(
    r_bar_d: NDArray[np.floating],
    subspace: MetricSubspace,
    *,
    solver: SolverName,
) -> NDArray[np.float64]:
    """Eigenvalues of a weighted matrix in a frozen ``Cxx`` subspace.

    Used by the random-phase test so surrogates reuse the fitted rank and
    whitening without reconstructing pair matrices from the data.
    """
    r_bar_d = symmetrize_matrix(r_bar_d, name="R_bar_d")
    if solver == "whitening":
        whitener = whitening_matrix(subspace)
        whitened = symmetrize_matrix(
            whitener @ r_bar_d @ whitener.T,
            name="Cxxz_white",
        )
        return np.asarray(eigh(whitened, eigvals_only=True), dtype=np.float64)
    if solver == "generalized":
        reduced = symmetrize_matrix(
            subspace.basis.T @ r_bar_d @ subspace.basis,
            name="R_bar_d_principal",
        )
        return np.asarray(
            eigh(reduced, np.diag(subspace.eigenvalues), eigvals_only=True),
            dtype=np.float64,
        )
    raise ValueError(
        "solver must be 'generalized' or 'whitening', "
        f"got {solver!r}."
    )


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
