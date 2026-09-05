"""Input and parameter validation for ReDisCA. No scientific orchestration."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

Aggregation = Literal["mean", "sum"]
SolverName = Literal["generalized", "whitening"]


def _reject_bool(value: object, *, name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{name} must not be a bool; got {value!r}."
        )


def _as_bool(value: object, *, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{name} must be a bool, got {type(value).__name__}."
        )
    return bool(value)


def _as_finite_array(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} contains NaN values.")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} contains Inf values.")
    return array


def validate_positive_int(value: object, *, name: str) -> int:
    """Return a strictly positive integer, rejecting bool."""
    _reject_bool(value, name=name)
    if not isinstance(value, (int, np.integer)):
        raise TypeError(
            f"{name} must be a positive integer, got {type(value).__name__}."
        )
    value = int(value)
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return value


def validate_estimator_params(
    *,
    n_components: int | None,
    demean_time: object,
    divide_by_t_minus_1: object,
    directed_pairs: object,
    aggregation: object,
    solver: object,
    rank: int | None,
    rank_tol: float,
) -> tuple[int | None, bool, bool, bool, Aggregation, SolverName, int | None, float]:
    """Validate constructor parameters. Does not depend on data shapes."""
    demean_time = _as_bool(demean_time, name="demean_time")
    divide_by_t_minus_1 = _as_bool(divide_by_t_minus_1, name="divide_by_t_minus_1")
    directed_pairs = _as_bool(directed_pairs, name="directed_pairs")

    if not isinstance(aggregation, str):
        raise TypeError(
            "aggregation must be 'mean' or 'sum', "
            f"got {type(aggregation).__name__}."
        )
    if aggregation not in ("mean", "sum"):
        raise ValueError(
            "aggregation must be 'mean' or 'sum', "
            f"got {aggregation!r}."
        )

    if not isinstance(solver, str):
        raise TypeError(
            "solver must be 'generalized' or 'whitening', "
            f"got {type(solver).__name__}."
        )
    if solver not in ("generalized", "whitening"):
        raise ValueError(
            "solver must be 'generalized' or 'whitening', "
            f"got {solver!r}."
        )

    if n_components is not None:
        n_components = validate_positive_int(n_components, name="n_components")

    if rank is not None:
        rank = validate_positive_int(rank, name="rank")

    _reject_bool(rank_tol, name="rank_tol")
    try:
        rank_tol = float(rank_tol)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "rank_tol must be a finite real scalar satisfying "
            f"0 < rank_tol < 1, got {type(rank_tol).__name__}."
        ) from exc
    if not np.isfinite(rank_tol) or not (0.0 < rank_tol < 1.0):
        raise ValueError(
            "rank_tol must be a finite real scalar satisfying "
            f"0 < rank_tol < 1, got {rank_tol}."
        )
    return (
        n_components,
        demean_time,
        divide_by_t_minus_1,
        directed_pairs,
        aggregation,
        solver,
        rank,
        rank_tol,
    )


def validate_fit_xy(
    X: ArrayLike,
    y: ArrayLike | None,
    *,
    demean_time: bool,
    divide_by_t_minus_1: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate condition-average data ``X`` and target RDM ``y``."""
    if y is None:
        raise TypeError(
            "y (target representational dissimilarity matrix) is required."
        )

    X_arr = _as_finite_array(X, name="X")
    y_arr = _as_finite_array(y, name="y")

    if X_arr.ndim != 3:
        raise ValueError(
            "X must have shape (n_conditions, n_channels, n_times), "
            f"got {X_arr.shape}."
        )
    n_conditions, n_channels, n_times = X_arr.shape
    if n_conditions < 3:
        raise ValueError(
            "Need at least 3 conditions for a meaningful standardized "
            f"ReDisCA target. Got n_conditions={n_conditions}."
        )
    if n_channels < 1:
        raise ValueError(
            f"Number of channels must be >= 1. Got {n_channels}."
        )
    if n_times < 1:
        raise ValueError(
            f"Number of time samples must be >= 1. Got {n_times}."
        )
    if demean_time and n_times < 2:
        raise ValueError(
            "demean_time=True requires at least 2 time samples; "
            "temporal centering of a single sample zeros every pair matrix."
        )
    if divide_by_t_minus_1 and n_times < 2:
        raise ValueError(
            "divide_by_t_minus_1=True requires at least 2 time samples."
        )

    if y_arr.ndim != 2:
        raise ValueError(
            "y (target representational dissimilarity matrix) must be a "
            f"2D array of shape (n_conditions, n_conditions). Got {y_arr.shape}."
        )
    if y_arr.shape[0] != y_arr.shape[1]:
        raise ValueError(
            f"y must be square. Got shape {y_arr.shape}."
        )
    if y_arr.shape[0] != n_conditions:
        raise ValueError(
            f"Size of y ({y_arr.shape[0]}) does not match "
            f"n_conditions ({n_conditions})."
        )
    if not np.allclose(y_arr, y_arr.T, rtol=1e-10, atol=1e-12):
        raise ValueError("y must be a symmetric matrix.")
    if not np.allclose(np.diag(y_arr), 0.0, rtol=1e-10, atol=1e-12):
        raise ValueError(
            "y diagonal must be zero (self-dissimilarity = 0)."
        )
    return X_arr, y_arr


def validate_transform_X(
    X: ArrayLike,
    n_features_in: int,
) -> NDArray[np.float64]:
    """Validate arrays passed to ``transform``."""
    X_arr = _as_finite_array(X, name="X")
    if X_arr.ndim != 3:
        raise ValueError(
            "X must have shape (n_observations, n_channels, n_times), "
            f"got {X_arr.shape}."
        )
    if X_arr.shape[1] != n_features_in:
        raise ValueError(
            f"X has {X_arr.shape[1]} channels, but ReDisCA was fitted "
            f"with n_features_in_={n_features_in}."
        )
    if X_arr.shape[2] < 1:
        raise ValueError(
            f"Number of time samples must be >= 1. Got {X_arr.shape[2]}."
        )
    return X_arr


def validate_inverse_U(
    U: ArrayLike,
    n_components: int,
) -> NDArray[np.float64]:
    """Validate arrays passed to ``inverse_transform``."""
    U_arr = _as_finite_array(U, name="U")
    if U_arr.ndim != 3:
        raise ValueError(
            "U must have shape (n_observations, n_components_, n_times), "
            f"got {U_arr.shape}."
        )
    if U_arr.shape[1] != n_components:
        raise ValueError(
            f"U has {U_arr.shape[1]} components, but this estimator uses "
            f"n_components_={n_components}."
        )
    if U_arr.shape[2] < 1:
        raise ValueError(
            f"Number of time samples must be >= 1. Got {U_arr.shape[2]}."
        )
    return U_arr
