"""Input and parameter validation for ReDisCA. No scientific orchestration."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _reject_bool(value: object, *, name: str) -> None:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(
            f"{name} must not be a bool; got {value!r}."
        )


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
    rank: int | None,
    rank_tol: float,
) -> tuple[int | None, bool, int | None, float]:
    """Validate constructor parameters. Does not depend on data shapes."""
    if not isinstance(demean_time, (bool, np.bool_)):
        raise TypeError(
            "demean_time must be a bool, "
            f"got {type(demean_time).__name__}."
        )
    demean_time = bool(demean_time)

    if n_components is not None:
        n_components = validate_positive_int(n_components, name="n_components")

    if rank is not None:
        rank = validate_positive_int(rank, name="rank")

    rank_tol = float(rank_tol)
    if not np.isfinite(rank_tol) or rank_tol <= 0.0:
        raise ValueError(
            "rank_tol must be a finite number > 0, "
            f"got {rank_tol}."
        )
    return n_components, demean_time, rank, rank_tol


def validate_fit_xy(
    X: ArrayLike,
    y: ArrayLike | None,
    *,
    demean_time: bool,
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
