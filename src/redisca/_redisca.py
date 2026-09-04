"""Sklearn-style public estimator for Representational Dissimilarity Component Analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ._core import (
    compute_patterns,
    mean_pair_matrix,
    pair_indices,
    pair_matrices,
    solve_generalized_eigenproblem,
    standardize_target,
    vectorize_rdm,
    weighted_centered_mean,
)
from ._validation import (
    validate_estimator_params,
    validate_fit_xy,
    validate_inverse_U,
    validate_transform_X,
)


class ReDisCA(TransformerMixin, BaseEstimator):
    """Representational Dissimilarity Component Analysis.

    Finds spatial filters whose condition-to-condition dissimilarity structure
    matches a target representational dissimilarity matrix (RDM).

    Parameters
    ----------
    n_components : int or None, default=None
        Number of leading ReDisCA components used by ``transform`` and
        ``inverse_transform``. ``None`` uses all ``rank_`` components. This
        parameter does not change whitening or the fitted decomposition.
        If greater than ``rank_`` after fit, ``fit`` raises ``ValueError``.
    demean_time : bool, default=True
        If True, temporally center each condition-pair difference per channel
        before forming the quadratic pair matrix (MATLAB ``cov`` centering).
        If False, use the uncentered Gram corresponding to the paper's printed
        squared-Euclidean expression. This flag does not apply a ``1/(T-1)``
        scale and is not used at transform time.
    rank : int or None, default=None
        Principal-space rank of ``R_bar`` used to solve the generalized
        eigenproblem. ``None`` uses the effective numerical rank defined by
        ``rank_tol``. This is not the number of ReDisCA output components.
        If the requested rank exceeds the effective numerical rank, ``fit``
        raises ``ValueError``.
    rank_tol : float, default=1e-6
        Relative eigenvalue threshold. Directions of ``R_bar`` with
        ``eigval > rank_tol * max_eigval`` define the effective numerical rank.

    Attributes
    ----------
    filters_ : ndarray of shape (rank_, n_channels)
        Spatial filters. Rows are components.
    patterns_ : ndarray of shape (rank_, n_channels)
        Spatial patterns (Haufe/SPoC topographies). Rows are components.
    eigenvalues_ : ndarray of shape (rank_,)
        Generalized eigenvalues, sorted in signed descending order.
    rank_ : int
        Principal-space rank actually used to solve the eigenproblem.
    n_components_ : int
        Number of leading components used by ``transform`` and
        ``inverse_transform``.
    n_features_in_ : int
        Number of channels seen during ``fit``.
    n_conditions_ : int
        Number of conditions seen during ``fit``.
    n_times_in_ : int
        Number of time samples seen during ``fit``.
    """

    def __init__(
        self,
        n_components: int | None = None,
        *,
        demean_time: bool = True,
        rank: int | None = None,
        rank_tol: float = 1e-6,
    ) -> None:
        self.n_components = n_components
        self.demean_time = demean_time
        self.rank = rank
        self.rank_tol = rank_tol

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ReDisCA":
        """Fit ReDisCA to condition-average data and a target RDM.

        Parameters
        ----------
        X : array-like of shape (n_conditions, n_channels, n_times)
            Condition-average evoked data.
        y : array-like of shape (n_conditions, n_conditions)
            Target representational dissimilarity matrix. Must be square,
            symmetric, finite, and have a zero diagonal.

        Returns
        -------
        self : ReDisCA
            Fitted estimator.
        """
        n_components, demean_time, rank, rank_tol = validate_estimator_params(
            n_components=self.n_components,
            demean_time=self.demean_time,
            rank=self.rank,
            rank_tol=self.rank_tol,
        )
        X_arr, y_arr = validate_fit_xy(X, y, demean_time=demean_time)
        n_conditions, n_channels, n_times = X_arr.shape

        pairs = pair_indices(n_conditions)
        pair_stack = pair_matrices(X_arr, pairs, demean_time=demean_time)
        target_vec = vectorize_rdm(y_arr, pairs)
        z = standardize_target(target_vec)
        r_bar = mean_pair_matrix(pair_stack)
        r_bar_d = weighted_centered_mean(pair_stack, r_bar, z)

        filters, eigenvalues = solve_generalized_eigenproblem(
            r_bar_d,
            r_bar,
            rank=rank,
            rank_tol=rank_tol,
        )
        patterns = compute_patterns(filters, r_bar)

        used_rank = int(filters.shape[1])
        if n_components is None:
            n_components_ = used_rank
        elif n_components > used_rank:
            raise ValueError(
                f"n_components={n_components} exceeds fitted rank_={used_rank}."
            )
        else:
            n_components_ = n_components

        self.filters_ = np.asarray(filters.T, dtype=np.float64)
        self.patterns_ = np.asarray(patterns.T, dtype=np.float64)
        self.eigenvalues_ = np.asarray(eigenvalues, dtype=np.float64)
        self.rank_ = used_rank
        self.n_components_ = n_components_
        self.n_features_in_ = int(n_channels)
        self.n_conditions_ = int(n_conditions)
        self.n_times_in_ = int(n_times)
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Apply fitted spatial filters.

        Parameters
        ----------
        X : array-like of shape (n_observations, n_channels, n_times)
            Data to filter. The number of observations and time samples may
            differ from ``fit``; only the channel count must match
            ``n_features_in_``.

        Returns
        -------
        U : ndarray of shape (n_observations, n_components_, n_times)
            Spatially filtered time series.
        """
        check_is_fitted(self, ["filters_", "rank_"])
        X_arr = validate_transform_X(X, self.n_features_in_)
        filters = self.filters_[: self.n_components_]
        return np.einsum("rc,oct->ort", filters, X_arr)

    def inverse_transform(self, U: ArrayLike) -> NDArray[np.float64]:
        """Reconstruct sensor-space data from component time series.

        Uses ``patterns_[:n_components_].T``. When ``n_components < rank_``
        this is a component-limited reconstruction.

        Parameters
        ----------
        U : array-like of shape (n_observations, n_components_, n_times)
            Component time series, typically from ``transform``.

        Returns
        -------
        X : ndarray of shape (n_observations, n_channels, n_times)
            Reconstructed sensor-space time series.
        """
        check_is_fitted(self, ["filters_", "rank_"])
        U_arr = validate_inverse_U(U, self.n_components_)
        patterns = self.patterns_[: self.n_components_]
        return np.einsum("rc,ort->oct", patterns, U_arr)
