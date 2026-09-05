"""Independent SPoC-style MATLAB-cov equivalence tests for the deterministic ReDisCA core.

These tests reimplement MATLAB-cov pair construction and the SPoC whitening
GEP without calling ``redisca._core.pair_matrix`` / ``pair_matrices``.

They validate SPoC-style MATLAB-cov covariance/whitening semantics under the
approved canonical unique unordered ``i < j`` pair representation. They are
not literal AIRI end-to-end numerical parity: the AIRI MATLAB script uses
directed duplicated pairs ``i != j``.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from redisca import ReDisCA


def _unique_pairs(n_conditions: int) -> list[tuple[int, int]]:
    return [
        (i, j)
        for i in range(n_conditions)
        for j in range(i + 1, n_conditions)
    ]


def _matlab_cov(delta: np.ndarray) -> np.ndarray:
    """MATLAB ``cov`` of a ``(T, n_channels)`` epoch stored as ``(n_channels, T)``."""
    n_times = delta.shape[-1]
    centered = delta - delta.mean(axis=-1, keepdims=True)
    return (centered @ centered.T) / (n_times - 1)


def _independent_spoc(X: np.ndarray, y: np.ndarray, *, rank_tol: float = 1e-6):
    n_conditions, n_channels, n_times = X.shape
    pairs = _unique_pairs(n_conditions)
    covariances = np.stack(
        [_matlab_cov(X[i] - X[j]) for i, j in pairs],
        axis=0,
    )
    cxx = covariances.mean(axis=0)
    values = np.array([y[i, j] for i, j in pairs], dtype=np.float64)
    z = (values - values.mean()) / values.std(ddof=1)
    cxxz = np.mean(z[:, None, None] * (covariances - cxx), axis=0)

    evals, evecs = np.linalg.eigh(0.5 * (cxx + cxx.T))
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    used_rank = int(np.sum(evals > rank_tol * evals[0]))
    basis = evecs[:, :used_rank]
    metric = evals[:used_rank]
    whitener = np.diag(metric ** -0.5) @ basis.T
    whitened = whitener @ cxxz @ whitener.T
    whitened = 0.5 * (whitened + whitened.T)
    lambdas, white_filters = np.linalg.eigh(whitened)
    order = np.argsort(lambdas)[::-1]
    lambdas = lambdas[order]
    filters = whitener.T @ white_filters[:, order]
    for index in range(filters.shape[1]):
        filters[:, index] /= np.sqrt(filters[:, index] @ cxx @ filters[:, index])
    patterns = cxx @ np.linalg.solve(filters.T @ cxx @ filters, filters.T).T
    return filters.T, patterns.T, lambdas


def _align_rows(reference: np.ndarray, estimated: np.ndarray) -> np.ndarray:
    aligned = estimated.copy()
    for index in range(estimated.shape[0]):
        if np.dot(reference[index], estimated[index]) < 0:
            aligned[index] *= -1
    return aligned


def _structured_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_conditions, n_channels, n_times = 5, 7, 36
    mixing = rng.standard_normal((n_channels, 3))
    sources = rng.standard_normal((3, n_times))
    # Nonzero temporal means so demeaning is not a no-op.
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


class TestIndependentSPoCEquivalence:
    """SPoC-style MATLAB-cov/whitening check on unique ``i < j`` pairs, not AIRI."""

    def test_demean_time_true_matches_independent_spoc(self):
        X, y = _structured_problem()
        n_times = X.shape[-1]
        ref_filters, ref_patterns, ref_lambdas = _independent_spoc(X, y)

        model = ReDisCA(demean_time=True).fit(X, y)
        assert model.rank_ == ref_filters.shape[0]
        assert_allclose(model.eigenvalues_, ref_lambdas, rtol=1e-8, atol=1e-10)

        # Metric-normalized filters differ from MATLAB-cov SPoC by 1/sqrt(T-1)
        # because ReDisCA uses unscaled Grams. Reconstruction is invariant.
        scale = np.sqrt(n_times - 1)
        aligned_filters = _align_rows(ref_filters, model.filters_ * scale)
        aligned_patterns = _align_rows(ref_patterns, model.patterns_ / scale)
        assert_allclose(aligned_filters, ref_filters, rtol=1e-6, atol=1e-8)
        assert_allclose(aligned_patterns, ref_patterns, rtol=1e-6, atol=1e-8)

        reconstruction_ours = model.patterns_.T @ model.filters_
        reconstruction_ref = ref_patterns.T @ ref_filters
        assert_allclose(reconstruction_ours, reconstruction_ref, rtol=1e-6, atol=1e-8)

    def test_scaled_whitening_solver_matches_independent_spoc(self):
        X, y = _structured_problem()
        ref_filters, ref_patterns, ref_lambdas = _independent_spoc(X, y)
        model = ReDisCA(
            demean_time=True,
            divide_by_t_minus_1=True,
            solver="whitening",
        ).fit(X, y)
        assert model.rank_ == ref_filters.shape[0]
        assert_allclose(model.eigenvalues_, ref_lambdas, rtol=1e-8, atol=1e-10)
        aligned_filters = _align_rows(ref_filters, model.filters_)
        aligned_patterns = _align_rows(ref_patterns, model.patterns_)
        assert_allclose(aligned_filters, ref_filters, rtol=1e-6, atol=1e-8)
        assert_allclose(aligned_patterns, ref_patterns, rtol=1e-6, atol=1e-8)


class TestDemeanFalseIsNotSPoC:
    def test_uncentered_grams_are_not_matlab_cov_equivalent(self):
        X, y = _structured_problem(seed=4)
        ref_filters, _, ref_lambdas = _independent_spoc(X, y)
        model = ReDisCA(demean_time=False).fit(X, y)
        with pytest.raises(AssertionError):
            assert_allclose(model.eigenvalues_, ref_lambdas, rtol=1e-4, atol=1e-6)
        # Even after the known cov scale, filters should not match.
        scale = np.sqrt(X.shape[-1] - 1)
        aligned = _align_rows(ref_filters, model.filters_ * scale)
        assert np.linalg.norm(aligned - ref_filters) > 0.05
