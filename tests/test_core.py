"""Tests for ReDisCA numerical primitives (approved Phase 1 contract)."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import eigh

from redisca._core import (
    compute_patterns,
    mean_pair_matrix,
    pair_indices,
    pair_matrices,
    pair_matrix,
    solve_generalized_eigenproblem,
    standardize_target,
    symmetrize_matrix,
    vectorize_rdm,
    weighted_centered_mean,
)


def _random_spd(n: int, rng: np.random.Generator, *, scale: float = 1.0) -> np.ndarray:
    matrix = rng.standard_normal((n, n))
    return scale * (matrix @ matrix.T + n * np.eye(n))


class TestPairIndices:
    def test_pair_count(self):
        for n_conditions in (3, 4, 5, 10):
            pairs = pair_indices(n_conditions)
            assert len(pairs) == n_conditions * (n_conditions - 1) // 2

    def test_i_less_than_j(self):
        for i, j in pair_indices(6):
            assert i < j

    def test_deterministic_row_major_order(self):
        assert pair_indices(4) == [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        ]

    def test_no_directed_duplicates(self):
        pairs = pair_indices(5)
        assert len(pairs) == len(set(pairs))
        assert all((j, i) not in pairs for i, j in pairs)


class TestPairMatrices:
    def test_demean_false_is_uncentered_gram(self):
        rng = np.random.default_rng(0)
        x_i = rng.standard_normal((6, 11))
        x_j = rng.standard_normal((6, 11))
        delta = x_i - x_j
        got = pair_matrix(x_i, x_j, demean_time=False)
        assert_allclose(got, delta @ delta.T)

    def test_demean_true_centers_each_channel_then_gram(self):
        rng = np.random.default_rng(1)
        x_i = rng.standard_normal((6, 11)) + 3.0
        x_j = rng.standard_normal((6, 11)) - 1.5
        delta = x_i - x_j
        delta = delta - delta.mean(axis=-1, keepdims=True)
        got = pair_matrix(x_i, x_j, demean_time=True)
        assert_allclose(got, delta @ delta.T)
        assert_allclose(delta.mean(axis=-1), 0.0, atol=1e-12)

    def test_true_and_false_differ_when_means_differ(self):
        rng = np.random.default_rng(2)
        x_i = rng.standard_normal((4, 9)) + 5.0
        x_j = rng.standard_normal((4, 9))
        uncentered = pair_matrix(x_i, x_j, demean_time=False)
        centered = pair_matrix(x_i, x_j, demean_time=True)
        assert np.linalg.norm(uncentered - centered) > 1.0

    def test_symmetry_and_psd(self):
        rng = np.random.default_rng(3)
        x_i = rng.standard_normal((5, 20))
        x_j = rng.standard_normal((5, 20))
        for demean_time in (True, False):
            matrix = pair_matrix(x_i, x_j, demean_time=demean_time)
            assert_allclose(matrix, matrix.T, atol=1e-12)
            evals = np.linalg.eigvalsh(matrix)
            assert np.all(evals >= -1e-10)

    def test_identical_conditions_are_zero(self):
        rng = np.random.default_rng(4)
        x = rng.standard_normal((5, 20))
        assert_allclose(
            pair_matrix(x, x, demean_time=False),
            np.zeros((5, 5)),
            atol=1e-14,
        )

    def test_does_not_mutate_inputs(self):
        rng = np.random.default_rng(5)
        x_i = rng.standard_normal((4, 8))
        x_j = rng.standard_normal((4, 8))
        x_i_copy = x_i.copy()
        x_j_copy = x_j.copy()
        pair_matrix(x_i, x_j, demean_time=True)
        assert_array_equal(x_i, x_i_copy)
        assert_array_equal(x_j, x_j_copy)

    def test_stacked_order_matches_pairs(self):
        rng = np.random.default_rng(6)
        X = rng.standard_normal((4, 5, 12))
        pairs = pair_indices(4)
        stacked = pair_matrices(X, pairs, demean_time=True)
        assert stacked.shape == (6, 5, 5)
        for k, (i, j) in enumerate(pairs):
            assert_allclose(
                stacked[k],
                pair_matrix(X[i], X[j], demean_time=True),
            )


class TestRDMVectorAndStandardization:
    def test_vector_order_matches_pairs(self):
        y = np.array(
            [
                [0.0, 1.0, 2.0, 4.0],
                [1.0, 0.0, 3.0, 5.0],
                [2.0, 3.0, 0.0, 6.0],
                [4.0, 5.0, 6.0, 0.0],
            ]
        )
        pairs = pair_indices(4)
        assert_array_equal(vectorize_rdm(y, pairs), [1.0, 2.0, 4.0, 3.0, 5.0, 6.0])

    def test_sample_std_ddof_one(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 7.0])
        z = standardize_target(values)
        assert_allclose(np.mean(z), 0.0, atol=1e-12)
        assert_allclose(np.std(z, ddof=1), 1.0, atol=1e-12)
        expected = (values - values.mean()) / values.std(ddof=1)
        assert_allclose(z, expected)
        assert not np.isclose(np.std(z, ddof=0), 1.0)

    def test_constant_target_raises(self):
        with pytest.raises(ValueError, match="uninformative"):
            standardize_target(np.ones(6))

    @pytest.mark.parametrize("scale", [1.0, 1e-15, 1e-100, 1e15, 1e100])
    def test_scale_invariance(self, scale):
        values = np.array([0.0, 1.0, 1.0, 2.0, 0.0, 0.0])
        baseline = standardize_target(values)
        scaled = standardize_target(scale * values)
        assert_allclose(scaled, baseline, rtol=1e-10, atol=1e-12)
        assert_allclose(np.mean(scaled), 0.0, atol=1e-12)
        assert_allclose(np.std(scaled, ddof=1), 1.0, atol=1e-12)


class TestRBarAndRBarD:
    def test_hand_computed_means(self):
        # C=3, N=2, T=2, demean_time=False, explicit arithmetic.
        X = np.array(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [1.0, 1.0]],
                [[1.0, 0.0], [1.0, 0.0]],
            ]
        )
        pairs = pair_indices(3)
        R01 = (X[0] - X[1]) @ (X[0] - X[1]).T
        R02 = (X[0] - X[2]) @ (X[0] - X[2]).T
        R12 = (X[1] - X[2]) @ (X[1] - X[2]).T
        expected_bar = (R01 + R02 + R12) / 3.0

        y = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ]
        )
        v = np.array([1.0, 2.0, 3.0])
        z = (v - v.mean()) / v.std(ddof=1)
        expected_bar_d = (
            z[0] * (R01 - expected_bar)
            + z[1] * (R02 - expected_bar)
            + z[2] * (R12 - expected_bar)
        ) / 3.0

        stacked = pair_matrices(X, pairs, demean_time=False)
        r_bar = mean_pair_matrix(stacked)
        r_bar_d = weighted_centered_mean(stacked, r_bar, z)
        assert_allclose(r_bar, expected_bar)
        assert_allclose(r_bar_d, expected_bar_d)
        assert_array_equal(vectorize_rdm(y, pairs), v)

    def test_means_not_sums(self):
        rng = np.random.default_rng(7)
        stacked = rng.standard_normal((5, 3, 3))
        stacked = 0.5 * (stacked + np.swapaxes(stacked, -1, -2))
        z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        r_bar = mean_pair_matrix(stacked)
        r_bar_d = weighted_centered_mean(stacked, r_bar, z)
        assert_allclose(r_bar, stacked.mean(axis=0))
        summed = np.sum(z[:, None, None] * (stacked - r_bar), axis=0)
        assert_allclose(r_bar_d, summed / 5.0)
        assert not np.allclose(r_bar_d, summed)


class TestFullRankGEP:
    def test_matches_scipy_symmetric_gep(self):
        rng = np.random.default_rng(8)
        n_channels = 6
        r_bar = _random_spd(n_channels, rng)
        raw = rng.standard_normal((n_channels, n_channels))
        r_bar_d = 0.5 * (raw + raw.T)

        filters, lambdas = solve_generalized_eigenproblem(
            r_bar_d,
            r_bar,
            rank=None,
            rank_tol=1e-6,
        )
        scipy_vals, scipy_vecs = eigh(r_bar_d, r_bar)
        order = np.argsort(scipy_vals)[::-1]
        scipy_vals = scipy_vals[order]
        scipy_vecs = scipy_vecs[:, order]
        for k in range(n_channels):
            scipy_vecs[:, k] /= np.sqrt(scipy_vecs[:, k] @ r_bar @ scipy_vecs[:, k])
            if np.dot(scipy_vecs[:, k], filters[:, k]) < 0:
                scipy_vecs[:, k] *= -1

        assert_allclose(lambdas, scipy_vals, rtol=1e-8, atol=1e-10)
        assert_allclose(filters, scipy_vecs, rtol=1e-6, atol=1e-8)

    def test_metric_normalization_and_lambda_identity(self):
        rng = np.random.default_rng(9)
        r_bar = _random_spd(5, rng)
        raw = rng.standard_normal((5, 5))
        r_bar_d = 0.5 * (raw + raw.T)
        filters, lambdas = solve_generalized_eigenproblem(r_bar_d, r_bar)
        assert np.all(lambdas[:-1] >= lambdas[1:])
        for k in range(filters.shape[1]):
            weight = filters[:, k]
            assert_allclose(weight @ r_bar @ weight, 1.0, atol=1e-8)
            assert_allclose(weight @ r_bar_d @ weight, lambdas[k], rtol=1e-8, atol=1e-10)


class TestRankDeficientGEP:
    def _low_rank_metric(self, n_channels: int, spectrum: np.ndarray, seed: int):
        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.standard_normal((n_channels, n_channels)))
        r_bar = q @ np.diag(spectrum) @ q.T
        r_bar = 0.5 * (r_bar + r_bar.T)
        coeffs = rng.standard_normal(n_channels)
        r_bar_d = q @ np.diag(coeffs * spectrum) @ q.T
        r_bar_d = 0.5 * (r_bar_d + r_bar_d.T)
        return r_bar, r_bar_d

    def test_default_rank_tol_matches_spoc_threshold(self):
        spectrum = np.array([1.0, 0.4, 0.2, 5e-7, 1e-10, 0.0, 0.0, 0.0])
        r_bar, r_bar_d = self._low_rank_metric(8, spectrum, seed=10)
        filters, _ = solve_generalized_eigenproblem(r_bar_d, r_bar, rank=None)
        evals = np.sort(np.linalg.eigvalsh(r_bar))[::-1]
        expected = int(np.sum(evals > 1e-6 * evals[0]))
        assert expected == 3
        assert filters.shape == (8, expected)

    def test_custom_rank_tol(self):
        spectrum = np.array([1.0, 0.5, 1e-4, 1e-8, 0.0, 0.0])
        r_bar, r_bar_d = self._low_rank_metric(6, spectrum, seed=11)
        filters_loose, _ = solve_generalized_eigenproblem(
            r_bar_d, r_bar, rank=None, rank_tol=1e-6
        )
        filters_tight, _ = solve_generalized_eigenproblem(
            r_bar_d, r_bar, rank=None, rank_tol=1e-3
        )
        assert filters_loose.shape[1] == 3
        assert filters_tight.shape[1] == 2

    def test_explicit_rank_keeps_leading_principal_directions(self):
        spectrum = np.array([1.0, 0.5, 0.2, 0.05, 1e-12, 0.0])
        r_bar, r_bar_d = self._low_rank_metric(6, spectrum, seed=12)
        filters_full, _ = solve_generalized_eigenproblem(r_bar_d, r_bar, rank=None)
        filters_two, _ = solve_generalized_eigenproblem(r_bar_d, r_bar, rank=2)
        assert filters_full.shape[1] == 4
        assert filters_two.shape[1] == 2

        evals, evecs = eigh(r_bar)
        leading = evecs[:, np.argsort(evals)[::-1][:2]]
        projector = leading @ leading.T
        recovered = projector @ filters_two
        # Columns live in the leading 2-D principal subspace.
        assert_allclose(recovered, filters_two, atol=1e-8)

    def test_requested_rank_above_effective_raises(self):
        spectrum = np.array([1.0, 0.3, 1e-12, 0.0, 0.0])
        r_bar, r_bar_d = self._low_rank_metric(5, spectrum, seed=13)
        with pytest.raises(ValueError, match="exceeds the effective numerical rank"):
            solve_generalized_eigenproblem(r_bar_d, r_bar, rank=4)

    def test_scale_invariant_effective_rank(self):
        spectrum = np.array([1.0, 0.2, 1e-7, 0.0, 0.0])
        r_bar, r_bar_d = self._low_rank_metric(5, spectrum, seed=14)
        filters, _ = solve_generalized_eigenproblem(r_bar_d, r_bar)
        filters_scaled, _ = solve_generalized_eigenproblem(
            1e-16 * r_bar_d, 1e-16 * r_bar
        )
        assert filters.shape[1] == filters_scaled.shape[1]


class TestPatterns:
    def test_haufe_formula_and_reconstruction(self):
        rng = np.random.default_rng(15)
        r_bar = _random_spd(6, rng)
        raw = rng.standard_normal((6, 6))
        r_bar_d = 0.5 * (raw + raw.T)
        filters, _ = solve_generalized_eigenproblem(r_bar_d, r_bar, rank=4)
        patterns = compute_patterns(filters, r_bar)
        expected = r_bar @ filters @ np.linalg.inv(filters.T @ r_bar @ filters)
        assert patterns.shape == (6, 4)
        assert_allclose(patterns, expected, rtol=1e-8, atol=1e-10)
        assert_allclose(filters.T @ patterns, np.eye(4), atol=1e-8)

    def test_material_asymmetry_raises(self):
        matrix = np.array([[1.0, 10.0], [0.0, 1.0]])
        with pytest.raises(ValueError, match="not symmetric"):
            symmetrize_matrix(matrix, name="demo")
