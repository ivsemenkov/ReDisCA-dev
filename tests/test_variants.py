"""Tests for the experimental core variant switches."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from redisca import ReDisCA
from redisca._core import (
    mean_pair_matrix,
    metric_subspace,
    pair_indices,
    pair_matrices,
    pair_matrix,
    solve_generalized_eigenproblem,
    solve_whitening_eigenproblem,
    weighted_centered_mean,
)

from _helpers import align_rows, snapshot_problem, structured_problem


DEFAULT_EIGENVALUES = np.array(
    [
        0.15963308231093393,
        0.1512415893978392,
        0.1114155610209294,
        0.0024345451887310245,
        -0.03920377537050012,
        -0.07780961190563122,
        -0.11475854079754395,
        -0.15746669609904404,
    ]
)
DEFAULT_FILTER_HEAD = np.array(
    [
        -0.017522630422614142,
        -0.04569424058176766,
        0.038243219338079094,
        -0.06176426476274942,
    ]
)


class TestDefaultBehaviorUnchanged:
    def test_constructor_defaults(self):
        model = ReDisCA()
        assert model.demean_time is True
        assert model.divide_by_t_minus_1 is False
        assert model.directed_pairs is False
        assert model.aggregation == "mean"
        assert model.solver == "generalized"
        assert model.n_components is None
        assert model.rank is None
        assert model.rank_tol == 1e-6

    def test_snapshot_eigenvalues_and_filters(self):
        X, y = snapshot_problem()
        model = ReDisCA().fit(X, y)
        assert_allclose(model.eigenvalues_, DEFAULT_EIGENVALUES, rtol=1e-12, atol=1e-14)
        assert_allclose(model.filters_[0, :4], DEFAULT_FILTER_HEAD, rtol=1e-12, atol=1e-14)
        assert model.rank_ == 8
        assert model.centered_pair_stack_.shape[0] == 6

    def test_explicit_old_defaults_match_bare_constructor(self):
        X, y = snapshot_problem()
        bare = ReDisCA().fit(X, y)
        explicit = ReDisCA(
            n_components=None,
            demean_time=True,
            divide_by_t_minus_1=False,
            directed_pairs=False,
            aggregation="mean",
            solver="generalized",
            rank=None,
            rank_tol=1e-6,
        ).fit(X, y)
        assert_allclose(explicit.eigenvalues_, bare.eigenvalues_, rtol=0, atol=0)
        assert_array_equal(explicit.filters_, bare.filters_)
        assert_array_equal(explicit.patterns_, bare.patterns_)


class TestPairEnumeration:
    def test_unique_count_and_order(self):
        assert pair_indices(4) == [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
        ]
        assert pair_indices(4, directed=False) == pair_indices(4)
        assert len(pair_indices(6)) == 15

    def test_directed_count_and_airi_order(self):
        directed = pair_indices(4, directed=True)
        assert directed == [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 0),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (2, 3),
            (3, 0),
            (3, 1),
            (3, 2),
        ]
        assert len(directed) == 4 * 3
        assert all(i != j for i, j in directed)
        assert (0, 1) in directed and (1, 0) in directed

    def test_estimator_pair_counts(self):
        X, y = snapshot_problem()
        unique = ReDisCA().fit(X, y)
        directed = ReDisCA(directed_pairs=True).fit(X, y)
        assert unique.z_.size == 6
        assert directed.z_.size == 12
        assert unique.centered_pair_stack_.shape[0] == 6
        assert directed.centered_pair_stack_.shape[0] == 12


class TestTemporalDemeaning:
    def test_true_centers_false_does_not(self):
        rng = np.random.default_rng(1)
        x_i = rng.standard_normal((6, 11)) + 3.0
        x_j = rng.standard_normal((6, 11)) - 1.5
        delta = x_i - x_j
        uncentered = pair_matrix(x_i, x_j, demean_time=False)
        centered_delta = delta - delta.mean(axis=-1, keepdims=True)
        centered = pair_matrix(x_i, x_j, demean_time=True)
        assert_allclose(uncentered, delta @ delta.T)
        assert_allclose(centered, centered_delta @ centered_delta.T)
        assert np.linalg.norm(uncentered - centered) > 1.0

    def test_estimator_demean_changes_fit(self):
        X, y = structured_problem(seed=4)
        with_mean = ReDisCA(demean_time=True).fit(X, y)
        without = ReDisCA(demean_time=False).fit(X, y)
        assert np.linalg.norm(with_mean.eigenvalues_ - without.eigenvalues_) > 1e-3


class TestMatlabCovScale:
    def test_divide_by_t_minus_1_is_exact_scale(self):
        rng = np.random.default_rng(2)
        x_i = rng.standard_normal((5, 20))
        x_j = rng.standard_normal((5, 20))
        gram = pair_matrix(x_i, x_j, demean_time=True, divide_by_t_minus_1=False)
        cov = pair_matrix(x_i, x_j, demean_time=True, divide_by_t_minus_1=True)
        assert_allclose(cov, gram / 19.0)
        stacked = pair_matrices(
            np.stack([x_i, x_j, x_i + 1.0], axis=0),
            pair_indices(3),
            demean_time=True,
            divide_by_t_minus_1=True,
        )
        assert_allclose(
            stacked[0],
            pair_matrix(x_i, x_j, demean_time=True, divide_by_t_minus_1=True),
        )

    def test_scale_cancels_in_eigenvalues(self):
        X, y = structured_problem()
        unscaled = ReDisCA(demean_time=True, divide_by_t_minus_1=False).fit(X, y)
        scaled = ReDisCA(demean_time=True, divide_by_t_minus_1=True).fit(X, y)
        assert_allclose(unscaled.eigenvalues_, scaled.eigenvalues_, rtol=1e-8, atol=1e-10)
        scale = np.sqrt(X.shape[-1] - 1)
        aligned = align_rows(scaled.filters_, unscaled.filters_ * scale)
        assert_allclose(aligned, scaled.filters_, rtol=1e-6, atol=1e-8)
        aligned_p = align_rows(scaled.patterns_, unscaled.patterns_ / scale)
        assert_allclose(aligned_p, scaled.patterns_, rtol=1e-6, atol=1e-8)

    def test_single_sample_scale_raises(self):
        rng = np.random.default_rng(3)
        X = rng.standard_normal((3, 4, 1))
        y = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ]
        )
        with pytest.raises(ValueError, match="divide_by_t_minus_1"):
            ReDisCA(demean_time=False, divide_by_t_minus_1=True).fit(X, y)


class TestAggregation:
    def test_sum_is_pair_count_times_mean(self):
        rng = np.random.default_rng(7)
        stacked = rng.standard_normal((5, 3, 3))
        stacked = 0.5 * (stacked + np.swapaxes(stacked, -1, -2))
        z = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        r_bar = mean_pair_matrix(stacked)
        mean_mat = weighted_centered_mean(stacked, r_bar, z, aggregation="mean")
        sum_mat = weighted_centered_mean(stacked, r_bar, z, aggregation="sum")
        assert_allclose(sum_mat, mean_mat * 5.0)

    def test_estimator_sum_scales_eigenvalues(self):
        X, y = structured_problem()
        mean_model = ReDisCA(aggregation="mean").fit(X, y)
        sum_model = ReDisCA(aggregation="sum").fit(X, y)
        n_pairs = mean_model.z_.size
        assert_allclose(
            sum_model.eigenvalues_,
            mean_model.eigenvalues_ * n_pairs,
            rtol=1e-8,
            atol=1e-10,
        )
        aligned = align_rows(mean_model.filters_, sum_model.filters_)
        assert_allclose(aligned, mean_model.filters_, rtol=1e-6, atol=1e-8)
        aligned_p = align_rows(mean_model.patterns_, sum_model.patterns_)
        assert_allclose(aligned_p, mean_model.patterns_, rtol=1e-6, atol=1e-8)


class TestSolvers:
    def test_generalized_matches_whitening_on_identical_inputs(self):
        X, y = structured_problem()
        generalized = ReDisCA(solver="generalized").fit(X, y)
        whitening = ReDisCA(solver="whitening").fit(X, y)
        assert generalized.rank_ == whitening.rank_
        assert_allclose(
            generalized.eigenvalues_,
            whitening.eigenvalues_,
            rtol=1e-8,
            atol=1e-10,
        )
        aligned = align_rows(generalized.filters_, whitening.filters_)
        assert_allclose(aligned, generalized.filters_, rtol=1e-6, atol=1e-8)
        aligned_p = align_rows(generalized.patterns_, whitening.patterns_)
        assert_allclose(aligned_p, generalized.patterns_, rtol=1e-6, atol=1e-8)
        reconstruction_g = generalized.patterns_.T @ generalized.filters_
        reconstruction_w = whitening.patterns_.T @ whitening.filters_
        assert_allclose(reconstruction_g, reconstruction_w, rtol=1e-6, atol=1e-8)

    def test_core_solvers_match_on_low_rank_metric(self):
        rng = np.random.default_rng(10)
        n_channels = 8
        spectrum = np.array([1.0, 0.4, 0.2, 5e-7, 1e-10, 0.0, 0.0, 0.0])
        q, _ = np.linalg.qr(rng.standard_normal((n_channels, n_channels)))
        r_bar = q @ np.diag(spectrum) @ q.T
        r_bar = 0.5 * (r_bar + r_bar.T)
        coeffs = rng.standard_normal(n_channels)
        r_bar_d = q @ np.diag(coeffs * spectrum) @ q.T
        r_bar_d = 0.5 * (r_bar_d + r_bar_d.T)

        filters_g, lambdas_g = solve_generalized_eigenproblem(r_bar_d, r_bar)
        filters_w, lambdas_w = solve_whitening_eigenproblem(r_bar_d, r_bar)
        evals = np.sort(np.linalg.eigvalsh(r_bar))[::-1]
        expected_rank = int(np.sum(evals > 1e-6 * evals[0]))
        assert expected_rank == 3
        assert filters_g.shape == (8, 3)
        assert filters_w.shape == (8, 3)
        assert_allclose(lambdas_g, lambdas_w, rtol=1e-8, atol=1e-10)
        for k in range(3):
            if np.dot(filters_g[:, k], filters_w[:, k]) < 0:
                filters_w[:, k] *= -1
        assert_allclose(filters_g, filters_w, rtol=1e-6, atol=1e-8)

    def test_rank_threshold_parity(self):
        rng = np.random.default_rng(11)
        spectrum = np.array([1.0, 0.5, 1e-4, 1e-8, 0.0, 0.0])
        q, _ = np.linalg.qr(rng.standard_normal((6, 6)))
        r_bar = 0.5 * (q @ np.diag(spectrum) @ q.T + (q @ np.diag(spectrum) @ q.T).T)
        r_bar_d = 0.5 * (q @ np.diag(rng.standard_normal(6) * spectrum) @ q.T)
        r_bar_d = 0.5 * (r_bar_d + r_bar_d.T)
        sub_loose = metric_subspace(r_bar, rank_tol=1e-6)
        sub_tight = metric_subspace(r_bar, rank_tol=1e-3)
        assert sub_loose.used_rank == 3
        assert sub_tight.used_rank == 2
        assert sub_loose.effective_rank == 3
        filters_g, _ = solve_generalized_eigenproblem(r_bar_d, r_bar, rank_tol=1e-3)
        filters_w, _ = solve_whitening_eigenproblem(r_bar_d, r_bar, rank_tol=1e-3)
        assert filters_g.shape[1] == filters_w.shape[1] == 2

    def test_metric_normalization_both_solvers(self):
        X, y = structured_problem(seed=9)
        for solver in ("generalized", "whitening"):
            model = ReDisCA(solver=solver).fit(X, y)
            for weight, value in zip(model.filters_, model.eigenvalues_):
                assert_allclose(weight @ model.r_bar_ @ weight, 1.0, atol=1e-8)
                assert_allclose(
                    weight @ model.r_bar_d_ @ weight,
                    value,
                    rtol=1e-8,
                    atol=1e-10,
                )


class TestInvalidVariantParams:
    def test_rejected_values(self):
        X, y = snapshot_problem()
        with pytest.raises(ValueError, match="aggregation"):
            ReDisCA(aggregation="average").fit(X, y)
        with pytest.raises(ValueError, match="solver"):
            ReDisCA(solver="spoc").fit(X, y)
        with pytest.raises(TypeError, match="directed_pairs"):
            ReDisCA(directed_pairs="yes").fit(X, y)
        with pytest.raises(TypeError, match="divide_by_t_minus_1"):
            ReDisCA(divide_by_t_minus_1=1).fit(X, y)
