"""Tests for the stock-SPoC random-phase test."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.exceptions import NotFittedError

from redisca import ReDisCA, random_phase_test
from redisca._core import metric_subspace, subspace_eigenvalues, weighted_aggregate
from redisca.inference import random_phase_surrogate

from _helpers import snapshot_problem, structured_problem


class TestRandomPhaseSurrogate:
    def test_preserves_non_nyquist_amplitudes(self):
        rng = np.random.default_rng(1)
        z = rng.standard_normal(32)
        surrogate, amplitudes = random_phase_surrogate(z, rng)
        got = np.abs(np.fft.fft(surrogate))
        nyquist = z.size // 2
        mask = np.ones(z.size, dtype=bool)
        mask[nyquist] = False
        assert_allclose(got[mask], amplitudes[mask], atol=1e-10)
        assert surrogate.shape == z.shape

    def test_odd_length(self):
        rng = np.random.default_rng(2)
        z = rng.standard_normal(15)
        surrogate, amplitudes = random_phase_surrogate(z, rng)
        got = np.abs(np.fft.fft(surrogate))
        assert_allclose(got, amplitudes, atol=1e-10)


class TestRandomPhaseTest:
    def test_requires_fitted_estimator(self):
        with pytest.raises(NotFittedError):
            random_phase_test(ReDisCA(), n_surrogates=4, random_state=0)

    def test_reuses_fitted_state_and_does_not_refit(self):
        X, y = structured_problem(seed=5)
        X = X.copy()
        y = y.copy()

        class CountingReDisCA(ReDisCA):
            def fit(self, X, y):
                self.fit_calls = getattr(self, "fit_calls", 0) + 1
                return super().fit(X, y)

        model = CountingReDisCA().fit(X, y)
        assert model.fit_calls == 1
        filters_before = model.filters_.copy()
        evals_before = model.eigenvalues_.copy()
        z_before = model.z_.copy()
        stack_before = model.centered_pair_stack_.copy()

        X += 1000.0
        y += 1.0
        np.fill_diagonal(y, 0.0)
        first = random_phase_test(model, n_surrogates=8, random_state=0)
        second = random_phase_test(model, n_surrogates=24, random_state=0)

        assert model.fit_calls == 1
        assert_array_equal(model.filters_, filters_before)
        assert_array_equal(model.eigenvalues_, evals_before)
        assert_array_equal(model.z_, z_before)
        assert_array_equal(model.centered_pair_stack_, stack_before)
        assert first.n_surrogates == 8
        assert second.n_surrogates == 24
        assert first.p_values.shape == model.eigenvalues_.shape
        assert second.null_statistic.shape == (24,)
        assert_allclose(second.null_statistic[:8], first.null_statistic)

    def test_max_abs_eigenvalue_null_and_count_over_b(self):
        X, y = structured_problem(seed=6)
        model = ReDisCA(solver="whitening").fit(X, y)
        result = random_phase_test(model, n_surrogates=5, random_state=0)

        rng = np.random.default_rng(0)
        subspace = metric_subspace(
            model.r_bar_, rank=model.rank, rank_tol=model.rank_tol
        )
        amplitudes = None
        expected_null = np.empty(5)
        z = model.z_.copy()
        for index in range(5):
            z_s, amplitudes = random_phase_surrogate(z, rng, amplitudes=amplitudes)
            cxxz_s = weighted_aggregate(
                model.centered_pair_stack_, z_s, aggregation=model.aggregation
            )
            evals = subspace_eigenvalues(cxxz_s, subspace, solver=model.solver)
            expected_null[index] = float(np.max(np.abs(evals)))
        assert_allclose(result.null_statistic, expected_null)

        expected_p = np.array(
            [
                float(np.sum(expected_null >= abs(value)) / 5)
                for value in model.eigenvalues_
            ]
        )
        assert_allclose(result.p_values, expected_p)
        assert np.all(result.p_values >= 0.0)
        assert np.all(result.p_values <= 1.0)

    def test_p_zero_allowed(self):
        observed = np.array([10.0, 0.1])
        null = np.array([1.0, 2.0, 3.0])
        p_values = np.array(
            [float(np.sum(null >= abs(value)) / null.size) for value in observed]
        )
        assert p_values[0] == 0.0
        assert p_values[1] == 1.0

        rng = np.random.default_rng(4)
        n_times = 50
        mixing = rng.standard_normal((8, 1))
        t = np.linspace(0.0, 1.0, n_times)
        X = np.stack(
            [
                mixing * np.sin(2 * np.pi * 3 * t) * amp
                for amp in (1.0, 1.05, 4.0, 4.1)
            ],
            axis=0,
        )
        y = np.array(
            [
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0, 0.0],
            ]
        )
        model = ReDisCA(
            demean_time=True,
            divide_by_t_minus_1=True,
            directed_pairs=True,
            solver="whitening",
        ).fit(X, y)
        result = random_phase_test(model, n_surrogates=20, random_state=4)
        leading = int(np.argmax(np.abs(model.eigenvalues_)))
        if np.abs(model.eigenvalues_[leading]) > np.max(result.null_statistic):
            assert result.p_values[leading] == 0.0
        assert np.all(
            np.isclose(result.p_values * 20.0, np.round(result.p_values * 20.0))
        )

    def test_deterministic_for_fixed_python_random_state(self):
        X, y = snapshot_problem()
        model = ReDisCA().fit(X, y)
        first = random_phase_test(model, n_surrogates=12, random_state=123)
        second = random_phase_test(model, n_surrogates=12, random_state=123)
        other = random_phase_test(model, n_surrogates=12, random_state=124)
        assert_array_equal(first.p_values, second.p_values)
        assert_array_equal(first.null_statistic, second.null_statistic)
        assert not np.array_equal(first.null_statistic, other.null_statistic)

    def test_invalid_n_surrogates(self):
        X, y = snapshot_problem()
        model = ReDisCA().fit(X, y)
        with pytest.raises(TypeError, match="n_surrogates"):
            random_phase_test(model, n_surrogates=True)
        with pytest.raises(ValueError, match="n_surrogates"):
            random_phase_test(model, n_surrogates=0)
        with pytest.raises(TypeError, match="random_state"):
            random_phase_test(model, n_surrogates=3, random_state=True)
