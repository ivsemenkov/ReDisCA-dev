"""Tests for the public sklearn-style ReDisCA estimator."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from redisca import ReDisCA


def _structured_data(seed: int = 0, *, n_channels: int = 8, n_times: int = 40):
    rng = np.random.default_rng(seed)
    n_conditions = 4
    mixing = rng.standard_normal((n_channels, 3))
    sources = rng.standard_normal((3, n_times))
    amplitudes = np.array(
        [
            [1.0, 0.2, 0.1],
            [1.0, -0.2, 0.1],
            [-1.0, 0.2, 0.1],
            [-1.0, -0.2, 0.1],
        ]
    )
    X = np.zeros((n_conditions, n_channels, n_times))
    for condition in range(n_conditions):
        X[condition] = mixing @ (amplitudes[condition, :, None] * sources)
        X[condition] += 0.05 * rng.standard_normal((n_channels, n_times))
    y = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    return X, y


class TestPublicOrientation:
    def test_component_rows(self):
        X, y = _structured_data()
        model = ReDisCA().fit(X, y)
        n_channels = X.shape[1]
        assert model.filters_.shape == (model.rank_, n_channels)
        assert model.patterns_.shape == (model.rank_, n_channels)
        assert model.eigenvalues_.shape == (model.rank_,)
        assert model.n_features_in_ == n_channels
        assert model.n_conditions_ == X.shape[0]
        assert model.n_times_in_ == X.shape[2]
        assert model.n_components_ == model.rank_
        assert model.n_components_ == model.filters_.shape[0]


class TestNComponents:
    def test_none_uses_all_components(self):
        X, y = _structured_data()
        model = ReDisCA(n_components=None).fit(X, y)
        transformed = model.transform(X)
        assert transformed.shape[1] == model.rank_

    def test_does_not_change_decomposition(self):
        X, y = _structured_data()
        full = ReDisCA().fit(X, y)
        sliced = ReDisCA(n_components=2).fit(X, y)
        assert sliced.rank_ == full.rank_
        assert_allclose(sliced.filters_, full.filters_)
        assert_allclose(sliced.patterns_, full.patterns_)
        assert_allclose(sliced.eigenvalues_, full.eigenvalues_)
        assert sliced.n_components_ == 2
        assert sliced.filters_.shape[0] == sliced.rank_
        assert sliced.transform(X).shape[1] == 2
        assert sliced.inverse_transform(sliced.transform(X)).shape[1] == X.shape[1]

    def test_greater_than_rank_raises(self):
        X, y = _structured_data(n_channels=5, n_times=20)
        fitted_rank = ReDisCA().fit(X, y).rank_
        with pytest.raises(ValueError, match="n_components"):
            ReDisCA(n_components=fitted_rank + 3).fit(X, y)

    def test_set_params_after_fit_does_not_change_transform_until_refit(self):
        X, y = _structured_data()
        model = ReDisCA(n_components=2).fit(X, y)
        before = model.transform(X)
        model.set_params(n_components=1)
        after = model.transform(X)
        assert after.shape == before.shape
        assert_allclose(after, before)
        model.fit(X, y)
        assert model.transform(X).shape[1] == 1


class TestSklearnAPI:
    def test_fit_returns_self(self):
        X, y = _structured_data()
        model = ReDisCA()
        assert model.fit(X, y) is model

    def test_clone_and_params(self):
        model = ReDisCA(n_components=2, demean_time=False, rank=3, rank_tol=1e-5)
        cloned = clone(model)
        assert cloned is not model
        assert cloned.get_params() == model.get_params()
        model.set_params(n_components=1, demean_time=True)
        assert model.n_components == 1
        assert model.demean_time is True
        assert cloned.n_components == 2
        assert not hasattr(cloned, "filters_")
        defaults = ReDisCA().get_params()
        assert defaults["divide_by_t_minus_1"] is False
        assert defaults["directed_pairs"] is False
        assert defaults["aggregation"] == "mean"
        assert defaults["solver"] == "generalized"

    def test_transform_before_fit_raises(self):
        X, y = _structured_data()
        model = ReDisCA()
        with pytest.raises(NotFittedError):
            model.transform(X)
        with pytest.raises(NotFittedError):
            model.inverse_transform(X)

    def test_fit_transform_matches_fit_then_transform(self):
        X, y = _structured_data()
        model = ReDisCA(n_components=2)
        chained = model.fit_transform(X, y)
        other = ReDisCA(n_components=2).fit(X, y).transform(X)
        assert_allclose(chained, other)

    def test_fit_without_y_raises(self):
        X, _ = _structured_data()
        with pytest.raises(TypeError, match="target representational dissimilarity"):
            ReDisCA().fit(X, None)

    def test_no_score_method(self):
        assert "score" not in ReDisCA.__dict__


class TestTransform:
    def test_accepts_different_observations_and_times(self):
        X, y = _structured_data(n_times=30)
        model = ReDisCA(n_components=2).fit(X, y)
        rng = np.random.default_rng(21)
        X_new = rng.standard_normal((7, X.shape[1], 13))
        U = model.transform(X_new)
        assert U.shape == (7, 2, 13)
        expected = np.einsum("rc,oct->ort", model.filters_[:2], X_new)
        assert_allclose(U, expected)

    def test_rejects_channel_mismatch_and_2d(self):
        X, y = _structured_data()
        model = ReDisCA().fit(X, y)
        with pytest.raises(ValueError, match="channels"):
            model.transform(np.zeros((3, X.shape[1] + 1, 10)))
        with pytest.raises(ValueError, match="shape"):
            model.transform(np.zeros((X.shape[1], 10)))

    def test_does_not_demean_or_mutate(self):
        X, y = _structured_data()
        X = X + 4.0
        model = ReDisCA(demean_time=True).fit(X, y)
        X_new = X[:2] + 10.0
        snapshot = X_new.copy()
        U = model.transform(X_new)
        assert_array_equal(X_new, snapshot)
        expected = np.einsum(
            "rc,oct->ort",
            model.filters_[: model.n_components_],
            snapshot,
        )
        assert_allclose(U, expected)

    def test_single_observation(self):
        X, y = _structured_data()
        model = ReDisCA(n_components=1).fit(X, y)
        U = model.transform(X[:1])
        assert U.shape == (1, 1, X.shape[2])


class TestInverseTransform:
    def test_explicit_patterns_multiplication(self):
        X, y = _structured_data()
        model = ReDisCA(n_components=2).fit(X, y)
        U = model.transform(X)
        reconstructed = model.inverse_transform(U)
        expected = np.einsum("rc,ort->oct", model.patterns_[:2], U)
        assert reconstructed.shape == (X.shape[0], X.shape[1], X.shape[2])
        assert_allclose(reconstructed, expected)

    def test_full_rank_subspace_roundtrip(self):
        X, y = _structured_data(n_channels=6, n_times=25)
        model = ReDisCA().fit(X, y)
        U = model.transform(X)
        reconstructed = model.inverse_transform(U)
        projector = model.patterns_.T @ model.filters_
        expected = np.einsum("cn,ont->oct", projector, X)
        assert_allclose(reconstructed, expected, rtol=1e-6, atol=1e-8)

    def test_component_count_mismatch_and_not_fitted(self):
        X, y = _structured_data()
        model = ReDisCA(n_components=2).fit(X, y)
        U = model.transform(X)
        with pytest.raises(ValueError, match="components"):
            model.inverse_transform(U[:, :1])
        with pytest.raises(NotFittedError):
            ReDisCA().inverse_transform(U)


class TestParameterTypesAndYConstraints:
    def test_bool_rank_and_n_components_rejected(self):
        X, y = _structured_data()
        with pytest.raises(TypeError, match="n_components"):
            ReDisCA(n_components=True).fit(X, y)
        with pytest.raises(TypeError, match="rank"):
            ReDisCA(rank=True).fit(X, y)

    def test_nonpositive_and_invalid_rank_tol(self):
        X, y = _structured_data()
        with pytest.raises(ValueError, match="n_components"):
            ReDisCA(n_components=0).fit(X, y)
        with pytest.raises(ValueError, match="rank"):
            ReDisCA(rank=-1).fit(X, y)
        with pytest.raises(TypeError, match="rank_tol"):
            ReDisCA(rank_tol=True).fit(X, y)
        for rank_tol in (1.0, 1.5, 0.0, -1.0, np.nan, np.inf):
            with pytest.raises(ValueError, match="rank_tol"):
                ReDisCA(rank_tol=rank_tol).fit(X, y)

    def test_signed_symmetric_y_is_accepted(self):
        X, y = _structured_data()
        y = y.copy()
        y[0, 2] = -1.5
        y[2, 0] = -1.5
        model = ReDisCA().fit(X, y)
        assert model.rank_ >= 1

    def test_y_validation_errors(self):
        X, y = _structured_data()
        with pytest.raises(ValueError, match="square"):
            ReDisCA().fit(X, y[:, :3])
        with pytest.raises(ValueError, match="does not match"):
            ReDisCA().fit(X, y[:3, :3])
        asymmetric = y.copy()
        asymmetric[0, 3] = 8.0
        with pytest.raises(ValueError, match="symmetric"):
            ReDisCA().fit(X, asymmetric)
        nonzero_diag = y.copy()
        nonzero_diag[1, 1] = 0.4
        with pytest.raises(ValueError, match="diagonal"):
            ReDisCA().fit(X, nonzero_diag)
        with pytest.raises(ValueError, match="2D"):
            ReDisCA().fit(X, y.ravel())
        with pytest.raises(ValueError, match="NaN"):
            bad = y.copy()
            bad[0, 1] = np.nan
            bad[1, 0] = np.nan
            ReDisCA().fit(X, bad)

    def test_too_few_conditions_and_constant_target(self):
        rng = np.random.default_rng(3)
        X2 = rng.standard_normal((2, 5, 10))
        y2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        with pytest.raises(ValueError, match="at least 3"):
            ReDisCA().fit(X2, y2)

        X, y = _structured_data()
        constant = np.ones_like(y) - np.eye(y.shape[0])
        with pytest.raises(ValueError, match="uninformative"):
            ReDisCA().fit(X, constant)

    @pytest.mark.parametrize("scale", [1.0, 1e-15, 1e-100, 1e15, 1e100])
    def test_scaled_target_rdm_gives_equivalent_decomposition(self, scale):
        X, y = _structured_data()
        baseline = ReDisCA().fit(X, y)
        scaled = ReDisCA().fit(X, scale * y)
        assert_allclose(
            scaled.eigenvalues_,
            baseline.eigenvalues_,
            rtol=1e-8,
            atol=1e-10,
        )
        assert scaled.rank_ == baseline.rank_
        aligned = scaled.filters_.copy()
        for index in range(aligned.shape[0]):
            if np.dot(baseline.filters_[index], aligned[index]) < 0:
                aligned[index] *= -1
        assert_allclose(aligned, baseline.filters_, rtol=1e-6, atol=1e-8)

    def test_demean_time_true_requires_two_samples(self):
        rng = np.random.default_rng(4)
        X = rng.standard_normal((3, 4, 1))
        y = np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 0.0, 3.0],
                [2.0, 3.0, 0.0],
            ]
        )
        with pytest.raises(ValueError, match="at least 2 time samples"):
            ReDisCA(demean_time=True).fit(X, y)
        model = ReDisCA(demean_time=False).fit(X, y)
        assert model.n_times_in_ == 1

    def test_fit_does_not_mutate_X(self):
        X, y = _structured_data()
        snapshot = X.copy()
        ReDisCA(demean_time=True).fit(X, y)
        assert_array_equal(X, snapshot)

    def test_row_orientation_identity_on_patterns(self):
        X, y = _structured_data()
        model = ReDisCA().fit(X, y)
        assert_allclose(
            model.filters_ @ model.patterns_.T,
            np.eye(model.rank_),
            atol=1e-6,
        )
