"""N170 unit tests that do not require the ERP CORE cache."""

from __future__ import annotations

import numpy as np

from paper.reproduction.common.method import fit_redisca
from paper.reproduction.n170.prepare import sliding_centers_ms, window_slice
from paper.reproduction.n170.rdms import face_rdm, meaning_rdm


def test_rdms_are_symmetric_zero_diag():
    for matrix in (meaning_rdm(), face_rdm()):
        assert matrix.shape == (4, 4)
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), 0.0)


def test_window_slice_100ms_at_256hz():
    times = -200.0 + np.arange(256) * (1000.0 / 256.0)
    data = np.zeros((4, 3, 256))
    sliced = window_slice(data, times, center_ms=200.0, duration_ms=100.0)
    assert sliced["n_samples"] >= 20
    assert sliced["data"].shape[-1] == sliced["n_samples"]


def test_sliding_25ms_includes_400():
    times = -200.0 + np.arange(256) * (1000.0 / 256.0)
    centers = sliding_centers_ms(times, duration_ms=150.0, step_ms=25.0)
    assert any(abs(c - 400.0) < 1e-6 for c in centers)


def test_fit_factory_on_tiny_n170_like_array():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, 6, 26))
    model = fit_redisca(X, face_rdm())
    assert model.eigenvalues_.size == model.rank_
    assert model.filters_.shape[1] == 6
