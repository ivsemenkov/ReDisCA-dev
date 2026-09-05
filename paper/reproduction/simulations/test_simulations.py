"""Simulation unit tests that do not require the AD forward model."""

from __future__ import annotations

import numpy as np

from paper.reproduction.common.method import fit_redisca
from paper.reproduction.simulations.generate import add_symmetric_rdm_noise, squared_euclidean_rdm
from paper.reproduction.simulations.metrics_roc import cosine_abs_scan, roc_from_mc


def test_squared_euclidean_rdm_properties():
    series = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float64)
    rdm = squared_euclidean_rdm(series)
    assert rdm.shape == (3, 3)
    assert np.allclose(np.diag(rdm), 0.0)
    assert np.allclose(rdm, rdm.T)
    assert rdm[0, 1] == 2.0


def test_rdm_noise_is_symmetric_nonneg():
    rng = np.random.default_rng(0)
    d0 = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
    noisy = add_symmetric_rdm_noise(d0, rng, relative_std=0.05)
    assert np.allclose(noisy, noisy.T)
    assert np.allclose(np.diag(noisy), 0.0)
    assert np.all(noisy >= 0.0)


def test_roc_and_cosine_helpers():
    scores = np.array([[1.0, 0.2, 0.0], [0.9, 0.1, 0.0]])
    inside = np.array([[True, False, False], [True, False, False]])
    roc = roc_from_mc(scores, inside, np.array([0.5, 0.0]))
    assert roc["auc"] >= 0.0
    gain = np.eye(3)
    scan = cosine_abs_scan(np.array([1.0, 0.0, 0.0]), gain)
    assert scan[0] == 1.0


def test_redisca_factory_on_simulated_averages():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((5, 8, 40))
    y = np.ones((5, 5))
    np.fill_diagonal(y, 0.0)
    y[0, 1] = y[1, 0] = 0.1
    model = fit_redisca(X, y)
    assert model.n_conditions_ == 5
    assert model.filters_.shape[1] == 8
