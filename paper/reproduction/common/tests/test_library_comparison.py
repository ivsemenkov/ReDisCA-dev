"""Documented comparison between the public library and the source-faithful reconstruction.

This test imports ``redisca`` on purpose. It is not an independent verification
of ``source_faithful.py``; it records where the two paths agree.
"""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from common.metrics import sign_align_vectors
from common.source_faithful import fit_condition_averages
from redisca import ReDisCA


def _problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_conditions, n_channels, n_times = 5, 8, 36
    mixing = rng.standard_normal((n_channels, 3))
    sources = rng.standard_normal((3, n_times))
    X = np.stack([mixing @ (sources * rng.uniform(0.4, 1.6, size=(3, 1))) for _ in range(n_conditions)])
    y = np.zeros((n_conditions, n_conditions))
    for i in range(n_conditions):
        for j in range(i + 1, n_conditions):
            y[i, j] = y[j, i] = abs(i - j)
    return X, y


def test_unique_matlab_cov_path_matches_library_demean_time_true() -> None:
    X, y = _problem(11)
    library = ReDisCA(demean_time=True).fit(X, y)
    faithful = fit_condition_averages(
        X, y, pair_mode="unique_unordered", matrix_mode="matlab_cov"
    )
    n = library.eigenvalues_.size
    # Global T-1 scale in MATLAB cov does not change the GEP. Temporal
    # demeaning is shared. Aggregation is a mean in both cases.
    assert_allclose(library.eigenvalues_, faithful.eigenvalues[:n], atol=1e-8, rtol=1e-8)
    aligned = sign_align_vectors(library.filters_, faithful.filters[:, :n].T)
    for k in range(n):
        corr = np.corrcoef(library.filters_[k], aligned[k])[0, 1]
        assert abs(corr) > 0.999


def test_unscaled_gram_path_matches_library_demean_time_false() -> None:
    X, y = _problem(12)
    library = ReDisCA(demean_time=False).fit(X, y)
    faithful = fit_condition_averages(
        X, y, pair_mode="unique_unordered", matrix_mode="unscaled_gram"
    )
    n = library.eigenvalues_.size
    assert_allclose(library.eigenvalues_, faithful.eigenvalues[:n], atol=1e-8, rtol=1e-8)
