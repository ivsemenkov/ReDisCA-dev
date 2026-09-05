"""Parity checks against the paper-branch source-faithful reconstruction.

Loads ``paper/reproduction/common/source_faithful.py`` from ``origin/paper``.
That module is an independent oracle and is not part of the library.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from redisca import ReDisCA

from _helpers import align_rows, structured_problem

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_source_faithful():
    try:
        source = subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "show",
                "origin/paper:paper/reproduction/common/source_faithful.py",
            ],
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"origin/paper source_faithful.py is unavailable: {exc}")
    path = Path("/tmp/source_faithful_paper_oracle.py")
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location("source_faithful_paper_oracle", path)
    if spec is None or spec.loader is None:
        pytest.skip("could not load source_faithful.py from origin/paper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def source_faithful():
    return _load_source_faithful()


def _distance_rdm(n_conditions: int) -> np.ndarray:
    y = np.zeros((n_conditions, n_conditions))
    for i in range(n_conditions):
        for j in range(i + 1, n_conditions):
            y[i, j] = y[j, i] = float(abs(i - j))
    return y


class TestSourceFaithfulParity:
    def test_airi_historical_config(self, source_faithful):
        X, y = structured_problem(seed=11, n_channels=8, n_times=36)
        library = ReDisCA(
            demean_time=True,
            divide_by_t_minus_1=True,
            directed_pairs=True,
            aggregation="mean",
            solver="whitening",
        ).fit(X, y)
        faithful = source_faithful.fit_condition_averages(
            X, y, pair_mode="airi_directed", matrix_mode="matlab_cov"
        )
        n = library.eigenvalues_.size
        assert library.rank_ == n
        assert_allclose(
            library.eigenvalues_,
            faithful.eigenvalues[:n],
            rtol=1e-8,
            atol=1e-10,
        )
        aligned = align_rows(library.filters_, faithful.filters[:, :n].T)
        assert_allclose(aligned, library.filters_, rtol=1e-6, atol=1e-8)
        aligned_p = align_rows(library.patterns_, faithful.patterns[:, :n].T)
        assert_allclose(aligned_p, library.patterns_, rtol=1e-6, atol=1e-8)

    def test_unique_matlab_cov_config(self, source_faithful):
        X, y = structured_problem(seed=12, n_channels=8, n_times=36)
        library = ReDisCA(
            demean_time=True,
            divide_by_t_minus_1=True,
            directed_pairs=False,
            aggregation="mean",
            solver="whitening",
        ).fit(X, y)
        faithful = source_faithful.fit_condition_averages(
            X, y, pair_mode="unique_unordered", matrix_mode="matlab_cov"
        )
        n = library.eigenvalues_.size
        assert_allclose(
            library.eigenvalues_,
            faithful.eigenvalues[:n],
            rtol=1e-8,
            atol=1e-10,
        )
        aligned = align_rows(library.filters_, faithful.filters[:, :n].T)
        assert_allclose(aligned, library.filters_, rtol=1e-6, atol=1e-8)

    def test_paper_unscaled_gram_config(self, source_faithful):
        rng = np.random.default_rng(13)
        X = rng.standard_normal((5, 7, 30))
        y = _distance_rdm(5)
        library = ReDisCA(
            demean_time=False,
            divide_by_t_minus_1=False,
            directed_pairs=False,
            aggregation="mean",
            solver="generalized",
        ).fit(X, y)
        faithful = source_faithful.fit_condition_averages(
            X, y, pair_mode="unique_unordered", matrix_mode="unscaled_gram"
        )
        n = library.eigenvalues_.size
        assert_allclose(
            library.eigenvalues_,
            faithful.eigenvalues[:n],
            rtol=1e-8,
            atol=1e-10,
        )
        aligned = align_rows(library.filters_, faithful.filters[:, :n].T)
        assert_allclose(aligned, library.filters_, rtol=1e-6, atol=1e-8)

    def test_default_library_matches_unique_matlab_cov_eigenvalues(self, source_faithful):
        """Default library omits ``1/(T-1)``; eigenvalues still match MATLAB cov."""
        X, y = structured_problem(seed=14)
        library = ReDisCA().fit(X, y)
        faithful = source_faithful.fit_condition_averages(
            X, y, pair_mode="unique_unordered", matrix_mode="matlab_cov"
        )
        n = library.eigenvalues_.size
        assert_allclose(
            library.eigenvalues_,
            faithful.eigenvalues[:n],
            rtol=1e-8,
            atol=1e-10,
        )
