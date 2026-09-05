"""Validate current-main AIRI-SPoC ReDisCA against the independent oracle."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from paper.reproduction.common.method import AIRI_SPOC_KWARGS, fit_redisca, make_redisca
from paper.reproduction.validation.oracle import align_rows, independent_airi_spoc


def _structured_problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    n_conditions, n_channels, n_times = 5, 8, 40
    mixing = rng.standard_normal((n_channels, 3))
    sources = rng.standard_normal((3, n_times))
    base = mixing @ sources
    X = np.stack([base + 0.15 * rng.standard_normal((n_channels, n_times)) for _ in range(n_conditions)])
    y = np.array(
        [
            [0.0, 0.1, 1.0, 1.0, 0.5],
            [0.1, 0.0, 1.0, 1.0, 0.5],
            [1.0, 1.0, 0.0, 0.1, 0.5],
            [1.0, 1.0, 0.1, 0.0, 0.5],
            [0.5, 0.5, 0.5, 0.5, 0.0],
        ],
        dtype=np.float64,
    )
    return X, y


def test_factory_kwargs_are_exactly_airi_spoc():
    model = make_redisca()
    for key, value in AIRI_SPOC_KWARGS.items():
        assert getattr(model, key) == value


def test_fit_matches_independent_oracle():
    X, y = _structured_problem()
    model = fit_redisca(X, y)
    ref = independent_airi_spoc(X, y)
    assert model.rank_ == int(ref["rank"][0])
    assert_allclose(model.eigenvalues_, ref["eigenvalues"], rtol=1e-8, atol=1e-10)
    assert_allclose(model.z_, ref["z"], rtol=1e-12, atol=1e-12)
    assert_allclose(model.r_bar_, ref["cxx"], rtol=1e-10, atol=1e-12)
    assert_allclose(model.r_bar_d_, ref["cxxz"], rtol=1e-10, atol=1e-12)
    aligned_filters = align_rows(ref["filters"], model.filters_)
    aligned_patterns = align_rows(ref["patterns"], model.patterns_)
    assert_allclose(aligned_filters, ref["filters"], rtol=1e-7, atol=1e-8)
    assert_allclose(aligned_patterns, ref["patterns"], rtol=1e-7, atol=1e-8)


def test_random_phase_does_not_refit():
    from redisca import random_phase_test

    X, y = _structured_problem(1)
    model = fit_redisca(X, y)
    evals_before = model.eigenvalues_.copy()
    filters_before = model.filters_.copy()
    result = random_phase_test(model, n_surrogates=32, random_state=0)
    assert result.n_surrogates == 32
    assert result.p_values.shape == model.eigenvalues_.shape
    assert_allclose(model.eigenvalues_, evals_before)
    assert_allclose(model.filters_, filters_before)
    assert np.all((result.p_values >= 0.0) & (result.p_values <= 1.0))


def _runtime_modules(root: Path):
    for path in root.rglob("*.py"):
        if path.name.startswith("test_") or path.name == "conftest.py":
            continue
        if "tests" in path.parts or "validation" in path.parts:
            continue
        yield path


def test_oracle_is_not_imported_by_experiment_modules():
    root = Path(__file__).resolve().parents[1]
    forbidden = ("source_faithful", "validation.oracle", "independent_airi_spoc")
    offenders = []
    for path in _runtime_modules(root):
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append(f"{path}: {token}")
    assert not offenders, offenders


def test_experiments_construct_redisca_only_via_factory():
    root = Path(__file__).resolve().parents[1]
    allowed = {root / "common" / "method.py"}
    offenders = []
    for path in _runtime_modules(root):
        if path in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if "ReDisCA(" in text:
            offenders.append(str(path))
    assert not offenders, offenders


def test_oracle_module_does_not_import_redisca_estimator():
    source = inspect.getsource(independent_airi_spoc)
    assert "ReDisCA" not in source
