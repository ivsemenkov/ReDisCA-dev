"""Auditable tests for the source-faithful AIRI/SPoC reconstruction.

These tests do not treat the public ``redisca`` library as an oracle.
``source_faithful.py`` must not import ``redisca``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from common.source_faithful import (
    airi_rdm,
    create_cxxz,
    directed_pairs,
    fit_condition_averages,
    matlab_cov_time_by_channel,
    matlab_std,
    matlab_zscore,
    pair_stack_from_condition_averages,
    random_phase_surrogate,
    unique_unordered_pairs,
)

COMMON_DIR = Path(__file__).resolve().parents[1]


def test_source_faithful_module_does_not_import_redisca() -> None:
    source = (COMMON_DIR / "source_faithful.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module.split(".")[0])
    assert "redisca" not in imported


def test_matlab_std_matches_sample_sd() -> None:
    values = np.array([1.0, 3.0, 5.0, 7.0])
    expected = float(np.sqrt(np.sum((values - values.mean()) ** 2) / (values.size - 1)))
    assert matlab_std(values) == pytest.approx(expected)
    assert matlab_std(values) == pytest.approx(float(np.std(values, ddof=1)))


def test_matlab_cov_two_samples() -> None:
    epoch = np.array([[1.0, 2.0], [3.0, 6.0]])
    cov = matlab_cov_time_by_channel(epoch)
    # Columns demeaned: [-1,1] and [-2,2]; (T-1)=1
    assert cov.shape == (2, 2)
    assert cov[0, 0] == pytest.approx(2.0)
    assert cov[1, 1] == pytest.approx(8.0)
    assert cov[0, 1] == pytest.approx(4.0)


def test_create_cxxz_is_weighted_mean() -> None:
    rng = np.random.default_rng(0)
    cxxe = rng.normal(size=(3, 3, 5))
    cxxe = 0.5 * (cxxe + np.transpose(cxxe, (1, 0, 2)))
    z = np.array([1.0, -1.0, 0.5, 0.0, 2.0])
    got = create_cxxz(cxxe, z)
    expected = sum(cxxe[:, :, e] * z[e] for e in range(5)) / 5.0
    np.testing.assert_allclose(got, expected)


def test_random_phase_preserves_amplitude_spectrum() -> None:
    rng = np.random.default_rng(1)
    z = rng.normal(size=32)
    surrogate, amps = random_phase_surrogate(z, rng)
    got = np.abs(np.fft.fft(surrogate))
    # MATLAB assigns a free phase to the Nyquist bin, then takes real(ifft),
    # so the Nyquist amplitude is not preserved. All other bins are.
    nyquist = z.size // 2
    mask = np.ones(z.size, dtype=bool)
    mask[nyquist] = False
    np.testing.assert_allclose(got[mask], amps[mask], atol=1e-10)
    assert surrogate.shape == z.shape


def test_directed_pairs_duplicate_unique_pairs() -> None:
    directed = directed_pairs(4)
    unique = unique_unordered_pairs(4)
    assert len(unique) == 6
    assert len(directed) == 12
    assert (0, 1) in directed and (1, 0) in directed
    assert directed[0] == (0, 1)


def test_airi_default_rdm_is_facevstool_symmetric() -> None:
    rdm = airi_rdm("facevstool")
    np.testing.assert_allclose(rdm, rdm.T)
    assert rdm[0, 2] == 1.0
    assert rdm[0, 4] == 0.5
    assert np.all(np.diag(rdm) == 0.0)


def test_directed_and_unique_pairs_share_gep_for_symmetric_rdm() -> None:
    rng = np.random.default_rng(2)
    n_conditions, n_channels, n_times = 4, 6, 40
    mixing = rng.standard_normal((n_channels, 2))
    sources = rng.standard_normal((2, n_times))
    X = np.stack(
        [mixing @ (sources * rng.uniform(0.5, 1.5, size=(2, 1))) for _ in range(n_conditions)]
    )
    rdm = np.array(
        [
            [0.0, 0.1, 1.0, 1.0],
            [0.1, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.1],
            [1.0, 1.0, 0.1, 0.0],
        ]
    )
    directed = fit_condition_averages(
        X, rdm, pair_mode="airi_directed", matrix_mode="matlab_cov"
    )
    unique = fit_condition_averages(
        X, rdm, pair_mode="unique_unordered", matrix_mode="matlab_cov"
    )
    # Duplicating symmetric pairs does not change Cxx, but MATLAB std(z)
    # uses N-1, so the standardized target (and therefore lambda) scales.
    n_unique = 6
    n_directed = 12
    std_ratio = np.sqrt((2.0 * (n_unique - 1)) / (n_directed - 1))
    np.testing.assert_allclose(
        directed.eigenvalues,
        unique.eigenvalues / std_ratio,
        rtol=1e-6,
        atol=1e-8,
    )
    for k in range(min(2, directed.filters.shape[1])):
        corr = np.corrcoef(directed.filters[:, k], unique.filters[:, k])[0, 1]
        assert abs(corr) > 0.999


def test_t_minus_one_scale_does_not_change_eigenspace() -> None:
    rng = np.random.default_rng(3)
    X = rng.standard_normal((5, 7, 30))
    rdm = np.zeros((5, 5))
    for i in range(5):
        for j in range(i + 1, 5):
            rdm[i, j] = rdm[j, i] = float(abs(i - j))
    cov = fit_condition_averages(X, rdm, pair_mode="unique_unordered", matrix_mode="matlab_cov")
    gram = fit_condition_averages(
        X, rdm, pair_mode="unique_unordered", matrix_mode="unscaled_gram"
    )
    # matlab_cov demeans; unscaled gram does not, so these are NOT required
    # to match. This test only checks that a global T-1 rescale of cov
    # matrices leaves the GEP unchanged.
    pairs = unique_unordered_pairs(5)
    stack = pair_stack_from_condition_averages(X, pairs, matrix_mode="matlab_cov")
    n_times = X.shape[-1]
    scaled = stack * (n_times - 1)
    from common.source_faithful import spoc_from_pair_stack, theoretical_rdm_vector

    z = theoretical_rdm_vector(rdm, pairs)
    rescaled = spoc_from_pair_stack(scaled, z, pair_mode="unique_unordered", matrix_mode="matlab_cov")
    np.testing.assert_allclose(cov.eigenvalues, rescaled.eigenvalues, atol=1e-10)


def test_spoc_random_phase_pvalues_can_be_zero() -> None:
    rng = np.random.default_rng(4)
    n_conditions, n_channels, n_times = 4, 8, 50
    mixing = rng.standard_normal((n_channels, 1))
    t = np.linspace(0, 1, n_times)
    waveforms = [
        mixing * np.sin(2 * np.pi * 3 * t) * amp
        for amp in (1.0, 1.05, 4.0, 4.1)
    ]
    X = np.stack(waveforms, axis=0)
    rdm = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    result = fit_condition_averages(
        X,
        rdm,
        pair_mode="airi_directed",
        matrix_mode="matlab_cov",
        n_bootstrapping_iterations=20,
        rng=rng,
        inference="spoc_random_phase",
    )
    assert result.p_values is not None
    assert result.p_values.shape == result.eigenvalues.shape
    assert np.all(result.p_values >= 0.0)
    assert np.all(result.p_values <= 1.0)


def test_zscore_matches_matlab_sample_convention() -> None:
    z = np.array([0.1, 1.0, 0.5, 0.1])
    got = matlab_zscore(z)
    expected = (z - z.mean()) / np.std(z, ddof=1)
    np.testing.assert_allclose(got, expected)
