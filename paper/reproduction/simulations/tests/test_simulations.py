"""Unit tests for the simulations reconstruction (no published-number tuning)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

SIM_DIR = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parents[4] / "src"
for path in (SIM_DIR, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from generate import (  # noqa: E402
    add_symmetric_rdm_noise,
    fft_pink_noise,
    filter_gaussian_rows,
    mix_signal_and_noise,
    sample_separated_vertices,
    scale_noise_to_snr,
    source_erp,
    squared_euclidean_rdm,
)
from metrics_roc import cosine_abs_scan, roc_from_mc, sphere_mask  # noqa: E402
from rsa_baselines import pearson_upper, spotlight_scores  # noqa: E402


def test_rdm_symmetric_zero_diag_nonneg() -> None:
    rng = np.random.default_rng(0)
    series = rng.standard_normal((5, 40))
    rdm = squared_euclidean_rdm(series)
    np.testing.assert_allclose(rdm, rdm.T)
    np.testing.assert_allclose(np.diag(rdm), 0.0)
    assert np.all(rdm >= -1e-12)
    # matches explicit pairwise formula
    for i in range(5):
        for j in range(5):
            expected = float(np.sum((series[i] - series[j]) ** 2))
            np.testing.assert_allclose(rdm[i, j], expected, rtol=1e-12, atol=1e-12)


def test_upsilon_d_is_symmetric_zero_diag() -> None:
    rng = np.random.default_rng(1)
    d0 = squared_euclidean_rdm(rng.standard_normal((6, 30)))
    noisy = add_symmetric_rdm_noise(d0, rng, relative_std=0.05)
    np.testing.assert_allclose(noisy, noisy.T)
    np.testing.assert_allclose(np.diag(noisy), 0.0)
    assert np.all(noisy >= 0.0)


def test_butterworth_attenuates_high_frequency() -> None:
    rng = np.random.default_rng(2)
    filtered = filter_gaussian_rows(rng, 4, 200, fs_hz=1000.0, cutoff_hz=2.0, order=6)
    spec = np.abs(np.fft.rfft(filtered, axis=-1)).mean(axis=0)
    freqs = np.fft.rfftfreq(200, d=0.001)
    low = spec[freqs <= 2.0].mean()
    high = spec[freqs >= 20.0].mean()
    assert low > high


def test_source_erp_mixing_shapes() -> None:
    rng = np.random.default_rng(3)
    mixing = rng.standard_normal((5, 5))
    s, z = source_erp(mixing, rng, 200, fs_hz=1000.0, cutoff_hz=2.0, order=6)
    assert s.shape == (5, 200)
    assert z.shape == (5, 200)
    np.testing.assert_allclose(s, mixing @ z)


def test_pink_noise_unit_rms_and_red_spectrum() -> None:
    rng = np.random.default_rng(4)
    pink = fft_pink_noise(rng, 32, 256, fs_hz=1000.0, exponent=1.0)
    rms = np.sqrt(np.mean(pink**2, axis=-1))
    np.testing.assert_allclose(rms, 1.0, atol=1e-10)
    spec = np.abs(np.fft.rfft(pink, axis=-1)).mean(axis=0)
    freqs = np.fft.rfftfreq(256, d=0.001)
    band_lo = spec[(freqs >= 5) & (freqs < 15)].mean()
    band_hi = spec[(freqs >= 80) & (freqs < 120)].mean()
    assert band_lo > band_hi


def test_snr_scaling() -> None:
    rng = np.random.default_rng(5)
    signal = rng.standard_normal((5, 8, 20))
    noise = rng.standard_normal((5, 8, 20))
    scaled, gamma = scale_noise_to_snr(signal, noise, 0.1)
    sig_rms = np.sqrt(np.mean(signal**2))
    n_rms = np.sqrt(np.mean(scaled**2))
    np.testing.assert_allclose(sig_rms / n_rms, 0.1, rtol=1e-10)
    np.testing.assert_allclose(scaled, gamma * noise)


def test_mix_snr_per_trial() -> None:
    rng = np.random.default_rng(6)
    signal = rng.standard_normal((4, 6, 10))
    noise = rng.standard_normal((7, 4, 6, 10))
    trials, gammas = mix_signal_and_noise(signal, noise, 0.2)
    assert trials.shape == noise.shape
    assert gammas.shape == (7,)
    for i in range(7):
        n = trials[i] - signal
        np.testing.assert_allclose(
            np.sqrt(np.mean(signal**2)) / np.sqrt(np.mean(n**2)), 0.2, rtol=1e-10
        )


def test_separated_vertices() -> None:
    rng = np.random.default_rng(7)
    vertices = rng.normal(size=(200, 3))
    idx = sample_separated_vertices(rng, vertices, 4, 0.5)
    assert idx.size == 4
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.linalg.norm(vertices[idx[i]] - vertices[idx[j]]) >= 0.5


def test_roc_perfect_detection() -> None:
    n_mc, n_v = 5, 20
    scores = np.zeros((n_mc, n_v))
    inside = np.zeros((n_mc, n_v), dtype=bool)
    inside[:, 0] = True
    scores[:, 0] = 1.0
    roc = roc_from_mc(scores, inside, np.array([1.5, 0.5, -0.5]))
    np.testing.assert_allclose(roc["tpr"], [0.0, 1.0, 1.0])
    np.testing.assert_allclose(roc["fpr"], [0.0, 0.0, 1.0])
    assert roc["auc"] == pytest.approx(1.0)


def test_sphere_mask() -> None:
    dist = np.array([0.0, 0.009, 0.01, 0.011])
    mask = sphere_mask(dist, 0.01)
    np.testing.assert_array_equal(mask, [True, True, True, False])


def test_cosine_abs_sign_invariant() -> None:
    rng = np.random.default_rng(8)
    gain = rng.standard_normal((10, 30))
    pattern = gain[:, 3]
    a = cosine_abs_scan(pattern, gain)
    b = cosine_abs_scan(-pattern, gain)
    np.testing.assert_allclose(a, b)
    assert int(np.argmax(a)) == 3


def test_spotlight_pearson_matches_corrcoef() -> None:
    rng = np.random.default_rng(9)
    c, m = 5, 12
    rdms = np.zeros((c, c, m))
    for v in range(m):
        s = rng.standard_normal((c, 8))
        d = squared_euclidean_rdm(s)
        rdms[:, :, v] = d
    target = rdms[:, :, 0]
    scores = spotlight_scores(rdms, target)
    for v in range(m):
        np.testing.assert_allclose(scores[v], pearson_upper(rdms[:, :, v], target), atol=1e-12)
    assert scores[0] == pytest.approx(1.0)
