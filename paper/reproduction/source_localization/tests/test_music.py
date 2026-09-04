"""Eq. 13 / Eq. 14 scanner tests, including the AIRI P=eye(1) bug."""

from __future__ import annotations

import numpy as np
import pytest

from source_localization.music import (
    AiriMusicDimensionError,
    airi_initial_projector,
    airi_music_scan,
    cosine_similarity_scan,
    dipole_tangent_bases,
    music_scan,
    subspace_correlation_scan,
)


def test_eq13_unit_vectors() -> None:
    g = np.eye(4)
    a = np.array([0.0, 2.0, 0.0, 0.0])
    rho = cosine_similarity_scan(g, a)
    np.testing.assert_allclose(rho, [0.0, 1.0, 0.0, 0.0])


def test_eq13_typesetting_slip_uses_a1_norm() -> None:
    g = np.array([[1.0, 0.0], [0.0, 1.0]])
    a_k = np.array([0.0, 4.0])
    a_1 = np.array([2.0, 0.0])
    rho_fix = cosine_similarity_scan(g, a_k, denominator="a_k")
    rho_slip = cosine_similarity_scan(
        g, a_k, denominator="a_1_typesetting_slip", a_1=a_1
    )
    np.testing.assert_allclose(rho_fix, [0.0, 1.0])
    # slip divides by ||a_1||=2 instead of ||a_k||=4, so the match is 4/2=2 (not a cosine)
    np.testing.assert_allclose(rho_slip, [0.0, 2.0])


def test_music_self_dipole_is_one() -> None:
    rng = np.random.default_rng(0)
    n_sns, n_src = 30, 12
    blocks = rng.normal(size=(n_sns, n_src, 3))
    bases = dipole_tangent_bases(blocks)
    assert bases.shape == (n_src, n_sns, 2)
    true = 4
    # Pattern in the free-orientation span of vertex `true`
    coeff = np.array([0.6, -0.8])
    pattern = bases[true] @ coeff
    scan = music_scan(blocks, pattern)
    assert scan.shape == (n_src,)
    assert int(np.argmax(scan)) == true
    assert scan[true] == pytest.approx(1.0, abs=1e-10)


def test_music_k1_matches_subspace_cosine() -> None:
    rng = np.random.default_rng(1)
    blocks = rng.normal(size=(16, 7, 3))
    bases = dipole_tangent_bases(blocks)
    pattern = rng.normal(size=16)
    pattern /= np.linalg.norm(pattern)
    scan = subspace_correlation_scan(bases, pattern)
    # Manual: ||U_{2d}^T a|| for orthonormal U
    manual = np.empty(7)
    for m in range(7):
        u = bases[m]
        manual[m] = float(np.linalg.norm(u.T @ pattern))
    np.testing.assert_allclose(scan, np.clip(manual, 0.0, 1.0), atol=1e-12)


def test_music_subspace_contains_two_dipoles() -> None:
    rng = np.random.default_rng(2)
    blocks = rng.normal(size=(24, 9, 3))
    bases = dipole_tangent_bases(blocks)
    a_k = np.column_stack([bases[1] @ [1.0, 0.0], bases[5] @ [0.2, 0.9]])
    scan = music_scan(blocks, a_k)
    # Both generating vertices should be near 1
    assert scan[1] == pytest.approx(1.0, abs=1e-9)
    assert scan[5] == pytest.approx(1.0, abs=1e-9)


def test_airi_literal_projector_is_eye1() -> None:
    P = airi_initial_projector(204, variant="literal_bug")
    assert P.shape == (1, 1)
    Pfix = airi_initial_projector(204, variant="eye_nsns_fix")
    assert Pfix.shape == (204, 204)
    np.testing.assert_array_equal(Pfix, np.eye(204))


def test_airi_music_literal_bug_raises() -> None:
    rng = np.random.default_rng(3)
    G = rng.normal(size=(20, 15))  # 5 sources × 3
    topo = rng.normal(size=20)
    with pytest.raises(AiriMusicDimensionError):
        airi_music_scan(G, topo, n_rap=1, projector_variant="literal_bug")


def test_airi_music_fix_nrap1_matches_eq14() -> None:
    rng = np.random.default_rng(4)
    n_sns, n_src = 18, 6
    blocks = rng.normal(size=(n_sns, n_src, 3))
    G = blocks.reshape(n_sns, n_src * 3)
    topos = rng.normal(size=(n_sns, 2))
    ctx = airi_music_scan(G, topos, n_rap=1, projector_variant="eye_nsns_fix")
    eq14 = music_scan(blocks, topos)
    np.testing.assert_allclose(ctx[:, 0], eq14, atol=1e-12)
