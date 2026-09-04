"""Tests for sign alignment and subspace comparison helpers."""

from __future__ import annotations

import numpy as np

from common.metrics import sign_align_vectors, subspace_similarity


def test_sign_align_flips_opposite_rows() -> None:
    reference = np.array([[1.0, 0.0], [0.0, 1.0]])
    estimated = np.array([[-1.0, 0.0], [0.0, 1.0]])
    aligned = sign_align_vectors(reference, estimated)
    np.testing.assert_allclose(aligned, reference)


def test_subspace_similarity_is_sign_invariant() -> None:
    basis = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    flipped = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    metrics = subspace_similarity(basis, flipped)
    assert metrics["min_cosine"] > 0.999
    assert metrics["max_angle_rad"] < 1e-6
