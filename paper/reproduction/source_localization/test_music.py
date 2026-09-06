"""MUSIC scanner unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from paper.reproduction.source_localization.music import (
    AiriMusicDimensionError,
    airi_music_scan,
    cosine_similarity_scan,
    music_scan,
)


def test_music_peaks_at_matching_dipole():
    rng = np.random.default_rng(0)
    blocks = rng.standard_normal((20, 7, 3))
    target = blocks[:, 3, :]
    u, _s, _vt = np.linalg.svd(target, full_matrices=False)
    patterns = u[:, :2]
    scan = music_scan(blocks, patterns)
    assert int(np.argmax(scan)) == 3
    assert scan[3] > 0.9


def test_literal_airi_projector_is_non_executable():
    G = np.ones((6, 9))
    topos = np.ones((6, 1))
    with pytest.raises(AiriMusicDimensionError):
        airi_music_scan(G, topos, projector_variant="literal_bug")


def test_eq13_cosine():
    lead = np.eye(3)
    scan = cosine_similarity_scan(lead, np.array([1.0, 0.0, 0.0]))
    assert scan[0] == 1.0
