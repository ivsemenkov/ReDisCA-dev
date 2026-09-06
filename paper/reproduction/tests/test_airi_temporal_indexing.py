"""Literal AIRI half-split indexing vs corrected pooled indexing."""

from __future__ import annotations

import numpy as np

from paper.reproduction.common.inference_secondary import airi_halfsplit_timecourse


def _toy():
    rng = np.random.default_rng(0)
    # 6 channels, 5 times, 20 trials. Real classes occupy 10–19.
    # Trials 0–9 are distractors that literal MATLAB indexing can hit
    # because it uses raw randperm slots instead of idxAll(rpm).
    data = rng.standard_normal((6, 5, 20)) * 0.01
    data[:, 2, 10:15] += 1.0
    data[:, 2, 15:20] -= 1.0
    data[:, :, :10] += 50.0
    filters = np.ones((2, 6))
    idx1 = np.array([10, 11, 12, 13, 14], dtype=np.intp)
    idx2 = np.array([15, 16, 17, 18, 19], dtype=np.intp)
    return data, idx1, idx2, filters


def test_literal_and_corrected_are_not_the_same():
    data, idx1, idx2, filters = _toy()
    lit = airi_halfsplit_timecourse(
        data, idx1, idx2, filters, nmc=20, rng=np.random.default_rng(1), indexing="literal"
    )
    cor = airi_halfsplit_timecourse(
        data, idx1, idx2, filters, nmc=20, rng=np.random.default_rng(1), indexing="corrected_pooled"
    )
    assert lit["indexing"] == "literal"
    assert cor["indexing"] == "corrected_pooled"
    assert not np.allclose(lit["pplus"], cor["pplus"])


def test_literal_uses_raw_permutation_slots_not_pooled_labels():
        # Corrected surrogates only see class trials 10–19. Literal uses
        # raw 0..n_pooled-1 and therefore averages the +50 distractors.
        data, idx1, idx2, filters = _toy()
        lit = airi_halfsplit_timecourse(
            data, idx1, idx2, filters, nmc=8, rng=np.random.default_rng(2), indexing="literal"
        )
        cor = airi_halfsplit_timecourse(
            data, idx1, idx2, filters, nmc=8, rng=np.random.default_rng(2), indexing="corrected_pooled"
        )
        assert np.linalg.norm(lit["observed_contrast_std"]) > 0.0
        assert np.allclose(lit["observed_contrast_std"], cor["observed_contrast_std"])
        assert not np.allclose(lit["pminus"], cor["pminus"])


def test_maxmin_over_time_is_used():
    data, idx1, idx2, filters = _toy()
    out = airi_halfsplit_timecourse(
        data, idx1, idx2, filters, nmc=5, rng=np.random.default_rng(3), indexing="literal"
    )
    assert out["pplus"].shape == (2, 5)
    assert out["pminus"].shape == (2, 5)
    assert np.all((out["pplus"] >= 0.0) & (out["pplus"] <= 1.0))
