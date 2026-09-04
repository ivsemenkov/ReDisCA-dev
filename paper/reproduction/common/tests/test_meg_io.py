"""Tests for MEG OSF loaders. Skipped when the gitignored cache is absent."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.meg_io import airi_condition_indices, load_meg_ad_run1, load_spm_trial_labels
from common.paths import MEG_DIR

MEG_MAT = MEG_DIR / "MEG_AD_run1.mat"
SPM_MAT = MEG_DIR / "ibfctfprespm8_AD_run1_raw_tsss_mc.mat"


pytestmark = pytest.mark.skipif(
    not (MEG_MAT.exists() and SPM_MAT.exists()),
    reason="MEG OSF cache not present",
)


def test_airi_condition_counts_are_80() -> None:
    labels = load_spm_trial_labels(SPM_MAT)
    assert len(labels) == 880
    indices = airi_condition_indices(labels)
    assert list(indices) == ["face1", "face2", "tool1", "tool2", "nons1", "nons2"]
    for name, idx in indices.items():
        assert idx.size == 80, name


def test_meg_run1_shape() -> None:
    payload = load_meg_ad_run1(MEG_MAT)
    assert payload["data"].shape == (207, 1501, 880)
    assert payload["fs"] == 1000.0
    assert payload["time_onset_s"] == -0.5
