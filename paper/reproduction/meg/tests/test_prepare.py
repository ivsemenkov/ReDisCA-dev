"""MEG prepare tests. Index counts use the small SPM labels file."""

from __future__ import annotations

import numpy as np
import pytest

from common.meg_io import airi_condition_indices, load_spm_trial_labels
from common.paths import MEG_DIR
from meg.prepare import AIRI_SLICE, airi_time_ms, time_vector_ms
from meg.rdms import CONDITION_NAMES

SPM_MAT = MEG_DIR / "ibfctfprespm8_AD_run1_raw_tsss_mc.mat"


def test_true_time_axis_1501_samples() -> None:
    t = time_vector_ms()
    assert t.size == 1501
    assert t[0] == pytest.approx(-500.0)
    assert t[500] == pytest.approx(0.0)
    assert t[-1] == pytest.approx(1000.0)


def test_airi_trange_is_99_to_999_ms() -> None:
    t = airi_time_ms()
    assert t.size == 901
    assert t[0] == pytest.approx(99.0)
    assert t[-1] == pytest.approx(999.0)
    full = time_vector_ms()
    np.testing.assert_allclose(t, full[AIRI_SLICE])


@pytest.mark.skipif(not SPM_MAT.exists(), reason="SPM labels cache missing")
def test_airi_condition_indices_80x6() -> None:
    labels = load_spm_trial_labels(SPM_MAT)
    assert len(labels) == 880
    indices = airi_condition_indices(labels)
    assert tuple(indices) == CONDITION_NAMES
    for name, idx in indices.items():
        assert idx.size == 80, name
    used = np.concatenate([indices[name] for name in CONDITION_NAMES])
    assert used.size == 480
    assert np.unique(used).size == 480
