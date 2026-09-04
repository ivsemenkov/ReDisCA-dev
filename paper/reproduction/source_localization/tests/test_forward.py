"""Tests for Neuromag planar indexing and the AD forward-model loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from common.paths import SOURCE_MODEL_DIR
from source_localization.forward import (
    N_PLANAR,
    index_audit,
    leadfield_blocks,
    load_source_model,
    mag_plus_first_grad_if_mag_first_0based,
    mag_plus_grad1_on_grad_grad_mag_0based,
    matlab_megplanarbst_1based,
    megplanarbst_0based,
)

KERNEL = SOURCE_MODEL_DIR / "results_sLORETA_MEG_GRAD_MEG_MAG_KERNEL_150924_1824.mat"
HEAD = SOURCE_MODEL_DIR / "headmodel_surf_os_meg.mat"
TESS = SOURCE_MODEL_DIR / "tess_cortex_pial_low.mat"

pytestmark_assets = pytest.mark.skipif(
    not (KERNEL.exists() and HEAD.exists() and TESS.exists()),
    reason="source-model OSF cache not present",
)


def test_matlab_megplanarbst_matches_airi_colon_syntax() -> None:
    got = matlab_megplanarbst_1based()
    expected = np.sort(np.concatenate([np.arange(1, 305, 3), np.arange(2, 306, 3)]))
    np.testing.assert_array_equal(got, expected)
    assert got.size == N_PLANAR
    assert got[0] == 1
    assert got[-1] == 305
    np.testing.assert_array_equal(megplanarbst_0based(), got - 1)


def test_megplanarbst_is_not_fsaverage_constant() -> None:
    # Guard against silently swapping in a template mesh in the loader.
    source = Path(__file__).resolve().parents[1].joinpath("forward.py").read_text()
    assert "fsaverage" in source
    assert "Refusing to use an fsaverage" in source


@pytestmark_assets
def test_airi_index_is_204_grad_on_this_kernel() -> None:
    model = load_source_model(SOURCE_MODEL_DIR)
    audit = index_audit(model.channel_types)
    assert model.channel_types[:3] == ("MEG GRAD", "MEG GRAD", "MEG MAG")
    assert audit["airi_megplanarbst_is_both_planars_on_this_file"] is True
    assert audit["airi_megplanarbst_n_grad"] == 204
    assert audit["airi_megplanarbst_n_mag"] == 0
    mixed = audit["d15_this_file_grad1_plus_mag_counts"]
    assert mixed["MEG MAG"] == 102
    assert mixed["MEG GRAD"] == 102
    assert audit["d15_same_integers_as_megplanarbst_if_one_assumes_MAG_first"] is True
    np.testing.assert_array_equal(
        mag_plus_first_grad_if_mag_first_0based(), megplanarbst_0based()
    )
    assert np.array_equal(model.good_channel_1based, np.arange(1, 307))
    assert "fsaverage" not in model.surface_comment.lower()
    blocks = leadfield_blocks(model.gain, megplanarbst_0based())
    assert blocks.shape == (204, 5002, 3)
    assert np.isfinite(blocks).all()
    mixed_idx = mag_plus_grad1_on_grad_grad_mag_0based()
    mixed_types = [model.channel_types[i] for i in mixed_idx]
    assert mixed_types.count("MEG MAG") == 102
