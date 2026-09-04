"""RDM catalog tests (no MEG data required)."""

from __future__ import annotations

import numpy as np
import pytest

from meg.inference import empirical_rdm_from_traces, unique_pair_pearson
from meg.rdms import (
    AIRI_RDM_NAMES,
    CONDITION_NAMES,
    class_labels,
    rdm_catalog,
    theoretical_rdm,
)


def test_condition_order_matches_airi() -> None:
    assert CONDITION_NAMES == ("face1", "face2", "tool1", "tool2", "nons1", "nons2")


def test_airi_facevstool_is_nonbinary() -> None:
    rdm = theoretical_rdm("facevstool", fill="airi")
    assert rdm[0, 2] == 1.0
    assert rdm[0, 4] == 0.5
    assert rdm[0, 1] == 0.1
    assert np.all(np.diag(rdm) == 0.0)
    with pytest.raises(ValueError, match="no 0/1"):
        theoretical_rdm("facevstool", fill="binary")


def test_binary_face_tool_meaning_are_01() -> None:
    catalog = rdm_catalog()
    for name in ("face", "tool", "meaning"):
        binary = catalog["binary_0_1"][name]
        airi = catalog["airi_numeric"][name]
        assert set(np.unique(binary).tolist()).issubset({0.0, 1.0})
        assert np.allclose(binary[np.isclose(airi, 1.0)], 1.0)
        assert np.allclose(binary[np.isclose(airi, 0.1)], 0.0)


def test_paper_vs_airi_face_class_split() -> None:
    paper_c1, paper_c2 = class_labels("face", convention="paper")
    airi_c1, airi_c2 = class_labels("face", convention="airi")
    assert paper_c1 == (0, 1)
    assert paper_c2 == (2, 3, 4, 5)
    assert airi_c1 == (0, 1)
    assert airi_c2 == (4, 5)


def test_empirical_rdm_pearson_identity() -> None:
    rng = np.random.default_rng(0)
    traces = rng.normal(size=(6, 40))
    dhat = empirical_rdm_from_traces(traces, demean_time=False)
    assert unique_pair_pearson(dhat, dhat) == pytest.approx(1.0)
    assert AIRI_RDM_NAMES[0] == "face"
