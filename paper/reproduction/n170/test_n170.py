"""Unit tests for the N170 track (RDMs, channels, library fit)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "paper" / "reproduction"))
sys.path.insert(0, str(HERE))

from rdms import (  # noqa: E402
    car_rdm,
    face_rdm,
    meaning_rdm,
    zscored_unique_pairs,
)
from prepare import (  # noqa: E402
    SCALP_LABELS,
    default_erp_path,
    load_n170_subject1,
    window_slice,
)
from inference import (  # noqa: E402
    assert_core_matches_estimator,
    exact_condition_label_null,
    unique_condition_relabelings,
)

from redisca import ReDisCA  # noqa: E402


def test_meaning_rdm_structure() -> None:
    d = meaning_rdm()
    assert d.shape == (4, 4)
    assert d[0, 1] == 0.0 and d[2, 3] == 0.0
    assert d[0, 2] == 1.0 and d[1, 3] == 1.0
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)


def test_face_and_car_detectors() -> None:
    face = face_rdm()
    car = car_rdm()
    assert np.allclose(face[0, 1:], 1.0)
    assert np.allclose(face[1:, 1:], np.zeros((3, 3)))
    assert np.allclose(car[1, [0, 2, 3]], 1.0)
    assert car[0, 2] == 0.0 and car[2, 3] == 0.0


def test_within_0_1_zscore_equivalent() -> None:
    for builder in (meaning_rdm, face_rdm, car_rdm):
        z0 = zscored_unique_pairs(builder(within=0.0))
        z1 = zscored_unique_pairs(builder(within=0.1))
        assert np.allclose(z0, z1, atol=1e-12)


def test_unique_relabeling_counts() -> None:
    meaning_u = unique_condition_relabelings(meaning_rdm())
    face_u = unique_condition_relabelings(face_rdm())
    assert len(meaning_u) == 3
    assert sum(item["multiplicity"] for item in meaning_u) == 24
    assert len(face_u) == 4
    assert sum(item["multiplicity"] for item in face_u) == 24


@pytest.mark.skipif(not default_erp_path().exists(), reason="ERP CORE cache absent")
def test_subject1_channel_subset() -> None:
    packed = load_n170_subject1()
    assert packed["data"].shape[0] == 4
    assert packed["data"].shape[1] == 28
    assert packed["data"].shape[2] == 256
    assert packed["srate_hz"] == 256.0
    assert packed["channel_labels"] == list(SCALP_LABELS)
    dropped = packed["channel_selection"]["dropped_labels"]
    assert "(corr) HEOG" in dropped and "HEOG_left" in dropped
    assert packed["n_accepted"] == [52, 38, 49, 52]
    assert packed["ica"]["erpcore_subject_1_components_1based"] == [2, 7]
    assert packed["ica"]["applied_in_this_script"] is False
    names = " ".join(packed["bin_descriptions"]).lower()
    assert "face" in names and "car" in names and "scrambled" in names


@pytest.mark.skipif(not default_erp_path().exists(), reason="ERP CORE cache absent")
def test_redisca_matches_inference_layer() -> None:
    packed = load_n170_subject1()
    win = window_slice(
        packed["data"], packed["times_ms"], center_ms=200.0, duration_ms=100.0
    )
    rdm = face_rdm()
    assert_core_matches_estimator(win["data"], rdm, demean_time=False)
    model = ReDisCA(demean_time=False).fit(win["data"], rdm)
    exact = exact_condition_label_null(
        win["data"], rdm, model.eigenvalues_[:4], demean_time=False
    )
    assert exact["n_permutations"] == 24
    assert exact["n_unique_rdms"] == 4
    assert 0.0 <= exact["p_greater_equal"][0] <= 1.0
