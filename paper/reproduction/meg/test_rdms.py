"""MEG RDM construction tests."""

from __future__ import annotations

import numpy as np

from paper.reproduction.meg.rdms import airi_rdm, class_labels, theoretical_rdm


def test_airi_facevstool_matches_matlab_literals():
    rdm = airi_rdm("facevstool")
    assert rdm[0, 1] == 0.1
    assert rdm[0, 2] == 1.0
    assert rdm[0, 4] == 0.5
    assert rdm[4, 5] == 0.1
    assert np.allclose(rdm, rdm.T)
    assert np.allclose(np.diag(rdm), 0.0)


def test_binary_face_maps_within_to_zero():
    binary = theoretical_rdm("face", fill="binary")
    assert binary[0, 1] == 0.0
    assert binary[0, 2] == 1.0


def test_airi_face_contrast_is_faces_vs_nons():
    class1, class2 = class_labels("face", convention="airi")
    assert class1 == (0, 1)
    assert class2 == (4, 5)
