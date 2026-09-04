"""AIRI constrained sLORETA precomp tests."""

from __future__ import annotations

import numpy as np

from source_localization.sloreta import precomp_abs_kernel_map


def test_precomp_is_abs_kernel_times_topo() -> None:
    rng = np.random.default_rng(0)
    kernel = rng.normal(size=(40, 12))
    cols = np.array([0, 2, 5, 7])
    topo = rng.normal(size=4)
    got = precomp_abs_kernel_map(kernel, topo, cols)
    expected = np.abs(kernel[:, cols] @ topo)
    np.testing.assert_allclose(got, expected)
    assert got.shape == (40,)
