"""Component-index helper tests."""

from __future__ import annotations

import numpy as np

from source_localization.meg_patterns import three_lowest_p_indices


def test_three_lowest_p_ties_break_on_abs_lambda() -> None:
    evals = np.array([0.1, 5.0, 4.0, 0.2])
    p = np.array([0.04, 0.01, 0.01, 0.9])
    idx = three_lowest_p_indices(evals, p, k=3)
    # p=0.01 for comps 1 and 2; |λ| 5 > 4 so 1 then 2, then p=0.04 → 0
    np.testing.assert_array_equal(idx, np.array([1, 2, 0]))


def test_three_lowest_p_without_pvalues_takes_leading() -> None:
    evals = np.array([3.0, 2.0, 1.0, 0.1])
    idx = three_lowest_p_indices(evals, None, k=3)
    np.testing.assert_array_equal(idx, np.array([0, 1, 2]))
