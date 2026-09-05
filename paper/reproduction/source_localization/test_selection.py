"""Fig. 17 lowest-p selection is not a p<0.05 cutoff."""

from __future__ import annotations

import numpy as np

from paper.reproduction.source_localization.selection import select_fig17_lowest_p


def test_lowest_p_takes_three_even_if_only_two_are_significant():
    p = np.array([0.007, 0.021, 0.093, 0.197, 0.35])
    chosen = select_fig17_lowest_p(p, n=3)
    assert list(chosen) == [0, 1, 2]
    assert int(np.sum(p[chosen] < 0.05)) == 2


def test_lowest_p_is_not_eigenvalue_order_when_p_is_permuted():
    p = np.array([0.20, 0.01, 0.40, 0.03, 0.02])
    chosen = select_fig17_lowest_p(p, n=3)
    assert list(chosen) == [1, 4, 3]


def test_p05_prefix_would_drop_the_third_component():
    p = np.array([0.007, 0.021, 0.093, 0.197])
    p05 = np.flatnonzero(p < 0.05)
    assert list(p05) == [0, 1]
    assert list(select_fig17_lowest_p(p, n=3)) == [0, 1, 2]


def test_paper1501_facevstool_pvalues_select_three_even_when_two_are_significant():
    from paper.reproduction.source_localization.run import _choose_fig18_components

    p = np.array([0.007, 0.021, 0.093, 0.197, 0.352])
    chosen, note = _choose_fig18_components(p, rank=5, selection="lowest_p")
    assert list(chosen) == [0, 1, 2]
    assert "lowest p-values" in note
    p05, _ = _choose_fig18_components(p, rank=5, selection="p05")
    assert list(p05) == [0, 1]
