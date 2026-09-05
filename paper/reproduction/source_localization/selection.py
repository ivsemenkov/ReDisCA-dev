"""Fig. 17 → Fig. 18 component selection (paper rule, not p<0.05)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def select_fig17_lowest_p(p_values: NDArray[np.floating], *, n: int = 3) -> NDArray[np.intp]:
    """Return indices of the ``n`` components with the lowest p-values.

    Paper Fig. 17: “The three components with the lowest p-values are shown.”
    This is not a p<0.05 cutoff and is not “the first three eigenvalues.”
    Ties are broken by smaller component index (stable mergesort).
    """
    p = np.asarray(p_values, dtype=np.float64).ravel()
    if p.size == 0:
        raise ValueError("p_values must be non-empty")
    take = min(int(n), int(p.size))
    order = np.argsort(p, kind="mergesort")
    return order[:take].astype(np.intp)
