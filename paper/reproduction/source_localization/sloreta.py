"""AIRI constrained sLORETA ``precomp`` path (not paper Fig. 18).

AIRI default::

    method = 'precomp'
    topos = A1(:,4)                 % MATLAB 1-based column 4
    W = io.ImagingKernel(:, megplanarbst)
    ctx_map = abs(W * topos)

``ImagingKernel`` is 5002 × 306, constrained (``nComponents=1``),
``Function='sloreta'``, comment ``sLORETA: MEG ALL(Constr)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def precomp_abs_kernel_map(
    imaging_kernel: NDArray[np.floating],
    topography: NDArray[np.floating],
    column_index_0based: NDArray[np.integer],
) -> NDArray[np.float64]:
    """``abs(ImagingKernel[:, cols] @ topo)`` on 5002 vertices.

    ``column_index_0based`` selects 204 columns of the 306-column kernel.
    AIRI uses ``megplanarbst`` (1-based) converted to 0-based here.
    """
    kernel = np.asarray(imaging_kernel, dtype=np.float64)
    topo = np.asarray(topography, dtype=np.float64).ravel()
    cols = np.asarray(column_index_0based, dtype=np.int64).ravel()
    if kernel.ndim != 2:
        raise ValueError(f"ImagingKernel must be 2-D, got {kernel.shape}")
    if cols.min() < 0 or cols.max() >= kernel.shape[1]:
        raise ValueError(
            f"column index out of range for kernel with {kernel.shape[1]} columns"
        )
    if topo.size != cols.size:
        raise ValueError(
            f"topography length {topo.size} != selected kernel columns {cols.size}"
        )
    weights = kernel[:, cols]
    return np.abs(weights @ topo)
