"""Simple 3-view cortical scatter. Not FieldTrip / ``show_on_cortex`` parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray


def save_cortex_scatter(
    path: Path,
    vertices: NDArray[np.floating],
    scan: NDArray[np.floating],
    *,
    title: str,
    peak_index: int | None = None,
) -> Path:
    """Write a matplotlib 3-panel scatter of mesh Vertices colored by the scan.

    Axes are native Brainstorm ``GridLoc`` / ``Vertices`` metres. They are not
    the paper’s A/P/S/L/R FieldTrip camera. ``show_on_cortex`` is not in the
    AIRI repo.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vertices = np.asarray(vertices, dtype=np.float64)
    scan = np.asarray(scan, dtype=np.float64).ravel()
    if vertices.shape[0] != scan.size:
        raise ValueError("vertices and scan length differ")
    finite = np.isfinite(scan)
    vmin = float(np.min(scan[finite])) if finite.any() else 0.0
    vmax = float(np.max(scan[finite])) if finite.any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-12

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), constrained_layout=True)
    views = (
        (0, 1, "mesh x (m)", "mesh y (m)", "xy"),
        (0, 2, "mesh x (m)", "mesh z (m)", "xz"),
        (1, 2, "mesh y (m)", "mesh z (m)", "yz"),
    )
    for ax, (i, j, xlab, ylab, _name) in zip(axes, views):
        pts = ax.scatter(
            vertices[:, i],
            vertices[:, j],
            c=scan,
            s=4,
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
            rasterized=True,
        )
        if peak_index is not None:
            ax.scatter(
                vertices[peak_index, i],
                vertices[peak_index, j],
                s=36,
                facecolors="none",
                edgecolors="cyan",
                linewidths=1.0,
            )
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(_name)
    fig.colorbar(pts, ax=axes.ravel().tolist(), shrink=0.82, label="scan")
    fig.suptitle(title, fontsize=10)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
