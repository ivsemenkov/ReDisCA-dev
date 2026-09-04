"""Path-separated MEG figures. Never mix paper_faithful and airi_executable."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from .rdms import CONDITION_NAMES

CONDITION_COLORS = {
    "face1": "#1f77b4",
    "face2": "#9ecae1",
    "tool1": "#d62728",
    "tool2": "#fcbba1",
    "nons1": "#636363",
    "nons2": "#bdbdbd",
}


def plot_rdm(
    rdm: NDArray[np.floating],
    path: Path,
    *,
    title: str,
    path_label: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    rdm = np.asarray(rdm, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(rdm, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(6), CONDITION_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(6), CONDITION_NAMES)
    ax.set_title(f"{path_label}: {title}", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("condition")
    ax.set_ylabel("condition")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_component_panel(
    path: Path,
    *,
    path_label: str,
    rdm_name: str,
    time_ms: NDArray[np.floating],
    traces: NDArray[np.floating],
    patterns: NDArray[np.floating],
    eigenvalues: NDArray[np.floating],
    p_values: NDArray[np.floating] | None,
    asterisk_hi: NDArray[np.bool_] | None,
    asterisk_lo: NDArray[np.bool_] | None,
    n_plot: int,
    extra_title: str = "",
) -> None:
    """One figure per path/RDM: rows = components, left traces, right pattern.

    ``traces``: (n_conditions, n_comp, n_times)
    ``patterns``: (n_comp, n_channels) row-oriented Haufe patterns.
    """
    n_plot = min(n_plot, traces.shape[1], patterns.shape[0])
    fig, axes = plt.subplots(n_plot, 2, figsize=(11.0, 3.2 * n_plot), squeeze=False)
    t = np.asarray(time_ms, dtype=np.float64)
    for k in range(n_plot):
        ax_t = axes[k, 0]
        ax_p = axes[k, 1]
        for c, name in enumerate(CONDITION_NAMES):
            ax_t.plot(t, traces[c, k], color=CONDITION_COLORS[name], lw=1.4, label=name)
        y = traces[:, k, :]
        y_lo, y_hi = float(np.min(y)), float(np.max(y))
        span = y_hi - y_lo if y_hi > y_lo else 1.0
        if asterisk_hi is not None and k < asterisk_hi.shape[0]:
            idx = np.flatnonzero(asterisk_hi[k])
            if idx.size:
                ax_t.plot(t[idx], np.full(idx.size, y_hi + 0.08 * span), "r.", ms=3)
        if asterisk_lo is not None and k < asterisk_lo.shape[0]:
            idx = np.flatnonzero(asterisk_lo[k])
            if idx.size:
                ax_t.plot(t[idx], np.full(idx.size, y_lo - 0.08 * span), "b.", ms=3)
        ptxt = ""
        if p_values is not None and k < p_values.size:
            ptxt = f", p={p_values[k]:.4g}"
        lam = eigenvalues[k] if k < eigenvalues.size else float("nan")
        ax_t.set_title(f"{rdm_name} comp{k + 1}  λ={lam:.4g}{ptxt}", fontsize=10)
        ax_t.set_xlabel("time (ms)")
        ax_t.axvline(0.0, color="k", lw=0.6, ls="--")
        ax_t.grid(True, alpha=0.3)
        if k == 0:
            ax_t.legend(loc="upper right", fontsize=7, ncol=3)
        _imshow_pattern(ax_p, patterns[k])
        ax_p.set_title("Haufe pattern (204 planars as 12×17)", fontsize=9)
    fig.suptitle(f"{path_label}  {rdm_name}  {extra_title}".strip(), fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_planar_rms_row(
    path: Path,
    *,
    path_label: str,
    rdm_name: str,
    patterns: NDArray[np.floating],
    n_plot: int,
) -> None:
    """AIRI-style planar-pair RMS: sqrt(odd^2+even^2) as 6×17 (102 sensors)."""
    n_plot = min(n_plot, patterns.shape[0])
    fig, axes = plt.subplots(1, n_plot, figsize=(3.2 * n_plot, 3.4), squeeze=False)
    for k in range(n_plot):
        topo = np.asarray(patterns[k], dtype=np.float64)
        rms = np.hypot(topo[0::2], topo[1::2])
        grid = np.full(6 * 17, np.nan)
        grid[: rms.size] = rms
        axes[0, k].imshow(grid.reshape(6, 17), cmap="magma")
        axes[0, k].set_title(f"comp{k + 1} planar RMS", fontsize=9)
        axes[0, k].axis("off")
    fig.suptitle(f"{path_label} {rdm_name}: planar-pair RMS (not FieldTrip helmet)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _imshow_pattern(ax, pattern: NDArray[np.floating]) -> None:
    vec = np.asarray(pattern, dtype=np.float64)
    grid = np.full(12 * 17, np.nan)
    n = min(vec.size, grid.size)
    grid[:n] = vec[:n]
    lim = np.nanmax(np.abs(grid)) or 1.0
    ax.imshow(grid.reshape(12, 17), cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.axis("off")
