"""Optional N170 figures (PNG under paper/results/n170/, gitignored)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

CONDITION_COLORS = {
    "Faces": "#1f77b4",
    "Cars": "#d62728",
    "Scrambled Faces": "#bcbd22",
    "Scrambled Cars": "#9467bd",
}


def _try_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    return plt


def _topo_xy(xyz: NDArray[np.floating]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """EEGLAB chanlocs: X=nose, Y=left ear → 2-D topo with nose up, right +."""
    xyz = np.asarray(xyz, dtype=np.float64)
    x = -xyz[:, 1]
    y = xyz[:, 0]
    return x, y


def _draw_topo(ax, xyz, values, labels, title: str) -> None:
    x, y = _topo_xy(xyz)
    values = np.asarray(values, dtype=np.float64)
    vmax = float(np.max(np.abs(values))) or 1.0
    ax.tricontourf(x, y, values, levels=16, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.scatter(x, y, c="k", s=8, zorder=3)
    for xi, yi, lab in zip(x, y, labels):
        if lab in {"O1", "Oz", "O2", "PO7", "PO8", "P7", "P8"}:
            ax.text(xi, yi, lab, fontsize=6, ha="center", va="bottom")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    ax.axis("off")


def _draw_rdm(ax, matrix, labels, title: str, vmin=None, vmax=None) -> None:
    matrix = np.asarray(matrix, dtype=np.float64)
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = float(np.max(matrix)) if np.max(matrix) > 0 else 1.0
    im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    short = [lab.replace("Scrambled ", "Scr.\n") for lab in labels]
    ax.set_xticklabels(short, fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_fig09(path: Path, face, car, meaning, labels: Sequence[str]) -> None:
    plt = _try_pyplot()
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    _draw_rdm(axes[0], meaning, labels, "Fig 7a meaning (0/1)", vmin=0, vmax=1)
    _draw_rdm(axes[1], face, labels, "Fig 9a face (0/1)", vmin=0, vmax=1)
    _draw_rdm(axes[2], car, labels, "Fig 9b car (0/1)", vmin=0, vmax=1)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_fig07(
    path: Path,
    *,
    meaning_rdm,
    labels,
    pmap: NDArray[np.floating],
    centers_ms: NDArray[np.floating],
    p_comp0: NDArray[np.floating],
    pattern,
    xyz,
    channel_labels,
    p_rule: str,
) -> None:
    plt = _try_pyplot()
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.0))
    _draw_rdm(axes[0, 0], meaning_rdm, labels, "a. Theoretical meaning RDM", vmin=0, vmax=1)
    im = axes[0, 1].imshow(
        np.asarray(pmap, dtype=np.float64),
        aspect="auto",
        origin="upper",
        cmap="magma_r",
        vmin=0.0,
        vmax=1.0,
        extent=[
            float(centers_ms[0]),
            float(centers_ms[-1]),
            np.asarray(pmap).shape[0] - 0.5,
            -0.5,
        ],
    )
    axes[0, 1].set_title(f"b. Component p-map ({p_rule})")
    axes[0, 1].set_xlabel("window center (ms)")
    axes[0, 1].set_ylabel("component (0 = leading)")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)
    _draw_topo(axes[1, 0], xyz, pattern, channel_labels, "c. Comp. 1 pattern @ ~400 ms")
    axes[1, 1].plot(centers_ms, p_comp0, color="k", lw=1.4)
    axes[1, 1].axhline(0.05, color="crimson", ls="--", lw=1, label="p=0.05")
    axes[1, 1].axvline(400.0, color="0.5", ls=":", lw=1)
    axes[1, 1].set_ylim(-0.02, 1.02)
    axes[1, 1].set_xlabel("window center (ms)")
    axes[1, 1].set_ylabel("uncorrected p (component 1)")
    axes[1, 1].set_title("d. Component-1 p(t)")
    axes[1, 1].legend(fontsize=8, loc="upper right")
    fig.suptitle("N170 meaning scan (ERP CORE subject 1, 28 scalp, paper Gram)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_fig08(
    path: Path,
    *,
    windows: list[dict[str, Any]],
    times_full_ms: NDArray[np.floating],
    labels: Sequence[str],
    xyz,
    channel_labels,
) -> None:
    plt = _try_pyplot()
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(windows)
    fig, axes = plt.subplots(3, n, figsize=(4.0 * n, 9.0))
    if n == 1:
        axes = np.array(axes).reshape(3, 1)
    for col, win in enumerate(windows):
        _draw_topo(
            axes[0, col],
            xyz,
            win["pattern"],
            channel_labels,
            f"center {win['center_ms']:.0f} ms",
        )
        traces = np.asarray(win["traces_full"], dtype=np.float64)
        for cond, name in enumerate(labels):
            axes[1, col].plot(
                times_full_ms,
                traces[cond],
                color=CONDITION_COLORS.get(name, "k"),
                label=name,
                lw=1.1,
            )
        lo = win["center_ms"] - win["duration_ms"] / 2.0
        hi = win["center_ms"] + win["duration_ms"] / 2.0
        axes[1, col].axvspan(lo, hi, color="0.85", zorder=0)
        axes[1, col].set_xlabel("ms")
        axes[1, col].set_title("component 1 traces")
        if col == 0:
            axes[1, col].legend(fontsize=7, loc="upper right")
        _draw_rdm(
            axes[2, col],
            win["empirical_rdm"],
            labels,
            f"empirical RDM  corr={win['rdm_corr']:.3f}",
        )
    fig.suptitle("Fig 8 analog: three adjacent T=150 ms windows ~400 ms", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_component_panel(
    path: Path,
    *,
    title: str,
    patterns: Sequence[NDArray[np.floating]],
    traces_list: Sequence[NDArray[np.floating]],
    empirical_rdms: Sequence[NDArray[np.floating]],
    corrs: Sequence[float],
    p_values: Sequence[float | None],
    times_ms: NDArray[np.floating],
    labels: Sequence[str],
    xyz,
    channel_labels,
    window_lo: float,
    window_hi: float,
) -> None:
    plt = _try_pyplot()
    if plt is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(patterns)
    fig, axes = plt.subplots(3, n, figsize=(4.2 * n, 9.0))
    if n == 1:
        axes = np.array(axes).reshape(3, 1)
    for col in range(n):
        ptxt = ""
        if p_values[col] is not None:
            ptxt = f"  p={p_values[col]:.3f}"
        _draw_topo(
            axes[0, col],
            xyz,
            patterns[col],
            channel_labels,
            f"comp {col + 1}{ptxt}",
        )
        traces = np.asarray(traces_list[col], dtype=np.float64)
        for cond, name in enumerate(labels):
            axes[1, col].plot(
                times_ms,
                traces[cond],
                color=CONDITION_COLORS.get(name, "k"),
                label=name,
                lw=1.1,
            )
        axes[1, col].axvspan(window_lo, window_hi, color="0.85", zorder=0)
        axes[1, col].set_xlabel("ms")
        if col == 0:
            axes[1, col].legend(fontsize=7)
        _draw_rdm(
            axes[2, col],
            empirical_rdms[col],
            labels,
            f"empirical RDM  corr={corrs[col]:.3f}",
        )
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
