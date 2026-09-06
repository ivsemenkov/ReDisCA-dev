"""External preprocessing used by Stage A pipelines.

These routines are independent of ReDisCA constructor settings.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, filtfilt, sosfilt, sosfiltfilt

AIRI_TRANGE_1BASED = (600, 1500)
AIRI_SLICE = slice(AIRI_TRANGE_1BASED[0] - 1, AIRI_TRANGE_1BASED[1])
AIRI_FILTER = {"order": 3, "low_hz": 0.25, "high_hz": 20.0, "fs": 1000.0}


def airi_bandpass_trials(
    trials: NDArray[np.floating],
    *,
    low_hz: float = 0.25,
    high_hz: float = 20.0,
    fs: float = 1000.0,
    order: int = 3,
) -> NDArray[np.float64]:
    """AIRI ``butter(3,[low,high]/(fs/2)); filtfilt`` per channel/trial.

    ``trials`` shape ``(n_channels, n_times, n_trials)``. SciPy ``filtfilt``
    is not MATLAB Signal Processing Toolbox bitwise parity.
    """
    trials = np.asarray(trials, dtype=np.float64)
    if trials.ndim != 3:
        raise ValueError("trials must have shape (n_channels, n_times, n_trials)")
    b, a = butter(order, [low_hz, high_hz], btype="bandpass", fs=fs)
    filtered = np.empty_like(trials)
    n_channels, _, n_trials = trials.shape
    for trial in range(n_trials):
        for channel in range(n_channels):
            filtered[channel, :, trial] = filtfilt(b, a, trials[channel, :, trial])
    return filtered


def butterworth_lowpass_rows(
    rows: NDArray[np.floating],
    *,
    cutoff_hz: float,
    fs_hz: float,
    order: int,
    zero_phase: bool,
) -> NDArray[np.float64]:
    """Filter rows of a ``(n, T)`` array with an Nth-order Butterworth LPF."""
    rows = np.asarray(rows, dtype=np.float64)
    sos = butter(order, cutoff_hz, btype="low", fs=fs_hz, output="sos")
    if zero_phase:
        return np.asarray(sosfiltfilt(sos, rows, axis=-1), dtype=np.float64)
    return np.asarray(sosfilt(sos, rows, axis=-1), dtype=np.float64)
