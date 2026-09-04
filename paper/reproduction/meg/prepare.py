"""Load Kozunov/AIRI MEG run-1 planars and build condition averages.

Paper-faithful path: full −500…+1000 ms (1501 samples), 204 planars, no
AIRI bandpass.

AIRI-executable path: MATLAB ``trange = 600:1500`` → 99–999 ms (901 samples)
and ``butter(3) 0.25–20 Hz filtfilt`` via ``source_faithful.airi_bandpass_trials``.

Do not load the companion ``.dat`` (AIRI only reads SPM trial labels from the
``.mat``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from common.meg_io import airi_condition_indices, load_meg_ad_run1, load_spm_trial_labels
from common.paths import MEG_DIR
from common.source_faithful import AIRI_FILTER, AIRI_TRANGE_1BASED, airi_bandpass_trials

from .rdms import CONDITION_NAMES

N_PLANARS = 204
N_TIMES_FILE = 1501
N_TRIALS_FILE = 880
N_CONDITIONS = 6
EPOCHS_PER_CONDITION = 80
FS_HZ = 1000.0
TIME_ONSET_MS = -500.0
# MATLAB 600:1500 inclusive on 1-based indices → Python slice [599:1500].
AIRI_SLICE = slice(AIRI_TRANGE_1BASED[0] - 1, AIRI_TRANGE_1BASED[1])


@dataclass(frozen=True)
class MegBundle:
    """In-memory MEG run used by both labeled paths.

    ``planars`` is unfiltered 204 × 1501 × 880. Filtering is an AIRI-path
    step and is not applied here.
    """

    planars: NDArray[np.float64]
    indices: dict[str, NDArray[np.intp]]
    time_ms: NDArray[np.float64]
    fs: float
    source_mat: str
    labels_mat: str
    n_file_trials: int
    n_used_trials: int
    unused_trial_count: int


def meg_paths() -> tuple[Path, Path]:
    data = MEG_DIR / "MEG_AD_run1.mat"
    labels = MEG_DIR / "ibfctfprespm8_AD_run1_raw_tsss_mc.mat"
    if not data.exists() or not labels.exists():
        raise FileNotFoundError(
            "MEG OSF cache missing. Expected "
            f"{data} and {labels}. Run: python paper/reproduction/common/download_osf.py meg-sensor"
        )
    return data, labels


def time_vector_ms(n_times: int = N_TIMES_FILE, *, onset_ms: float = TIME_ONSET_MS, fs: float = FS_HZ) -> NDArray[np.float64]:
    """True SPM time axis: 1 kHz from −500 ms through +1000 ms (1501 samples)."""
    return onset_ms + np.arange(n_times, dtype=np.float64) * (1000.0 / fs)


def airi_time_ms(time_ms: NDArray[np.floating] | None = None) -> NDArray[np.float64]:
    """Crop the true time axis to AIRI ``600:1500`` (99…999 ms)."""
    if time_ms is None:
        time_ms = time_vector_ms()
    cropped = np.asarray(time_ms, dtype=np.float64)[AIRI_SLICE]
    if cropped.size != 901:
        raise RuntimeError(f"AIRI crop has {cropped.size} samples; expected 901")
    return cropped


def load_meg_bundle() -> MegBundle:
    """Load planars + AIRI trial index sets. Does not bandpass."""
    data_path, labels_path = meg_paths()
    payload = load_meg_ad_run1(data_path)
    labels = load_spm_trial_labels(labels_path)
    indices = airi_condition_indices(labels)
    _validate_indices(indices)
    planars = np.asarray(payload["data"][:N_PLANARS], dtype=np.float64)
    if planars.shape != (N_PLANARS, N_TIMES_FILE, N_TRIALS_FILE):
        raise ValueError(f"Unexpected planar shape {planars.shape}")
    n_used = int(sum(idx.size for idx in indices.values()))
    return MegBundle(
        planars=planars,
        indices=indices,
        time_ms=time_vector_ms(),
        fs=float(payload["fs"]),
        source_mat=str(data_path),
        labels_mat=str(labels_path),
        n_file_trials=N_TRIALS_FILE,
        n_used_trials=n_used,
        unused_trial_count=N_TRIALS_FILE - n_used,
    )


def _validate_indices(indices: dict[str, NDArray[np.intp]]) -> None:
    if tuple(indices) != CONDITION_NAMES:
        raise ValueError(f"Unexpected condition key order {list(indices)}")
    for name, idx in indices.items():
        if idx.size != EPOCHS_PER_CONDITION:
            raise ValueError(f"{name} has {idx.size} trials; expected {EPOCHS_PER_CONDITION}")


def used_trial_indices(indices: dict[str, NDArray[np.intp]]) -> NDArray[np.intp]:
    """Concatenate the six 80-epoch sets in condition order (480 trials)."""
    return np.concatenate([indices[name] for name in CONDITION_NAMES]).astype(np.intp)


def trial_condition_labels(indices: dict[str, NDArray[np.intp]]) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Return ``(used_idx, labels)`` with labels in ``0…5`` matching ``used_idx``."""
    used = used_trial_indices(indices)
    labels = np.concatenate(
        [np.full(indices[name].size, c, dtype=np.intp) for c, name in enumerate(CONDITION_NAMES)]
    )
    return used, labels


def condition_averages(
    planars: NDArray[np.floating],
    indices: dict[str, NDArray[np.intp]],
    *,
    time_slice: slice | None = None,
) -> NDArray[np.float64]:
    """Mean over the 80 epochs of each subcategory.

    Returns ``(6, 204, T)``.
    """
    data = np.asarray(planars, dtype=np.float64)
    if time_slice is not None:
        data = data[:, time_slice, :]
    n_times = data.shape[1]
    averages = np.empty((N_CONDITIONS, N_PLANARS, n_times), dtype=np.float64)
    for c, name in enumerate(CONDITION_NAMES):
        averages[c] = data[:, :, indices[name]].mean(axis=2)
    return averages


def extract_used_trials(
    planars: NDArray[np.floating],
    indices: dict[str, NDArray[np.intp]],
    *,
    time_slice: slice | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Return ``(channels, times, 480)`` used trials and 0…5 labels in the same order."""
    used, labels = trial_condition_labels(indices)
    data = np.asarray(planars, dtype=np.float64)[:, :, used]
    if time_slice is not None:
        data = data[:, time_slice, :]
    return np.asarray(data, dtype=np.float64), labels


def bandpass_airi(planars: NDArray[np.floating]) -> NDArray[np.float64]:
    """AIRI ``butter(3,[0.25,20]/500); filtfilt`` per trial via source_faithful.

    SciPy ``filtfilt`` is not a bit-exact MATLAB Signal Processing Toolbox
    substitute (padding / Gustafsson method). This is the labeled AIRI path.
    """
    return airi_bandpass_trials(
        np.asarray(planars, dtype=np.float64),
        low_hz=float(AIRI_FILTER["low_hz"]),
        high_hz=float(AIRI_FILTER["high_hz"]),
        fs=float(AIRI_FILTER["fs"]),
        order=int(AIRI_FILTER["order"]),
    )


def airi_channel_time_std(
    planars: NDArray[np.floating],
    *,
    ddof: int = 1,
) -> NDArray[np.float64]:
    """MATLAB ``std(d, 0, 3)``: sample SD over the file's 880 trials.

    Shape ``(n_channels, n_times)``. AIRI then divides every trial by this
    map before the Nmc=100 half-split (D5b). SPoC itself is fit on
    unnormalized condition averages.
    """
    data = np.asarray(planars, dtype=np.float64)
    return np.std(data, axis=2, ddof=ddof)


def prepare_provenance(bundle: MegBundle) -> dict[str, Any]:
    t_airi = airi_time_ms(bundle.time_ms)
    return {
        "source_mat": bundle.source_mat,
        "labels_mat": bundle.labels_mat,
        "n_planars": N_PLANARS,
        "n_times_file": int(bundle.time_ms.size),
        "n_file_trials": bundle.n_file_trials,
        "n_used_trials": bundle.n_used_trials,
        "unused_trial_count": bundle.unused_trial_count,
        "epochs_per_condition": EPOCHS_PER_CONDITION,
        "condition_order": list(CONDITION_NAMES),
        "fs_hz": bundle.fs,
        "paper_window_ms": [float(bundle.time_ms[0]), float(bundle.time_ms[-1])],
        "paper_n_samples": int(bundle.time_ms.size),
        "airi_trange_1based": list(AIRI_TRANGE_1BASED),
        "airi_slice_0based": [AIRI_SLICE.start, AIRI_SLICE.stop],
        "airi_window_ms": [float(t_airi[0]), float(t_airi[-1])],
        "airi_n_samples": int(t_airi.size),
        "paper_printed_matrix": "204 x 1500 (D16: file is 1501 samples; paper_faithful uses 1501)",
        "dat_file_loaded": False,
        "airi_filter": dict(AIRI_FILTER),
        "scipy_filtfilt_vs_matlab": (
            "AIRI MATLAB uses Signal Processing Toolbox filtfilt (Gustafsson "
            "method / MATLAB padding). This reconstruction uses scipy.signal.filtfilt "
            "with default padtype='odd' and padlen=3*(max(len(a),len(b))-1). "
            "Not bit-exact (D8 numeric)."
        ),
    }
