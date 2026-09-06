"""Load Kozunov/AIRI MEG run-1 files from the gitignored OSF cache."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat


def load_meg_ad_run1(path: Path) -> dict[str, Any]:
    """Load ``MEG_AD_run1.mat`` (MATLAB v7.3).

    Returned ``data`` has MATLAB layout ``(207, 1501, 880)``.
    """
    with h5py.File(path, "r") as handle:
        stored = np.array(handle["d"], dtype=np.float64)
    data = np.transpose(stored, (2, 1, 0))
    if data.shape != (207, 1501, 880):
        raise ValueError(f"Unexpected MEG data shape {data.shape}; expected (207, 1501, 880)")
    return {
        "path": str(path),
        "data": data,
        "n_channels": 207,
        "n_times": 1501,
        "n_trials": 880,
        "fs": 1000.0,
        "time_onset_s": -0.5,
    }


def load_spm_trial_labels(path: Path) -> list[str]:
    """Load trial labels from ``ibfctfprespm8_AD_run1_raw_tsss_mc.mat``."""
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    trials = payload["D"].trials
    return [str(trial.label) for trial in trials]


def airi_condition_indices(labels: list[str]) -> dict[str, np.ndarray]:
    """Reconstruct AIRI ``idxTrial`` selection from 3-character SPM labels."""
    t = np.array([[int(s[0]), int(s[1]), int(s[2])] for s in labels], dtype=int)
    valid = (t[:, 0] == 1) | ((t[:, 0] == 2) & (t[:, 1] == 0))
    masks = {
        "face1": valid & (t[:, 1] == 5) & (t[:, 2] == 1),
        "face2": valid & (t[:, 1] == 6) & (t[:, 2] == 1),
        "tool1": valid & (t[:, 1] == 7) & (t[:, 2] == 1),
        "tool2": valid & (t[:, 1] == 8) & (t[:, 2] == 1),
        "nons1": valid & (t[:, 1] == 0),
        "nons2": valid & (t[:, 1] == 9),
    }
    return {name: np.flatnonzero(mask) for name, mask in masks.items()}
